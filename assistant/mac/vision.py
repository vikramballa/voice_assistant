"""Seeing the screen: a screenshot of the window the user is working in, read two ways.

- Apple's text recognition (Vision framework, ~0.1 s): where a piece of text is, to click it.
- A small vision-language model (Qwen3-VL 2B, ~2 s): answers questions about the screen and points at
  things by description ("the video with red and blue lines").
- A GUI-agent model (Mano-CUA-2.0-4B, same Qwen3-VL family): where to click for click()'s last-resort
  step (the general model guessed wrong on ordinals and icons), and the next action of a task done on the
  screen step by step (`next_action`, used by actions/agent.py). Loaded on first use, not at startup, so
  normal commands never pay its weight; if it can't load (out of memory, or on a server where no
  transformers release of it exists), `locate_click` returns None and `next_action` raises.
- The Mac's camera (`photo`, OpenCV): one picture when the user asks ("what am I holding?"), deleted after use.

Model backend: MLX (Metal, fast) on this Mac when `mlx_vlm` is installed; otherwise plain `transformers`
+ torch (CPU or CUDA) — e.g. on a Linux server. `backend()` picks one; `ask`/`point`/`locate_click` are
the same either way. The screen-capture / Accessibility-OCR functions below (`capture`, `read_text`,
`find_text`, `has_access`) stay macOS-only (there is no "screen" on a headless server) and only import
Quartz/Vision/Foundation when actually called, so this module still imports cleanly off a Mac — e.g. to
run just the model behind `assistant.vision_service` (a small HTTP wrapper for deploying it elsewhere).

Needs the Screen Recording permission for the app Laya runs in.

Test from the terminal:
    python -m assistant.mac.vision "what is on the screen?"
    python -m assistant.mac.vision --image shot.png "point to the search box"
"""

import difflib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import transformers

transformers.logging.set_verbosity_error()  # mlx-vlm passes kwargs the way transformers now warns about

from assistant.gpu import MLX_LOCK

ON_MAC = platform.system() == "Darwin"

MODEL = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
AGENT_MODEL = "Mininglamp-2718/Mano-CUA-2.0-4B-MLX-8bit"
TRANSFORMERS_MODEL = "Qwen/Qwen3-VL-2B-Instruct"              # off a Mac: plain HF weights, no MLX
TRANSFORMERS_AGENT_MODEL = "Mininglamp-2718/Mano-CUA-2.0-4B"  # same org's non-MLX release, if published
IMAGE_WIDTH = 1024  # pointing: 8/8 targets right at 1024 px wide, ~2 s; full size (1440) took ~4 s
ASK_WIDTH = 768     # questions: ~1.1 s instead of ~2.5 s; describing doesn't need the detail
AGENT_WIDTH = 1024  # measured 2026-10-11: 1280 (the model card's suggestion) took ~5.6-8.6 s warm for the
                    # same answer as 1024 at ~4 s; same point on both real click() test cases at 1024
_vlm: dict = {}
_agent: dict = {}


class NoScreenAccess(Exception):
    """macOS refused the screenshot (Screen Recording permission missing)."""


class NoCameraAccess(Exception):
    """No picture from the camera (Camera permission missing, or the camera is in use / closed lid)."""


def backend() -> str:
    """"mlx" on this Mac when mlx_vlm is installed (fast, Metal); "transformers" anywhere else (CPU/CUDA)."""
    if ON_MAC:
        try:
            import mlx_vlm  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "transformers"


def has_access() -> bool:
    import Quartz
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def capture(window_id: int | None = None) -> Path:
    """PNG of one window (by window-server number) or of the whole screen."""
    path = Path(tempfile.gettempdir()) / f"laya-screen-{time.time_ns()}.png"
    cmd = ["screencapture", "-x", "-o"] + (["-l", str(window_id)] if window_id else []) + [str(path)]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode or not path.exists():
        raise NoScreenAccess()
    return path


CAMERA_WARMUP_FRAMES = 15  # the first frames are dark while the camera sets its exposure (~0.5 s)


def photo() -> Path:
    """One picture from the Mac's camera (its green light shows while it is taken)."""
    import cv2
    camera = cv2.VideoCapture(0)
    frame = None
    try:
        for _ in range(CAMERA_WARMUP_FRAMES if camera.isOpened() else 0):
            ok, grabbed = camera.read()
            frame = grabbed if ok else frame
    finally:
        camera.release()
    if frame is None:
        raise NoCameraAccess()
    path = Path(tempfile.gettempdir()) / f"laya-camera-{time.time_ns()}.png"
    cv2.imwrite(str(path), frame)
    return path


def discard(image: Path) -> None:
    """Delete a screenshot and its resized copies: pictures of the user's screen must not pile up in /tmp."""
    for path in [image, *image.parent.glob(f"{image.stem}-*.png")]:
        path.unlink(missing_ok=True)


# --- text on screen ----------------------------------------------------------------------------

def read_text(image: Path) -> list[tuple[str, object, object]]:
    """[(line text, VNRecognizedText, bounding box normalized with origin bottom-left)]. macOS only."""
    import Quartz
    import Vision
    from Foundation import NSURL
    source = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(image)), None)
    cg = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)  # names and buttons, not prose
    Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None).performRequests_error_([request], None)
    lines = []
    for result in request.results() or []:
        best = result.topCandidates_(1)[0]
        lines.append((best.string(), best, result.boundingBox()))
    return lines


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def find_text(image: Path, name: str) -> tuple[float, float] | None:
    """Centre of the text best matching `name`, as fractions of the image (0-1, origin top-left). macOS only."""
    from Foundation import NSMakeRange
    want = _norm(name)
    best, best_score = None, 0.0
    for text, candidate, box in read_text(image):
        have = _norm(text)
        if not have:
            continue
        if have == want:
            score = 1.0
        elif re.search(rf"\b{re.escape(want)}\b", have):
            score = 0.9  # the name inside a longer line: use the box of just those characters
            start = text.lower().find(name.lower().strip())
            if start >= 0:
                part, _ = candidate.boundingBoxForRange_error_(NSMakeRange(start, len(name.strip())), None)
                if part is not None:
                    box = part.boundingBox()
        else:
            score = difflib.SequenceMatcher(None, want, have).ratio() * 0.85
        if score > best_score:
            best, best_score = box, score
    if best is None or best_score < 0.75:
        return None
    return best.origin.x + best.size.width / 2, 1 - (best.origin.y + best.size.height / 2)


def text_at(image: Path, spot: tuple[float, float], reach: float = 0.12) -> str:
    """The text at a point (fractions, origin top-left): what a click there would press; else the nearest line
    within `reach` (an empty box's placeholder or label: "Search products", "Password"). macOS only."""
    x, y = spot[0], 1 - spot[1]
    lines = [(math.hypot(max(box.origin.x - x, 0, x - box.origin.x - box.size.width),
                         max(box.origin.y - y, 0, y - box.origin.y - box.size.height)), text)
             for text, _, box in read_text(image)]
    inside = [text for distance, text in lines if distance == 0]
    near = min(((d, t) for d, t in lines if d <= reach), default=(0, ""))[1]
    return " ".join(inside) or near


# --- vision-language model ---------------------------------------------------------------------

def _load(agent: bool) -> dict:
    """Load the general model or (agent=True) the GUI-grounding model, on whichever backend fits this
    machine. Returns a cache entry for `_run`."""
    if backend() == "mlx":
        from mlx_vlm import load
        with MLX_LOCK:
            model, processor = load(AGENT_MODEL if agent else MODEL)
        return {"backend": "mlx", "model": model, "processor": processor}
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    name = TRANSFORMERS_AGENT_MODEL if agent else TRANSFORMERS_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(name, dtype="auto").to(device).eval()
    processor = AutoProcessor.from_pretrained(name)
    return {"backend": "transformers", "model": model, "processor": processor, "device": device}


def warm_up() -> None:
    """Load the model (~1.5 s, 1.8 GB on MLX) and the text recognizer (its first use ever takes ~25 s)."""
    if not _vlm:
        _vlm.update(_load(agent=False))
    blank = Path(tempfile.gettempdir()) / "laya-blank.png"
    if not blank.exists():
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (200, 60), "white")
        ImageDraw.Draw(img).text((10, 20), "Laya", fill="black")
        img.save(blank)
    if ON_MAC:
        read_text(blank)
    if not _vlm.get("ready"):  # the first answer ever is slow (~8 s more); do it now, not for the user
        _generate(blank, "Say ok.", 1)
        _vlm["ready"] = True


def _small(image: Path, width: int = IMAGE_WIDTH) -> Path:
    from PIL import Image
    img = Image.open(image).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    out = image.with_name(f"{image.stem}-{width}.png")
    img.save(out)
    return out


def _run(entry: dict, image: Path, prompt: str, max_tokens: int) -> str:
    """Run one prompt+image through a loaded model (an entry from `_load`)."""
    if entry["backend"] == "mlx":
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor = entry["model"], entry["processor"]
        text = apply_chat_template(processor, model.config, prompt, num_images=1)
        with MLX_LOCK:
            out = generate(model, processor, text, [str(image)], max_tokens=max_tokens, temperature=0.0,
                           verbose=False)
        return getattr(out, "text", out).strip()
    import torch
    from PIL import Image as PILImage
    model, processor, device = entry["model"], entry["processor"], entry["device"]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[PILImage.open(image).convert("RGB")], return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def _generate(image: Path, prompt: str, max_tokens: int, width: int = IMAGE_WIDTH) -> str:
    if not _vlm:
        warm_up()
    return _run(_vlm, _small(image, width), prompt, max_tokens)


def ask(image: Path, question: str, what: str = "a screenshot of the user's screen") -> str:
    """A short spoken answer about the picture (`what` says what it is: a screenshot, a camera photo)."""
    answer = _generate(image, f"This is {what}. {question}\n"
                              "Answer in one or two short sentences that sound natural when spoken. No lists.", 80,
                       ASK_WIDTH)
    return re.sub(r"[*#`]+", "", answer).strip()


def point(image: Path, description: str) -> tuple[float, float] | None:
    """Where `description` is, as fractions of the image (0-1, origin top-left), or None."""
    out = _generate(image, f'Point to {description}. Answer only with JSON {{"point":[x,y]}}.', 40)
    m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", out)
    if not m:
        return None
    x, y = float(m.group(1)) / 1000, float(m.group(2)) / 1000  # Qwen3-VL answers in 0-1000 of the image
    return (x, y) if 0 <= x <= 1 and 0 <= y <= 1 else None


ACTION = re.compile(r"<action>(.*?)(?:</action>|$)", re.S)
ONE_ACTION = re.compile(r"(\w+)\(((?:[^()']|'(?:[^'\\]|\\.)*')*)\)")  # click(start_box='(1,2)'), parens in quotes
ACTION_ARG = re.compile(r"(\w+)='((?:[^'\\]|\\.)*)'")
POINT = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")  # inside start_box='...', sometimes wrapped in <|box_start|>
AGENT_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to \
perform the next action to complete the task.

## Output Format
<action>action</action>

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
doubleclick(start_box='<|box_start|>(x1,y1)<|box_end|>')
type(content='') # type the content.
hotkey(key='') # Trigger a keyboard shortcut.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left', amount='scroll_amount')
wait(duration='') # Sleep for specified duration (in seconds).
finish() # The task is completed.
stop(reason='') # If the item can not found in the image, give the reason

## User Instruction
{task}"""


def _agent_run(image: Path, prompt: str, max_tokens: int) -> str:
    """The GUI-agent model's answer (loaded on first use; raises if it can't be)."""
    if not _agent:
        _agent.update(_load(agent=True))
    return _run(_agent, _small(image, AGENT_WIDTH), prompt, max_tokens)


def _spot(text: str) -> tuple[float, float] | None:
    m = POINT.search(text)
    if not m:
        return None
    x, y = int(m.group(1)) / 1000, int(m.group(2)) / 1000  # the model answers in 0-1000 of the image
    return (x, y) if 0 <= x <= 1 and 0 <= y <= 1 else None


def locate_click(image: Path, description: str) -> tuple[float, float] | None:
    """Where to click for `description`, as fractions of the image (0-1) — using Mano-CUA, a model trained
    specifically to ground GUI instructions to a click point (more reliable than the general vision model
    on ordinals/icons). None if the model can't load or finds nothing."""
    try:
        out = _agent_run(image, "You are a GUI agent. Find this element on the screenshot, then answer with exactly "
                                "one line:\n<action>click(start_box='(x,y)')</action>\nwhere x and y are its centre, "
                                f"0-1000 scale.\nElement: {description}", 40)
    except Exception:
        return None
    m = re.search(r"start_box='([^']*)'", out)
    return _spot(m.group(1)) if m else None


def next_actions(image: Path, task: str, done: list[str]) -> list[dict]:
    """The next steps of `task` on this screenshot, after the steps in `done` ("Click Pricing"): usually one,
    sometimes a few ("click the box, type shoes"): [{"name": "click", "spot": (x, y) or None, "args": {...}}].
    [] if its answer has no action; raises if the model can't be loaded."""
    history = "".join(f"\n{i}. {step}" for i, step in enumerate(done, 1))
    prompt = AGENT_PROMPT.format(task=task) + (f"\n\n## Action History{history}" if history else "")
    m = ACTION.search(_agent_run(image, prompt, 80))  # it writes no description of its own: agent.py makes one
    actions = []
    for name, inner in ONE_ACTION.findall(m.group(1) if m else ""):
        args = {key: re.sub(r"\\(.)", r"\1", value) for key, value in ACTION_ARG.findall(inner)}
        actions.append({"name": name, "spot": _spot(args.get("start_box", "")), "args": args})
    return actions


if __name__ == "__main__":
    args = sys.argv[1:]
    shot = Path(args[1]) if args[:1] == ["--image"] else capture()
    question = " ".join(args[2:] if args[:1] == ["--image"] else args) or "What is on the screen?"
    t = time.perf_counter()
    if question.lower().startswith("point to "):
        print(point(shot, question[9:]))
    else:
        print(ask(shot, question))
    print(json.dumps({"ms": round((time.perf_counter() - t) * 1000)}))
