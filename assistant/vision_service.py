"""Small HTTP wrapper around the vision model (ask / point / locate_click), for running it off the Mac —
e.g. on a Linux server (Contabo): heavier compute there, images sent over the network from wherever needs
them. Only needs assistant.mac.vision's model functions, not the macOS-only screen capture / Accessibility
OCR, so this runs fine on a plain Linux box (CPU or CUDA) via the `transformers` backend in vision.py.

Run:    python -m assistant.vision_service          (port 8000, or $PORT; set $VISION_HOST to bind elsewhere)
Call:   curl -F image=@shot.png -F question="What is on the screen?" http://host:8000/ask
        curl -F image=@shot.png -F description="the search box" http://host:8000/point
        curl -F image=@shot.png -F description="the second video" http://host:8000/locate_click

Not authenticated: put it behind a reverse proxy / VPN, or a token header via your own proxy, if it is
reachable from outside a trusted network — it runs model inference on whatever image it is given.
"""

import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from assistant.mac import vision

app = FastAPI(title="laya-vision")


def _save(image: UploadFile) -> Path:
    path = Path(tempfile.gettempdir()) / f"vision-service-{os.urandom(8).hex()}.png"
    path.write_bytes(image.file.read())
    return path


@app.get("/health")
def health() -> dict:
    return {"ok": True, "backend": vision.backend()}


@app.on_event("startup")
def _warm_up() -> None:
    vision.warm_up()


@app.post("/ask")
async def ask_endpoint(image: UploadFile = File(...), question: str = Form("What is on the screen?")) -> dict:
    path = _save(image)
    t = time.perf_counter()
    try:
        return {"answer": vision.ask(path, question), "ms": round((time.perf_counter() - t) * 1000)}
    finally:
        path.unlink(missing_ok=True)


@app.post("/point")
async def point_endpoint(image: UploadFile = File(...), description: str = Form(...)) -> dict:
    path = _save(image)
    t = time.perf_counter()
    try:
        return {"point": vision.point(path, description), "ms": round((time.perf_counter() - t) * 1000)}
    finally:
        path.unlink(missing_ok=True)


@app.post("/locate_click")
async def locate_click_endpoint(image: UploadFile = File(...), description: str = Form(...)) -> dict:
    path = _save(image)
    t = time.perf_counter()
    try:
        return {"point": vision.locate_click(path, description), "ms": round((time.perf_counter() - t) * 1000)}
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=os.environ.get("VISION_HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
