"""Fine-tune Laya on our Mac-command intents, on the Mac's GPU (MPS).

Plain supervised training: for each sentence in train/dataset.jsonl, Laya
scores all 14 intent options and we push the correct one up (cross-entropy).
Only the decision head and the top encoder layers are trained, so it fits
in 16 GB RAM and takes a few minutes.

Afterwards it fits a calibration temperature on held-out sentences, so a
confidence of 0.9 really means ~90% right, then saves a normal Laya
checkpoint to train/laya-mac/ that assistant/brain/fast.py picks up automatically.

Run from the project folder:
    python train/finetune.py
"""

import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from laya.agent import Agent
from laya.common import collate_items
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.brain.fast import INTENTS, QUESTIONS  # noqa: E402

BASE = "convaiinnovations/laya"
OUT = ROOT / "train" / "laya-mac"
TRAIN_TOP_LAYERS = 6    # of ModernBERT-large's 28 encoder layers
EPOCHS = 4
BATCH = 16
LR_HEAD = 1e-4
LR_ENCODER = 2e-5
VAL_FRACTION = 0.1

random.seed(0)
torch.manual_seed(0)

LABELS = list(INTENTS)


def encode(agent: Agent, rows: list[dict]) -> list[dict]:
    internal = {"intent": agent._to_internal(QUESTIONS["intent"])}
    items = []
    for r in rows:
        item = agent._encode_state(r["text"], ["intent"], internal)[0]
        item["label"] = LABELS.index(r["intent"])
        items.append(item)
    return items


def forward(agent: Agent, items: list[dict]):
    b = collate_items([items], pad_id=agent.tok.pad_token_id)
    dev = agent.device
    logits, _ = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                            b["marker_mask"].to(dev), b["qtype"].to(dev))
    return logits, b["label"].to(dev)


@torch.no_grad()
def val_logits(agent: Agent, items: list[dict]):
    agent.model.eval()
    all_logits, all_labels = [], []
    for i in range(0, len(items), 32):
        logits, labels = forward(agent, items[i:i + 32])
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Pick the temperature that makes confidences honest (lowest NLL) on held-out data."""
    best_t, best_nll = 1.0, math.inf
    for t in [x / 20 for x in range(4, 101)]:  # 0.2 .. 5.0
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def main() -> None:
    rows = [json.loads(l) for l in (ROOT / "train" / "dataset.jsonl").read_text().splitlines() if l.strip()]
    random.shuffle(rows)
    n_val = int(len(rows) * VAL_FRACTION)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    print("loading laya...")
    agent = Agent(BASE, device="mps")
    model = agent.model
    print(f"device: {agent.device}")

    # Freeze everything, then unfreeze the head and the top encoder layers.
    for p in model.parameters():
        p.requires_grad = False
    encoder_params = []
    for layer in model.encoder.layers[-TRAIN_TOP_LAYERS:]:
        encoder_params += list(layer.parameters())
    encoder_params += list(model.encoder.final_norm.parameters())
    head_params = list(model.head.parameters()) + list(model.scorer.parameters()) + list(model.type_emb.parameters())
    for p in encoder_params + head_params:
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": head_params, "lr": LR_HEAD},
                             {"params": encoder_params, "lr": LR_ENCODER}], weight_decay=0.01)

    train_items, val_items = encode(agent, train_rows), encode(agent, val_rows)
    print(f"train {len(train_items)}  val {len(val_items)}")

    # Balance the classes a little: rare intents count more.
    counts = torch.tensor([sum(it["label"] == i for it in train_items) for i in range(len(LABELS))], dtype=torch.float)
    class_weight = (counts.mean() / counts.clamp(min=1)).sqrt().to(agent.device)

    logits, labels = val_logits(agent, val_items)
    print(f"before: val accuracy {(logits.argmax(-1) == labels).float().mean():.1%}")

    steps = EPOCHS * math.ceil(len(train_items) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_HEAD, LR_ENCODER], total_steps=steps, pct_start=0.1)
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(train_items)
        total = 0.0
        for i in range(0, len(train_items), BATCH):
            logits, labels = forward(agent, train_items[i:i + BATCH])
            loss = F.cross_entropy(logits, labels, weight=class_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params + head_params, 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(labels)
        logits, labels = val_logits(agent, val_items)
        acc = (logits.argmax(-1) == labels).float().mean()
        print(f"epoch {epoch + 1}/{EPOCHS}: train loss {total / len(train_items):.3f}  "
              f"val accuracy {acc:.1%}  ({time.time() - start:.0f}s)")

    temperature = fit_temperature(logits, labels)
    print(f"calibration temperature: {temperature}")

    # Save as a normal Laya checkpoint: same tokenizer/encoder config, new weights and temperature.
    src = Path(agent.model_id)
    if not src.exists():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(BASE, allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"]))
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(src / "tokenizer", OUT / "tokenizer")
    shutil.copytree(src / "encoder", OUT / "encoder")
    cfg = dict(agent.cfg)
    cfg["temperature"] = [temperature] + list(cfg["temperature"][1:])
    cfg.pop("temperature_by_options", None)  # these would override our fitted value
    cfg["training"] = {"base": BASE, "task": "mac voice intents", "examples": len(train_items),
                       "epochs": EPOCHS, "val_accuracy": round(float(acc), 4)}
    (OUT / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()},
              str(OUT / "model.safetensors"))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
