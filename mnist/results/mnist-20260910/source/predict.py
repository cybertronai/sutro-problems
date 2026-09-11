"""Run a saved baseline checkpoint on the competition test images."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mnist.models import build_model


def predict(checkpoint, dataset, output, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_model(saved["config"], saved["image_size"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with np.load(dataset, allow_pickle=False) as archive:
        images = archive["test_images"]
    if images.shape[1:] != (1, saved["image_size"], saved["image_size"]):
        raise ValueError("Checkpoint resolution does not match dataset")
    norm = saved["normalization"]
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(images), 1024):
            x = torch.from_numpy(images[start:start + 1024]).to(device)
            logits = model((x - norm["mean"]) / norm["std"])
            predictions.append(logits.argmax(1).cpu().numpy())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predictions=np.concatenate(predictions))
    print(f"Saved {len(images)} predictions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    predict(**vars(parser.parse_args()))
