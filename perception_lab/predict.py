"""Inspect one RGB image with the saved model, without claiming real-world validity."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .data import PALETTE
from .model import PixelClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("results/pixel_classifier.npz"))
    parser.add_argument("--output", type=Path, default=Path("prediction.png"))
    args = parser.parse_args()
    # Training resolution is fixed. Other camera geometry is out of scope.
    with Image.open(args.image) as source:
        rgb = np.asarray(source.convert("RGB").resize((96, 64)))
    labels = PixelClassifier.load(args.model).predict(rgb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(PALETTE[labels]).save(args.output)
    print(f"Saved experimental mask: {args.output}")


if __name__ == "__main__":
    main()
