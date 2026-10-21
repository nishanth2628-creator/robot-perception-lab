"""Generate labelled scenes without external downloads or hidden assets."""

from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ("wall", "floor", "obstacle")
PALETTE = np.array([[145, 174, 196], [74, 179, 124], [237, 145, 71]], dtype=np.uint8)
HEIGHT, WIDTH = 64, 96


@dataclass
class Scene:
    scene_id: str
    seed: int
    domain: str
    image: np.ndarray
    labels: np.ndarray


def scene(seed: int, domain: str = "standard") -> Scene:
    """Render a deliberately simple indoor scene and exact class labels.

    Shifted scenes have darker floor/walls and pale low-saturation obstacles.
    They expose reliance on synthetic colour cues; they are not a real-camera test.
    """
    if domain not in {"standard", "shifted"}:
        raise ValueError("domain must be standard or shifted")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    horizon = int(rng.integers(17, 26))
    slope = float(rng.uniform(-0.035, 0.035))
    floor = yy >= horizon + slope * (xx - WIDTH / 2)
    labels = floor.astype(np.uint8)
    wall_color = np.array([211, 220, 225]) + rng.uniform(-14, 14, 3)
    floor_color = np.array([162, 149, 127]) + rng.uniform(-13, 13, 3)
    if domain == "shifted":
        wall_color *= rng.uniform(0.57, 0.73)
        floor_color *= rng.uniform(0.55, 0.72)
    rgb = np.where(floor[..., None], floor_color, wall_color).astype(float)
    rgb += rng.normal(0, 4.5, rgb.shape)
    rgb += (yy / HEIGHT - 0.5)[..., None] * rng.uniform(5, 22)
    # Mild floor tile marks remain floor, never obstacles.
    seams = floor & (((xx + (yy // 16) * 11) % 31 == 0) | (yy % 17 == 0))
    rgb[seams] -= 9
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    mask = Image.fromarray(labels)
    draw, lab = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    for _ in range(int(rng.integers(1, 5))):
        bottom = int(rng.integers(37, HEIGHT - 1))
        width = int(rng.integers(9, 23))
        height = int(rng.integers(10, 27))
        left = int(rng.integers(1, WIDTH - width - 4))
        top = max(horizon - 2, bottom - height)
        depth = int(rng.integers(2, 6))
        color = np.array(([62, 93, 143], [143, 70, 61], [74, 118, 87])[int(rng.integers(3))])
        color = color + rng.uniform(-18, 18, 3)
        if domain == "shifted":
            color = np.array([173, 170, 161]) + rng.uniform(-18, 18, 3)
        front = [(left, top), (left + width, top), (left + width, bottom), (left, bottom)]
        top_face = [(left, top), (left + depth, top - depth),
                    (left + width + depth, top - depth), (left + width, top)]
        side = [(left + width, top), (left + width + depth, top - depth),
                (left + width + depth, bottom - depth), (left + width, bottom)]
        for points, lighting in [(front, 1.0), (top_face, 1.12), (side, 0.82)]:
            draw.polygon(points, fill=tuple(np.clip(color * lighting, 0, 255).astype(int)))
            lab.polygon(points, fill=2)
    result = np.asarray(image).copy().astype(float)
    result += rng.normal(0, 2.0, result.shape)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Scene(f"{domain}-{seed}", seed, domain, result, np.asarray(mask).copy())


def make_splits():
    """Split at scene level before selecting pixels; seeds never overlap."""
    specifications = {"train": (1000, 80, "standard"),
                      "validation": (2000, 20, "standard"),
                      "test": (3000, 30, "standard"),
                      "domain_shift": (4000, 30, "shifted"),
                      "style_train": (5000, 80, "shifted"),
                      "style_validation": (6000, 20, "shifted")}
    return {name: [scene(start + i, domain) for i in range(count)]
            for name, (start, count, domain) in specifications.items()}


def features(rgb):
    """21 interpretable colour, position and local-edge features per pixel."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 RGB image")
    im = rgb.astype(np.float64) / 255.0
    height, width = im.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    x, y = xx / max(width - 1, 1), yy / max(height - 1, 1)
    r, g, b = im.transpose(2, 0, 1)
    gray = im.mean(axis=2)
    padded = np.pad(gray, 1, mode="edge")
    local_mean = sum(padded[dy:dy + height, dx:dx + width]
                     for dy in range(3) for dx in range(3)) / 9
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    planes = [r, g, b, r*r, g*g, b*b, r*g, r*b, g*b,
              x, y, x*x, y*y, im.max(axis=2) - im.min(axis=2), gray,
              dx, dy, np.abs(gray-local_mean), r*y, g*y, b*y]
    return np.stack(planes, axis=-1).reshape(-1, len(planes))


def sample_pixels(scenes, per_class=320, seed=42):
    """Balance each training scene without sharing pixels across data splits."""
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for item in scenes:
        flat = item.labels.ravel()
        indices = []
        for label in range(len(CLASS_NAMES)):
            available = np.flatnonzero(flat == label)
            indices.extend(rng.choice(available, min(per_class, len(available)), replace=False))
        indices = np.asarray(indices, dtype=int)
        inputs.append(features(item.image)[indices])
        targets.append(flat[indices])
    return np.concatenate(inputs), np.concatenate(targets)
