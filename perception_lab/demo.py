"""Run the complete train/validate/test experiment: python -m perception_lab.demo."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from .data import CLASS_NAMES, PALETTE, features, make_splits, sample_pixels
from .metrics import confusion_matrix, summarize
from .model import PixelClassifier


def evaluate(model, scenes):
    total = np.zeros((3, 3), dtype=np.int64)
    per_scene = []
    for item in scenes:
        matrix = confusion_matrix(item.labels, model.predict(item.image))
        total += matrix
        scores = summarize(matrix)
        per_scene.append({"scene_id": item.scene_id, "mean_iou": scores["mean_iou"],
                          "pixel_accuracy": scores["pixel_accuracy"]})
    result = summarize(total)
    result["scenes"] = len(scenes)
    result["pixels"] = int(total.sum())
    result["per_scene"] = per_scene
    return result


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def comparison(model, scenes, destination, comparison_model=None):
    canvas = Image.new("RGB", (1480, 1155), "#10212d")
    pen = ImageDraw.Draw(canvas)
    pen.text((40, 26), "ROBOT PERCEPTION LAB", font=font(34), fill="#f2f7fa")
    pen.text((40, 76), "A learned NumPy baseline on synthetic camera scenes", font=font(23), fill="#a9c1d1")
    columns = (40, 400, 760, 1120)
    titles = ("Camera input", "Ground truth", "Baseline mask",
              "Style-augmented mask" if comparison_model is not None else "Errors in red")
    for left, title in zip(columns, titles):
        pen.text((left, 127), title, font=font(21), fill="#e5eef4")
    for row, item in enumerate(scenes):
        top = 179 + row * 285
        pred = model.predict(item.image)
        score = summarize(confusion_matrix(item.labels, pred))
        pen.text((40, top - 27), f"{item.scene_id}  |  baseline mean IoU {score['mean_iou']:.3f}",
                 font=font(18), fill="#a9c1d1")
        errors = np.zeros_like(item.image)
        errors[:] = [29, 49, 62]
        errors[pred != item.labels] = [252, 86, 93]
        fourth = PALETTE[comparison_model.predict(item.image)] if comparison_model is not None else errors
        tiles = (item.image, PALETTE[item.labels], PALETTE[pred], fourth)
        for left, pixels in zip(columns, tiles):
            tile = Image.fromarray(pixels).resize((320, 214), Image.Resampling.NEAREST)
            canvas.paste(tile, (left, top))
    legend_y = 1053
    for index, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        left = 40 + index * 180
        pen.rectangle((left, legend_y, left + 20, legend_y + 20), fill=tuple(color.astype(int)))
        pen.text((left + 30, legend_y - 2), name, font=font(19), fill="#e5eef4")
    caption = ("Synthetic-only experiment. Known shifted style is included in the augmented model's training; no safety claim."
               if comparison_model is not None else
               "Synthetic-only experiment. Shifted colours test a known failure mode; no robot safety claim.")
    pen.text((40, 1100), caption,
             font=font(20), fill="#a9c1d1")
    canvas.save(destination)


def learning_curve(history, destination):
    canvas = Image.new("RGB", (1260, 480), "#f8fafc")
    pen = ImageDraw.Draw(canvas)
    pen.text((30, 20), "Training history - validation selects the saved epoch", font=font(25), fill="#10212d")
    panels = [(70, "training_cross_entropy", "Training cross entropy"),
              (700, "validation_mean_iou", "Validation mean IoU")]
    for left, key, title in panels:
        top, width, height = 115, 460, 260
        values = [row[key] for row in history]
        low = 0.0 if key == "training_cross_entropy" else max(0, min(values) - 0.02)
        high = max(values) * 1.05 if key == "training_cross_entropy" else min(1, max(values) + 0.02)
        high = max(high, low + 0.01)
        pen.text((left, 76), title, font=font(21), fill="#10212d")
        for tick in range(5):
            yy = top + height - tick * height / 4
            pen.line((left, yy, left + width, yy), fill="#dce5ec", width=1)
            pen.text((left - 55, yy - 9), f"{low+(high-low)*tick/4:.2f}", font=font(15), fill="#42566a")
        points = [(left + i * width / max(1, len(values)-1), top + height - (v-low)/(high-low)*height)
                  for i, v in enumerate(values)]
        if len(points) > 1:
            pen.line(points, fill="#146d8b", width=4)
        for px, py in points:
            pen.ellipse((px-3, py-3, px+3, py+3), fill="#146d8b")
        pen.text((left, top + height + 15), "epoch 1", font=font(16), fill="#42566a")
        pen.text((left + width - 85, top + height + 15), f"epoch {len(values)}", font=font(16), fill="#42566a")
    canvas.save(destination)


def run(output, epochs=18):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = make_splits()
    train_x, train_y = sample_pixels(splits["train"])
    # Validation uses every pixel; neither test set participates in fitting/selection.
    val_x = np.concatenate([features(s.image) for s in splits["validation"]])
    val_y = np.concatenate([s.labels.ravel() for s in splits["validation"]])
    model = PixelClassifier()
    history = model.fit(train_x, train_y, val_x, val_y, epochs=epochs)
    model.save(output / "pixel_classifier.npz")
    results = {name: evaluate(model, splits[name]) for name in ("validation", "test", "domain_shift")}
    # An intervention, not another unseen-domain claim: add separate training and
    # validation scenes from the known shifted style. All test seeds stay untouched.
    style_x, style_y = sample_pixels(splits["style_train"], seed=43)
    mixed_x, mixed_y = np.concatenate((train_x, style_x)), np.concatenate((train_y, style_y))
    style_val_x = np.concatenate([features(s.image) for s in splits["style_validation"]])
    style_val_y = np.concatenate([s.labels.ravel() for s in splits["style_validation"]])
    augmented = PixelClassifier()
    augmented_history = augmented.fit(mixed_x, mixed_y,
                                      np.concatenate((val_x, style_val_x)),
                                      np.concatenate((val_y, style_val_y)), epochs=epochs)
    augmented.save(output / "style_augmented_classifier.npz")
    results["style_augmentation"] = {
        name: evaluate(augmented, splits[name]) for name in ("test", "domain_shift")}
    results["style_augmentation"]["training"] = {
        "training_scenes": len(splits["train"]) + len(splits["style_train"]),
        "training_pixels_sampled": len(mixed_y),
        "validation_scenes": len(splits["validation"]) + len(splits["style_validation"]),
        "selected_epoch": augmented.best_epoch,
        "seed_additional_pixel_sampling": 43,
        "interpretation": "The known shifted rendering style is now represented in training; test scenes remain held out."}
    results["experiment"] = {
        "training_scenes": len(splits["train"]), "training_pixels_sampled": len(train_y),
        "feature_count": train_x.shape[1], "epochs": epochs,
        "selected_epoch": model.best_epoch, "selection_metric": "validation mean IoU",
        "seed_training": 7, "seed_sampling": 42,
        "batch_size": 4096, "learning_rate": 0.025, "l2_regularization": 0.0005,
        "maximum_sampled_pixels_per_class_per_training_scene": 320,
        "python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "scope": "Synthetic images only; no real robot or real-camera validation."}
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    manifest = {name: [{"scene_id": s.scene_id, "seed": s.seed, "domain": s.domain,
                        "sha256_pixels_and_labels": hashlib.sha256(s.image.tobytes()+s.labels.tobytes()).hexdigest()}
                       for s in scenes] for name, scenes in splits.items()}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    comparison(model, [splits["test"][0], splits["test"][1], splits["domain_shift"][0]],
               output / "mask_comparison.png")
    learning_curve(history, output / "learning_curve.png")
    comparison(model, [splits["test"][0], splits["test"][1], splits["domain_shift"][0]],
               output / "augmentation_comparison.png", comparison_model=augmented)
    learning_curve(augmented_history, output / "augmentation_learning_curve.png")
    Image.fromarray(splits["domain_shift"][0].image).save(output / "example_camera.png")
    with (output / "augmentation_training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(augmented_history[0]))
        writer.writeheader()
        writer.writerows(augmented_history)
    for name in ("test", "domain_shift"):
        print(f"{name}: mean IoU={results[name]['mean_iou']:.4f}, "
              f"pixel accuracy={results[name]['pixel_accuracy']:.4f}, "
              f"floor precision={results[name]['floor_precision']:.4f}")
        print(f"style-augmented {name}: mean IoU={results['style_augmentation'][name]['mean_iou']:.4f}, "
              f"floor precision={results['style_augmentation'][name]['floor_precision']:.4f}")
    print(f"Saved results to {output.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    run(args.output, args.epochs)
