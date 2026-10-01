"""Classify your own image into one of the 10 CIFAR-10 classes.

    python -m src.predict samples/*.png
    python -m src.predict my_cat.jpg --gradcam     # also saves a heatmap of where the model looked

Any size works: the image is centre-cropped to a square and shrunk to 32x32,
which is all the model has ever seen. That shrinking is also its biggest limitation.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import CLASSES, EVAL_TRANSFORM, MODELS_DIR, OUTPUTS_DIR, denormalize
from src.gradcam import GradCAM
from src.models import load_trained

DEFAULT_WEIGHTS = MODELS_DIR / "cifarnet_final.pt"  # written by src.evaluate


def to_cifar(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((32, 32), Image.LANCZOS)


def classify(img: Image.Image, weights=DEFAULT_WEIGHTS):
    model = load_trained(weights)
    x = EVAL_TRANSFORM(to_cifar(img)).unsqueeze(0)
    heat, cls, probs = GradCAM(model, model.last_conv)(x)
    return CLASSES[cls], probs, heat, x[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--gradcam", action="store_true")
    args = parser.parse_args()
    for path in args.images:
        label, probs, heat, x = classify(Image.open(path))
        top3 = np.argsort(probs)[::-1][:3]
        print(f"{path}: {label}   (" + ", ".join(f"{CLASSES[i]} {probs[i]:.0%}" for i in top3) + ")")
        if args.gradcam:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            img = denormalize(x).permute(1, 2, 0).numpy()
            fig, axes = plt.subplots(1, 2, figsize=(5, 2.6))
            axes[0].imshow(img)
            axes[1].imshow(img)
            axes[1].imshow(heat, cmap="jet", alpha=0.45)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{label} ({probs.max():.0%})")
            out = OUTPUTS_DIR / f"gradcam_{Path(path).stem}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  heatmap -> {out}")


if __name__ == "__main__":
    main()
