# 🖼️ Image Classification on CIFAR-10

After getting 99.6% on handwritten digits, I wanted something that would humble me. CIFAR-10 does that: 60,000 tiny (32×32) colour photos in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The photos are real, the backgrounds are cluttered, and cats look a lot like dogs at this resolution.

The goal wasn't just a high number. I wanted to find out **which of the standard CNN tricks actually matter**: batch normalisation, data augmentation, dropout. So I built a custom CNN and ran a *leave-one-out ablation*, switching each idea off one at a time and measuring what it was worth.

## The models

| | BaselineCNN | CifarNet (mine) |
|---|---|---|
| Layers | 3 conv → 2 dense | 3 blocks × (2 conv 3×3 → BN → ReLU) → max-pool, then global average pooling |
| Channels | 32 → 64 → 128 | 48 → 96 → 192 |
| Parameters | 620k | 647k |
| Batch norm / dropout | no / no | yes / yes (more dropout deeper in) |
| Augmentation | none | random crop (pad 4, reflect) + horizontal flip |

Both have about the same number of parameters, but they spend them very differently. The baseline puts most of its weights in a big dense layer. CifarNet puts them into convolutions and replaces the dense layer with global average pooling.

## Ablation results (8 epochs each, test set of 10,000 images)

| Experiment | Test accuracy | Train accuracy (last epoch) |
|---|---|---|
| Baseline CNN | 78.1% | **98.5%** |
| CifarNet, everything on | 80.8% | 76.7% |
| CifarNet − batch norm | 75.6% | 71.3% |
| CifarNet − augmentation | 82.6% | 81.8% |
| CifarNet − dropout | **86.9%** | 88.7% |

I didn't expect this. What the numbers say:

- **The baseline overfits badly.** It hits 98.5% on training data and only 78.1% on test data, so it memorised the training set.
- **Batch norm matters most.** Removing it costs about 5 points (80.8% → 75.6%), and those epochs trained noticeably slower too.
- **Dropout *hurt*, at least with this budget.** Removing it was the single biggest improvement: 80.8% → 86.9%. With augmentation *and* batch norm already regularising the network, dropout on top was too much. After 8 epochs the "full" model was still *underfitting* (train accuracy 76.7%, below its test accuracy!).
- **Augmentation also slowed learning in the short run** (82.6% without it vs 80.8% with). It pays off over longer training, which is exactly what the 30-epoch runs below are designed to check.

The lesson I'm taking from this: regularisation isn't free. Each trick fixes overfitting, and stacking all of them on a short training budget means the model never gets to fit at all.

## The final model: not trained yet ⏳

`run_experiments.sh` ends with two 30-epoch candidates, CifarNet with everything on and CifarNet without dropout. `src.evaluate` then picks the final model by **validation** accuracy (never test). On this laptop (no GPU) that's about 2.5 hours, so it hasn't been run yet. Once it has, `outputs/` will contain the confusion matrix, per-class accuracy, the model's most confident mistakes, and Grad-CAM heatmaps showing which pixels each prediction was based on.

All of that code is written and checked: every script was run end to end on a small slice of the data.

## Project structure

```
cifar10-image-classification/
├── src/
│   ├── data.py       # downloads CIFAR-10 (Hugging Face mirror, ~50x faster), caches as .npz, loaders + augmentation
│   ├── models.py     # BaselineCNN, CifarNet, and the 5 named experiment configs
│   ├── train.py      # AdamW + one-cycle LR, channels-last for faster CPU training, checkpoint on best val accuracy
│   ├── evaluate.py   # ablation chart, learning curves, confusion matrix, mistakes, Grad-CAM; picks the final model
│   ├── gradcam.py    # Grad-CAM written from scratch with forward/backward hooks
│   └── predict.py    # classify any image (+ optional heatmap)
├── run_experiments.sh   # the whole study in one go
├── samples/          # one test image per class to try predict.py on
├── tests/
├── outputs/runs/     # per-run history + test accuracy (the ablation results above)
└── Makefile
```

## Run it

```bash
git clone https://github.com/Shreyabhawsar/cifar10-image-classification.git
cd cifar10-image-classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
# CPU-only PyTorch is a much smaller download:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

./run_experiments.sh                   # ablation + final candidates (~2.5 h CPU, minutes on a GPU)
python -m src.train --experiment no-dropout --epochs 30   # or just one run
python -m src.evaluate                 # figures + metrics.json, picks the final model
python -m src.predict samples/*.png --gradcam
pytest -q
```

## If I took this further

- **Residual connections** (a small ResNet) are the standard next step; they'd likely push this past 90%.
- **Stronger augmentation** (Cutout, MixUp) combined with longer training, now that I know regularisation needs epochs to pay off.
- **Repeat each ablation run with 3 seeds.** A single run per setting can't tell a 1-point difference from noise.

---

## Part of a series

This is one of five intermediate ML projects, each in its own repo:

- [🖼️ CIFAR-10 Image Classification](https://github.com/Shreyabhawsar/cifar10-image-classification) ← you are here
- [💬 Sentiment Analysis with LSTMs](https://github.com/Shreyabhawsar/sentiment-analysis-lstm)
- [💳 Credit Card Fraud Detection](https://github.com/Shreyabhawsar/credit-card-fraud-detection)
- [🦓 Object Detection with YOLO](https://github.com/Shreyabhawsar/object-detection-yolo)
- [🛒 Customer Segmentation](https://github.com/Shreyabhawsar/customer-segmentation-clustering)

MIT licensed. See [LICENSE](LICENSE). CIFAR-10: Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*.
