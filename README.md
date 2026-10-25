# tomato-leaf-disease-cnn

Classification of five leaf diseases and healthy leaves in tomato from RGB images, using transfer
learning with MobileNetV2 (TensorFlow/Keras).

## Data

Tomato leaf images from the Tomato Leaf Disease Dataset (Mendeley Data, version 1). Six classes were used:

| Class | Type | Images |
|---|---|---|
| Early blight | Fungal | 298 |
| Late blight | Oomycete (grouped with the fungal diseases in the dataset) | 111 |
| Leaf mould | Fungal | 400 |
| Septoria leaf spot | Fungal | 106 |
| Target spot | Fungal | 99 |
| Healthy | – | 157 |

The classes are imbalanced (99 to 400 images). Images are not redistributed here; download them from Mendeley
Data and place one folder per class in `data/raw/`.

## Method

1. **Split:** 70% training, 15% validation, 15% test, stratified by class (`split-folders`, seed 42).
2. **Augmentation (training only):** rotation (±15°), zoom (10%), shifts (10%), brightness (0.8–1.2) and
   horizontal flips; all images rescaled to 224 × 224 and to [0, 1].
3. **Model:** MobileNetV2 pre-trained on ImageNet with its convolutional base frozen, followed by global average
   pooling, dropout (0.3), a dense layer of 128 units, dropout (0.2) and a six-class softmax output.
4. **Training:** Adam (learning rate 1e-4), categorical cross-entropy, batch size 32, up to 25 epochs with early
   stopping and checkpointing on validation accuracy.
5. **Evaluation:** accuracy, per-class precision, recall and F1, and a confusion matrix on the held-out test set.

## Results

On the 182 test images: accuracy 59.3%, macro F1 0.60. Per-class results are in
[`results/classification_report.txt`](results/classification_report.txt).

## Limitations and next steps

- **Frozen base.** Only the classification head was trained. Fine-tuning the upper MobileNetV2 blocks at a low
  learning rate is the obvious next step.
- **Class imbalance** was not corrected; class weights or balanced sampling would likely help the smaller classes.
- **Small test set.** With 16 to 60 test images per class, per-class estimates are uncertain; cross-validation
  would give more reliable figures.
- **Curated images.** The images come from one curated dataset, so accuracy on field photographs would likely be
  lower.

## Reproduce

```bash
pip install -r requirements.txt
python src/train.py --data data/raw --out results
```

## Licence

Code: MIT. The image data remain under the licence of the original dataset.

Ali Hamza · alihamxa49@gmail.com
