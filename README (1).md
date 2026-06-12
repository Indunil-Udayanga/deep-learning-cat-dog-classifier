# Cat vs Dog Classifier 🐱🐶

Binary image classification using two approaches — a custom CNN and Transfer Learning (Xception) — built with TensorFlow/Keras.

## Notebooks

| Notebook | Approach | Epochs |
|---|---|---|
| `cat_dog_classifier.ipynb` | Custom CNN (Conv2D × 3, Dropout, BatchNorm) | 20 |
| `cat_dog_classifier_transfer_learning.ipynb` | Xception pretrained on ImageNet | 3 |

## Dataset

[Cats and Dogs for Classification](https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification) — downloaded automatically via `opendatasets`.

## Setup

```bash
pip install -r requirements.txt
```

Then open either notebook in Google Colab or Jupyter and run all cells.

## Results

Both models evaluated on Precision, Recall, and Binary Accuracy.  
Transfer Learning converges in 3 epochs vs 20 for the custom CNN.
