# 🐱🐶 Deep Learning — Cat vs Dog Classifier

A deep learning project that classifies images as either a cat or a dog using two different approaches — a Custom CNN built from scratch and Transfer Learning using the Xception model.

---

## What is this project?

This project explores image classification using TensorFlow/Keras. The same dataset and pipeline are used for both approaches, making it easy to compare how a simple CNN performs against a powerful pretrained model.

---

## Approaches

| Notebook | Method | Epochs |
|---|---|---|
| `cat_dog_classifier.ipynb` | Custom CNN — Conv2D × 3, Dropout, BatchNorm | 20 |
| `cat_dog_classifier_transfer_learning.ipynb` | Transfer Learning — Xception (ImageNet) | 3 |

---

## Dataset

[Cats and Dogs for Classification — Kaggle](https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification)

Downloaded automatically inside the notebook using `opendatasets`.

---

## Setup

```bash
pip install -r requirements.txt
```

Open either notebook in **Google Colab** or **Jupyter Notebook** and run all cells.

---

## Key Takeaway

Transfer Learning (Xception) achieves strong results in just **3 epochs** by reusing features learned from millions of ImageNet images — while the custom CNN requires **20 epochs** to train from scratch.

---

## Tech Stack

`Python` `TensorFlow` `Keras` `NumPy` `Matplotlib` `OpenCV` `Kaggle`
