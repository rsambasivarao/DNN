

---

# 🚀 Multimodal Image–Text Sequence Model

*A lightweight Transformer-free multimodal model for image-sequence → caption generation*

---

## 📌 Overview

This project implements a **multimodal encoder–decoder** model that takes:

* A **sequence of images**
* A **sequence of tokenized captions**

…and learns to **predict the final caption** in the sequence + a **visual feature reconstruction target** for the last image.

The model is optimized for **fast debugging**, **low GPU memory usage**, and **clear modular code structure**.

---

## 📁 Project Structure

```
project/
│
├── src/
│   ├── utils.py              # dataset loading, preprocessing, tf.data builder
│   ├── model.py              # multimodal model architecture
│   ├── train.py              # full training script
│
├── notebooks/
│   ├── debug_train.ipynb     # small-debug training notebook (5–50 samples)
│
├── config.yaml               # all dataset + training hyperparameters
├── README.md                 # this file
```

---

## ⚙️ Installation

### 1. Create environment

```bash
pip install -r requirements.txt
```

### 2. Additional packages (if missing)

```bash
pip install nltk rouge-score transformers tensorflow pillow
```

---

## 📦 Dataset Format

Each training example contains:

* `images`: shape **(seq_len, H, W, 3)**
* `input_ids`: shape **(seq_len, T)** — tokenized caption sequence

Your dataset folders must follow config paths defined in `config.yaml`.

---

## 🧠 Model Architecture

### Visual Encoder

* CNN backbone (e.g., EfficientNet / ConvNet)
* Outputs one feature vector per image

### Caption Sequence Encoder

* Embeds historical captions
* Produces temporal text context

### Multimodal Fusion

* Concatenates image sequence embedding + caption sequence embedding
* Passes through dense fusion layers

### Text Decoder

* Teacher forcing during training
* Greedy autoregressive decoding during inference

---

## 🔥 Training

### Train via Notebook (debug fast)

```python
processed, tokenizer = prepare_dataset(cfg, split="train", keep_small=True)
tfds = make_tf_dataset(processed, cfg)
...
loss = train_step_impl(...)
```

### Train via script (full dataset)

```bash
python src/train.py --config config.yaml
```

Training progress and checkpoints are written to:

```
results/checkpoints/
```

---

## 🧪 Evaluation

### Metrics implemented:

* **BLEU-1**
* **ROUGE-L**
* Token-level comparison (decoder output vs ground truth)

### Example model prediction vs ground truth:

```
GT:  a small boy running in the park
PR:  a small kid running in the garden
BLEU=0.58  ROUGE-L=0.62
```

---

## 🖼️ Visualization (Notebook)

The notebook provides:

* Display of input image sequence
* Generated caption
* Ground truth caption

```python
plt.imshow(images[i])
plt.title(...)
plt.show()
```

---

## ⚠️ Notes on Speed

For **fastest debugging**, the notebook uses:

```python
keep_small=True
processed = processed[: int(len(processed) * 0.2) ]
cfg['dataset']['resize'] = 128
cfg['dataset']['batch_size'] = 8
```

These reduce:

* number of examples
* image resolution
* memory load
* iterations per epoch

Use full dataset only for real training.

---

## 📌 Roadmap

* [ ] Improve text decoder to Transformer-based
* [ ] Add beam search decoding
* [ ] Add CLIP image encoder option
* [ ] Add validation + test split evaluation
* [ ] Add TensorBoard logging

---

## 🙌 Credits

This project design was developed collaboratively with iterative debugging and model refinement.
Feel free to extend, optimize, or integrate with a larger multimodal pipeline.


