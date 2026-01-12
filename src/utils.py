# src/utils.py
import os
import yaml
import numpy as np
from PIL import Image
import tensorflow as tf

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def make_image_processor(image_size):
    def proc(img):
        if isinstance(img, str):
            im = Image.open(img).convert("RGB")
        else:
            im = Image.fromarray(img) if not isinstance(img, Image.Image) else img
            im = im.convert("RGB")
        im = im.resize((image_size, image_size))
        arr = np.array(im).astype(np.float32) / 255.0
        arr = tf.keras.applications.resnet.preprocess_input(arr*255.0)
        return arr
    return proc

def prepare_dataset(cfg, split="train", keep_small=False):
    # lazy imports to avoid import-time errors
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(cfg['dataset']['hf_name'], split=split)
    if keep_small:
        ds = ds.select(range(min(256, len(ds))))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    image_size = cfg['dataset']['image_size']
    seq_len = cfg['dataset']['seq_len']
    max_cap_len = cfg['dataset']['max_caption_len']
    image_proc = make_image_processor(image_size)

    def map_example(example):
        frames = None
        captions = None
        for k in ["frames", "images", "frames_paths", "image_paths", "imgs"]:
            if k in example:
                frames = example[k]
                break
        for k in ["captions", "descriptions", "texts", "story"]:
            if k in example:
                captions = example[k]
                break
        if frames is None:
            frames = []
        if captions is None:
            captions = []

        frames = frames[:seq_len]
        captions = captions[:seq_len]

        imgs = []
        for f in frames:
            try:
                imgs.append(image_proc(f))
            except Exception:
                imgs.append(np.zeros((image_size, image_size, 3), dtype=np.float32))
        while len(imgs) < seq_len:
            imgs.append(np.zeros((image_size, image_size, 3), dtype=np.float32))

        tok_out = tokenizer(captions,
                            padding='max_length',
                            truncation=True,
                            max_length=max_cap_len,
                            return_tensors="np")
        input_ids = tok_out['input_ids']
        if input_ids.shape[0] < seq_len:
            pad_rows = np.zeros((seq_len - input_ids.shape[0], max_cap_len), dtype=np.int32)
            input_ids = np.vstack([input_ids, pad_rows])

        return {
            "images": np.stack(imgs).astype(np.float32),   # seq_len, H, W, C
            "input_ids": input_ids.astype(np.int32)       # seq_len, T
        }

    processed = []
    for ex in ds:
        processed.append(map_example(ex))
    return processed, tokenizer

def generator_from_processed(processed_list, cfg):
    def gen():
        for ex in processed_list:
            yield ex['images'], ex['input_ids']
    return gen

def make_tf_dataset(processed_list, cfg, shuffle=True):
    seq_len = cfg['dataset']['seq_len']
    max_cap_len = cfg['dataset']['max_caption_len']
    batch_size = cfg['dataset']['batch_size']
    image_size = cfg['dataset']['image_size']
    import tensorflow as tf
    out_types = (tf.float32, tf.int32)
    out_shapes = ((seq_len, image_size, image_size, 3), (seq_len, max_cap_len))
    ds = tf.data.Dataset.from_generator(generator_from_processed(processed_list, cfg),
                                        output_types=out_types, output_shapes=out_shapes)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
