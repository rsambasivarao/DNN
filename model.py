# src/model.py (only the build_multimodal_model function needs to match this)
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_visual_encoder(image_size=224, feat_dim=512, backbone='resnet50'):
    base = tf.keras.applications.ResNet50(include_top=False, pooling='avg',
                                          input_shape=(image_size, image_size, 3), weights='imagenet')
    out_dim = base.output_shape[-1]
    inp = tf.keras.Input(shape=(image_size, image_size, 3), name='image_input')
    feat = base(inp)
    if out_dim != feat_dim:
        feat = layers.Dense(feat_dim, activation='relu')(feat)
    return Model(inputs=inp, outputs=feat, name='visual_encoder')

def build_text_encoder(vocab_size, embed_dim=300, hidden_dim=512, max_len=64):
    inp = tf.keras.Input(shape=(max_len,), dtype='int32', name='caption_input')
    # Disable mask_zero to avoid propagation of boolean masks through TimeDistributed
    # which can cause BroadcastTo errors when Keras attempts to align masks with
    # outer layers. Padding is handled manually in the loss computation.
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=False)(inp)
    _, state_h, _ = layers.LSTM(hidden_dim, return_state=True)(x)
    return Model(inputs=inp, outputs=state_h, name='text_encoder')

def build_multimodal_model(cfg):
    seq_len = cfg['dataset']['seq_len']
    image_size = cfg['dataset']['image_size']
    max_cap_len = cfg['dataset']['max_caption_len']

    visual_enc = build_visual_encoder(image_size=image_size, feat_dim=cfg['model']['image_feat_dim'])
    text_enc = build_text_encoder(vocab_size=cfg['model']['vocab_size'],
                                  embed_dim=cfg['model']['text_embed_dim'],
                                  hidden_dim=cfg['model']['text_hidden_dim'],
                                  max_len=max_cap_len)

    images_in = tf.keras.Input(shape=(seq_len, image_size, image_size, 3), name='images_seq')
    captions_in = tf.keras.Input(shape=(seq_len, max_cap_len), dtype='int32', name='captions_seq')

    td_visual = layers.TimeDistributed(visual_enc)(images_in)   # (B, seq_len, image_feat_dim)
    td_text = layers.TimeDistributed(text_enc)(captions_in)     # (B, seq_len, text_hidden_dim)

    fused = layers.Concatenate(axis=-1)([td_visual, td_text])
    fused = layers.TimeDistributed(layers.Dense(cfg['model']['multimodal_dim'], activation='relu'))(fused)

    temporal_out, state_h, state_c = layers.LSTM(cfg['model']['temporal_hidden_dim'],
                                                 return_sequences=True, return_state=True)(fused)

    context = temporal_out[:, -1, :]  # (B, temporal_hidden_dim)

    # Text decoder (teacher forcing) — build subgraph that accepts dec_input of fixed length
    dec_input = tf.keras.Input(shape=(max_cap_len,), dtype='int32', name='dec_input')
    # Disable mask_zero here as well to prevent mask tensors (bool) from being
    # propagated into higher-level layers where they can cause unexpected
    # BroadcastTo operations. We compute padding masks explicitly in the loss.
    dec_emb = layers.Embedding(cfg['model']['vocab_size'], cfg['model']['text_embed_dim'], mask_zero=False)(dec_input)

    # Repeat the context vector across time safely:
    ctx_tile = layers.RepeatVector(max_cap_len)(context)  # (B, T, H)
    dec_lstm_in = layers.Concatenate(axis=-1)([dec_emb, ctx_tile])
    dec_lstm_out = layers.LSTM(cfg['model']['text_decoder_hidden'], return_sequences=True)(dec_lstm_in)
    logits = layers.TimeDistributed(layers.Dense(cfg['model']['vocab_size']))(dec_lstm_out)

    text_decoder_model = tf.keras.Model(inputs=[images_in, captions_in, dec_input], outputs=logits, name='text_decoder')

    # Image decoder
    img_pred = layers.Dense(cfg['model']['temporal_hidden_dim'], activation='relu')(context)
    img_pred = layers.Dense(cfg['model']['image_feat_dim'])(img_pred)

    full_model = tf.keras.Model(inputs=[images_in, captions_in, dec_input],
                                outputs=[logits, img_pred],
                                name='multimodal_model')

    return {
        "full_model": full_model,
        "visual_enc": visual_enc,
        "text_enc": text_enc,
        "text_decoder": text_decoder_model
    }
