import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
from config import Config

config = Config()


def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def char_split(inputs):
    return tf.strings.unicode_split(inputs, "UTF-8")


def get_vectorize_layer(texts, max_seq=20, special_tokens=["x"]):
    """Build Text vectorization layer

    Args:
      texts (list): List of string i.e input texts
      max_seq (int): Maximum sequence lenght.
      special_tokens (list, optional): List of special tokens. Defaults to ['x'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
    vectorize_layer = TextVectorization(
        output_mode="int",
        split=char_split,
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)

    # Insert mask token in vocabulary
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2:] + ["x"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer


def get_masked_input_and_labels(encoded_texts):
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 2] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    print(inp_mask_2mask[0])
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights


# Prepare data for masked language model

text = get_text_list_from_files(["/Users/miaowu/Documents/GitHub/bert-mlm/5jue.txt"])
vectorize_layer = get_vectorize_layer(text, max_seq=config.MAX_LEN)

x = vectorize_layer.get_vocabulary()
# Get mask token id for masked language model
mask_token_id = vectorize_layer(["x"]).numpy()[0][0]

encoded_text = vectorize_layer(text).numpy()


def get_train_ds():
    print('666666')
    x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(encoded_text)

    mlm_ds = tf.data.Dataset.from_tensor_slices(
        (x_masked_train, y_masked_labels, sample_weights)
    )
    mlm_ds = mlm_ds.repeat(100).shuffle(10000).batch(config.BATCH_SIZE)

    return mlm_ds



