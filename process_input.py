import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow_text import MaskValuesChooser, mask_language_model
from tensorflow_text.python.ops.item_selector_ops import RandomItemSelector

preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_zh_preprocess/3")
tokenize = hub.KerasLayer(preprocessor.tokenize)
special_tokens_dict = preprocessor.tokenize.get_special_tokens_dict()

sentences = tf.constant(["春眠不觉晓，处处闻啼鸟。"])
input_word_ids = tokenize(sentences)
input_word_ids = tf.squeeze(input_word_ids, [-1])

masked_input_ids, masked_positions, masked_ids = mask_language_model(
    input_word_ids, RandomItemSelector(10000, 0.1), MaskValuesChooser(
        special_tokens_dict['vocab_size'], special_tokens_dict['mask_id'], mask_token_rate=0.8, random_token_rate=0.1))

seq_length = 20

input_word_ids = input_word_ids.to_tensor(default_value=0, shape=[None, seq_length])
input_mask = masked_input_ids.to_tensor(default_value=0, shape=[None, seq_length])
input_type_ids = tf.zeros([1, seq_length])
masked_lm_positions = tf.constant([7, 9])

mlm_inputs = dict(
    input_mask=tf.keras.layers.Input(shape=(seq_length,), name="input_mask", dtype=tf.int32),
    masked_lm_positions=tf.keras.layers.Input(shape=(None,), name="masked_lm_positions", dtype=tf.int32),
    input_type_ids=tf.keras.layers.Input(shape=(seq_length,), name="input_type_ids", dtype=tf.int32),
    input_word_ids=tf.keras.layers.Input(shape=(seq_length,), name="input_word_ids", dtype=tf.int32),
)

mlm_inputs = dict(
    input_mask=input_mask,
    masked_lm_positions=masked_lm_positions,
    input_type_ids=input_type_ids,
    input_word_ids=input_word_ids,
)

encoder = hub.load("https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4")
mlm = hub.KerasLayer(encoder.mlm, trainable=True)
mlm_outputs = mlm(mlm_inputs)

model = tf.keras.Model(mlm_inputs, mlm_outputs)

mlm_logits = mlm_outputs["mlm_logits"]