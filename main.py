import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from pprint import pprint
from config import Config
from data_pipeline import get_train_ds, vectorize_layer, mask_token_id

config = Config()
checkpoint_dir = "ckpts/"

def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
loss_tracker = tf.keras.metrics.Mean(name="loss")
acc_tracker = tf.keras.metrics.CategoricalAccuracy()


# class MaskedLanguageModel(tf.keras.Model):
#     def train_step(self, inputs):
#         if len(inputs) == 3:
#             features, labels, sample_weight = inputs
#         else:
#             features, labels = inputs
#             sample_weight = None
#
#         with tf.GradientTape() as tape:
#             predictions = self(features, training=True)
#             loss = loss_fn(labels, predictions, sample_weight=sample_weight)
#
#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#
#         # Compute our own metrics
#         loss_tracker.update_state(loss, sample_weight=sample_weight)
#         acc_tracker.update_state(tf.one_hot(labels, depth=config.VOCAB_SIZE), predictions, sample_weight=sample_weight)
#
#         # Return a dict mapping metric names to current value
#         return {"loss": loss_tracker.result(), 'accuracy':acc_tracker.result() }
#
#     @property
#     def metrics(self):
#         # We list our `Metric` objects here so that `reset_states()` can be
#         # called automatically at the start of each epoch
#         # or at the start of `evaluate()`.
#         # If you don't implement this property, you have to call
#         # `reset_states()` yourself at the time of your choosing.
#         return [loss_tracker]


def create_masked_language_bert_model():
    # if tf.io.gfile.exists(os.path.join(checkpoint_dir, "savedmodel")):
    #     mlm_model = tf.keras.models.load_model(os.path.join(checkpoint_dir, "savedmodel"))
    #     optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    #     mlm_model.compile(optimizer=optimizer)
    #     return mlm_model

    features = layers.Input((config.MAX_LEN,), name="x", dtype=tf.int64)
    y = layers.Input((config.MAX_LEN,), name="y", dtype=tf.int64)
    w = layers.Input((config.MAX_LEN,), name="w", dtype=tf.int64)

    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
    )(features)
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
        encoder_output
    )
    mlm_model = tf.keras.Model([features, y, w], mlm_output)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, mlm_output, w)
    mlm_model.add_loss(loss)

    acc = tf.keras.layers.Dot(axes=-1)([w, tf.cast(tf.equal(y, tf.argmax(mlm_output, axis=-1)), dtype=tf.int64)]) / tf.cast(tf.reduce_sum(w), tf.int64)
    mlm_model.add_metric(acc, aggregation='mean', name='acc')


    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        print(latest)
        mlm_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model


id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    # def on_epoch_end(self, epoch, logs=None):
    #     prediction = self.model.predict(self.sample_tokens)
    #
    #     masked_index = np.where(self.sample_tokens == mask_token_id)
    #     masked_index = masked_index[1]
    #     mask_prediction = prediction[0][masked_index]
    #
    #     top_indices = mask_prediction[0].argsort()[-self.k:][::-1]
    #     values = mask_prediction[0][top_indices]
    #
    #     for i in range(len(top_indices)):
    #         p = top_indices[i]
    #         v = values[i]
    #         tokens = np.copy(sample_tokens[0])
    #         tokens[masked_index[0]] = p
    #         result = {
    #             "input_text": self.decode(sample_tokens[0].numpy()),
    #             "prediction": self.decode(tokens),
    #             "probability": v,
    #             "predicted mask token": self.convert_ids_to_tokens(p),
    #         }
    #         pprint(result)

    def on_epoch_end(self, epoch, logs=None):
        result = ''
        # for j in range(7, 10):
        #     prediction = self.model.predict(self.sample_tokens)
        #     mask_prediction = prediction[0][j]
        #
        #     top_index = mask_prediction.argsort()[-1]
        #     print(mask_prediction)
        #     print(top_index)
        #     self.sample_tokens[0][j+1] = self.sample_tokens[0][j]
        #     self.sample_tokens[0][j] = top_index
        #     result += id2token[top_index]
        # print(result)


sample_tokens = vectorize_layer(["?????????????????????x??????"])
print(sample_tokens)
generator_callback = MaskedTextGenerator(sample_tokens.numpy())
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir, update_freq='epoch')
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, 'ckpt-{epoch}'), save_weights_only=True)
train_ds = get_train_ds()


class DataCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global train_ds
        train_ds = get_train_ds()
        print(train_ds.take(1))


bert_masked_model = create_masked_language_bert_model()
for element in train_ds.take(1):
    input_x = element['x'].numpy()
    input_y = element['y'].numpy()
    pred = tf.math.argmax(bert_masked_model(element), -1).numpy()

    for i in range(10):
        print("".join([id2token[t] for t in input_x[i] if t != 0]))
        print("".join([id2token[t] for t in input_y[i] if t != 0]))
        print("".join([id2token[t] for t in pred[i] if t != 0]))

bert_masked_model.fit(train_ds, epochs=10000, callbacks=[DataCallback(), generator_callback, tensorboard_callback, ckpt_callback])



