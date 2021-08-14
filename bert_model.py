import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from bert_pipeline import get_data_and_label_from_files

data, label = get_data_and_label_from_files("/Users/heixiu/Documents/GitHub/bert-mlm/5jue.txt")
preprocessor = hub.load("/Users/heixiu/Downloads/bert_zh_preprocess_3")
tokenize = hub.KerasLayer(preprocessor.tokenize)
ds = tf.data.Dataset.from_tensor_slices((data,  tf.squeeze(tokenize(label).to_tensor(shape=(None, 128, 1)), -1)))
ds = ds.repeat(1).shuffle(10000).batch(16).prefetch(20)

# preprocess_layer = hub.KerasLayer("/Users/heixiu/Downloads/bert_zh_preprocess_3")
# for text_batch, label_batch in ds.take(1):
#     print(preprocess_layer(text_batch))


def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocess_layer = hub.KerasLayer("/Users/heixiu/Downloads/bert_zh_preprocess_3")
    encoder_inputs = preprocess_layer(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4", trainable=True)
    outputs = encoder(encoder_inputs)
    net = outputs['sequence_output']
    # net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(25000, activation=None)(net)
    return tf.keras.Model(text_input, net)


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metrics = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 5
steps_per_epoch = 1000
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model = build_classifier_model()
classifier_model.compile(optimizer=optimizer,
                         loss=loss, metrics=metrics)

history = classifier_model.fit(x=ds, epochs=epochs)