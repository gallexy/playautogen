# filename: mymodel.py
import tensorflow as tf
import numpy as np

# Step 1: Define the Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Step 2: Define the Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x

# Step 3: Define the Transformer Model
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.final_linear = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x, training=False):
        x = self.encoder(x, training=training)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = self.final_linear(x)
        return x

# Step 4: Compile and test the model
model = Transformer(num_layers=2, d_model=128, num_heads=8, dff=128)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Preprocess the input data
input_text = "What is the capital of France"
input_ids = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
input_ids.fit_on_texts([input_text])
input_ids = input_ids.texts_to_sequences([input_text])
input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=50)

# Convert the input data to a tensor
input_data = tf.convert_to_tensor(input_ids)

# Test the model
output = model(input_data, training=False)
print(output)