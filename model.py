import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten
from tensorflow.keras.models import Model
from dendrite_layer import DendriteLayer
from attention_block import Attention_Block, get_positional_encoding
from transformer_encoder import TransformerEncoder

def create_attention_cnn_model(target_height, target_width, your_sequence_length, your_num_features, embed_dim=64, dense_dim=64, num_heads=4, encoder_layers=4):
    waveform_input = Input(shape=(target_height, target_width, 1))

    temporal_conv1 = Conv2D(32, (1, 5), activation='relu', padding='same', strides=(1, 2))(waveform_input)
    temporal_conv2 = Conv2D(16, (1, 5), activation='relu', padding='same', strides=(1, 2))(temporal_conv1)
    temporal_conv3 = Conv2D(8, (1, 5), activation='relu', padding='same', strides=(1, 3))(temporal_conv2)
    temporal_conv4 = Conv2D(4, (1, 5), activation='relu', padding='same', strides=(1, 4))(temporal_conv3)

    temporal_maxpool_1 = Attention_Block(temporal_conv1, (1, 64))
    temporal_maxpool_2 = Attention_Block(temporal_conv2, (1, 32))
    temporal_maxpool_3 = Attention_Block(temporal_conv3, (1, 11))
    temporal_maxpool_4 = Attention_Block(temporal_conv4, (1, 2))

    frequency_conv1 = Conv2D(32, (5, 1), activation='relu', padding='same', strides=(2, 1))(waveform_input)
    frequency_conv2 = Conv2D(16, (5, 1), activation='relu', padding='same', strides=(2, 1))(frequency_conv1)
    frequency_conv3 = Conv2D(8, (5, 1), activation='relu', padding='same', strides=(3, 1))(frequency_conv2)
    frequency_conv4 = Conv2D(4, (5, 1), activation='relu', padding='same', strides=(4, 1))(frequency_conv3)

    frequency_maxpool_1 = Attention_Block(frequency_conv1, (64, 1))
    frequency_maxpool_2 = Attention_Block(frequency_conv2, (32, 1))
    frequency_maxpool_3 = Attention_Block(frequency_conv3, (11, 1))
    frequency_maxpool_4 = Attention_Block(frequency_conv4, (2, 1))

    concatenated_layers = tf.keras.layers.Concatenate(axis=-1)([
        temporal_maxpool_1, temporal_maxpool_2, temporal_maxpool_3, temporal_maxpool_4,
        frequency_maxpool_1, frequency_maxpool_2, frequency_maxpool_3, frequency_maxpool_4
    ])

    lstm_input = Input(shape=(your_sequence_length, your_num_features))

    seq_length = lstm_input.shape[1]
    positional_encoding = get_positional_encoding(seq_length, embed_dim)
    x = tf.keras.layers.Dense(embed_dim)(lstm_input)
    x += positional_encoding

    for _ in range(encoder_layers):
        transformer_x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    transformer_x_flat = Flatten()(transformer_x)

    concat = tf.keras.layers.concatenate([concatenated_layers, transformer_x_flat])
    tabular_input = Input(shape=(7,))
    concat = tf.keras.layers.concatenate([concat, tabular_input])

    flatten_layer = Flatten()(concat)
    dense1 = DendriteLayer(n_units=256, act=lambda x: x * tf.tanh(tf.math.log(1 + tf.sigmoid(x))), branch=3)(flatten_layer)
    dense1 = DendriteLayer(n_units=128, act=lambda x: x * tf.tanh(tf.math.log(1 + tf.sigmoid(x))), branch=4)(dense1)
    output_layer = DendriteLayer(1)(dense1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model_cnn_bilstm = Model(inputs=[waveform_input, lstm_input, tabular_input], outputs=output_layer)
    model_cnn_bilstm.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model_cnn_bilstm
