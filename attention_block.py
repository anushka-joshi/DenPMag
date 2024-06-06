import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Flatten

def taylor_softmax(x, degree=2):
    numerator = tf.math.pow(x, degree)
    denominator = tf.reduce_sum(tf.math.pow(x, degree), axis=-1, keepdims=True)
    return numerator / (denominator + 1e-10)

def fft2d_on_channels(image_tensor):
    channels = tf.unstack(image_tensor, axis=-1)
    fft_results = []
    for channel in channels:
        fft_channel = tf.signal.fft2d(tf.cast(channel, tf.complex64))
        fft_results.append(fft_channel)
    fft_output = tf.stack(fft_results, axis=-1)
    return fft_output

def Attention_Block(temporal_conv3, shape1):
    temporal_conv3_complex = tf.cast(temporal_conv3, tf.complex64)
    fft_output = fft2d_on_channels(temporal_conv3_complex)
    H = tf.cast(tf.shape(temporal_conv3)[1], tf.float32)
    W = tf.cast(tf.shape(temporal_conv3)[0], tf.float32)
    N = H * W
    fft_output_amplitude = tf.square(tf.abs(fft_output))
    power_spectrum = fft_output_amplitude / (N ** 2)
    power_spectrum_mul = tf.multiply(temporal_conv3, power_spectrum)
    temporal_conv3_compressed = Conv2D(1, shape1, strides=shape1, activation=lambda x: taylor_softmax(x, degree=2))(power_spectrum_mul)
    softmax_output_time = temporal_conv3_compressed
    attention_mul = tf.multiply(softmax_output_time, temporal_conv3)
    temporal_maxpool = Flatten()(attention_mul)
    return temporal_maxpool

def get_positional_encoding(seq_length, d_model):
    pos = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    positional_encoding = np.zeros((seq_length, d_model))
    positional_encoding[:, 0::2] = np.sin(pos * div_term)
    positional_encoding[:, 1::2] = np.cos(pos * div_term)
    return tf.cast(positional_encoding, dtype=tf.float32)
