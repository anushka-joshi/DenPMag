import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dendrite_layer import DendriteLayer
from attention_block import Attention_Block, get_positional_encoding
from transformer_encoder import TransformerEncoder
from model import create_attention_cnn_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

your_target_height = 128
your_target_width = 128
your_sequence_length = 300 
your_num_features = 3

model = create_attention_cnn_model(your_target_height, your_target_width, your_sequence_length, your_num_features)
checkpoint = ModelCheckpoint(filepath='Attention_weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Replace X_train, seismic_train, tab_train, y_train, X_val, seismic_val, tab_val, y_val with actual data variables
model.fit(x=[X_train, seismic_train, tab_train], y=y_train, batch_size=32, epochs=50, 
          validation_data=([X_val, seismic_val, tab_val], y_val), callbacks=[checkpoint, early_stopping])
model.load_weights('Attention_weights.hdf5')
