# **DenPMag**

## **Overview**

*DenPMag* is a deep learning framework designed for the prediction of earthquake magnitude and peak ground acceleration (PGA) from single-station data. The model leverages both temporal and frequency information to achieve superior performance in predicting seismic events. This framework includes custom neural network layers, attention mechanisms, and transformer encoders to enhance the feature extraction process.

## **Structure**

The project is organized into the following files:

* **main.py**: Main script to run the training process.
* **dendrite_layer.py**: Definition of the custom dendritic neural network layer.
* **attention_block.py**: Functions for the attention blocks.
* **transformer_encoder.py**: Transformer encoder layer definition.
* **model.py**: Function to create the complete model.
* **utils.py**: Utility functions for mask generation and data preprocessing.

## **Requirements**

* *Python 3.7+*
* *TensorFlow 2.x*
* *NumPy*

## **Installation**

1. *Clone the repository.*
2. *Install the required Python packages:*

## **Usage**

1. *Prepare your data:*
    * Ensure you have your data formatted correctly. You will need seismic waveform data, sequence data, and tabular data for training.
2. *Train the model:*
    * Edit `main.py` to include your data loading process.
    * Run the training script: `python main.py`

## **File Descriptions**

### **main.py**

*The main script to run the training process. It initializes the model, sets up the training configuration, and starts the training process. Ensure to replace placeholder variables with actual data variables.*

### **dendrite_layer.py**

*Contains the implementation of the DendriteLayer, a custom dense layer designed to mimic the behavior of dendritic neurons. This layer enhances the model's ability to learn complex patterns.*

### **attention_block.py**

*Defines the functions for the attention blocks, including Fourier transform-based attention mechanisms and the Taylor softmax function. Also includes positional encoding for transformer layers.*

### **transformer_encoder.py**

*Implements the TransformerEncoder class, which is used to encode sequence data with attention mechanisms.*

### **model.py**

*Contains the function `create_attention_cnn_model` which assembles the complete model architecture, combining CNN layers, attention mechanisms, transformer encoders, and dendritic layers.*

### **utils.py**

*Provides utility functions for generating masks and preprocessing data. These functions are used to create the masks needed for the dendritic layers and other preprocessing tasks.*

## **License**

*This project is licensed under the terms of the MIT license.*
