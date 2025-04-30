"""
This script contains functions for creating, compiling, training, 
evaluating, saving or loading the EIT reconstruction models used
in conducting the research. 
The 1D-CNN network builds upon previous research: 
DOI: 10.1063/5.0025881 or URL: https://pubmed.ncbi.nlm.nih.gov/33380008/
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Input, BatchNormalization,  
    Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError

from metrics import *


class EITReconstructionModel:
    def __init__(self, num_voltages, num_classes):
        self.num_voltages = num_voltages
        self.num_classes = num_classes
        self.model = None

    def create_dnn_model_classification(self):
        """
        Builds and compiles a deep neural network (DNN) model for multi-label classification
        of boundary voltages for EIT reconstruction. 

        The model architecture includes:
            - Input layer with batch normalisation and dropout.
            - Three hidden dense layers with ELU activation, batch normalisation, 
            L2 regularisation, and dropout.
            - Output layer with sigmoid activation for multi-label outputs.

        Compilation details:
            - Optimiser: Adam with a learning rate of 0.01.
            - Loss: Binary cross-entropy (suitable for multi-label classification).
            - Metrics: Jaccard accuracy, Hamming loss, Example-based F1 score, and Root Mean Squared Error (RMSE).

        Returns:
            keras.models.Sequential: The compiled DNN model ready for training.
        """

        self.model = Sequential()

        # Input Layer
        self.model.add(Dense(self.num_voltages, input_shape=(self.num_voltages,)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        # Layer 1 - Dense, elu
        self.model.add(Dense(self.num_voltages * 2, activation='elu', kernel_regularizer=l2(1e-3)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        # Layer 2 - Dense, elu
        self.model.add(Dense(self.num_voltages, activation='elu',
                            kernel_regularizer=l2(1e-3)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        # Layer 3 - Dense, elu
        self.model.add(Dense(self.num_classes * 2, activation='elu'))
        self.model.add(BatchNormalization())

        # Output Layer - Dense, sigmoid
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
        
        # Using binary cross entropy loss function for multi-label classification
        optimiser = Adam(learning_rate=1e-2)

        self.model.compile(optimizer=optimiser, 
                    loss='binary_crossentropy', 
                    metrics=[jaccard_accuracy, 
                             hamming_loss, 
                             example_based_f1, 
                             RootMeanSquaredError()])
        
        return self.model

    def create_1d_cnn_model_classification(self):
        """
        Builds and compiles a 1D Convolutional Neural Network (CNN) model for multi-label classification
        of boundary voltages for EIT reconstruction.

        The model architecture includes:
            - Input reshaped to (pairs_electrodes, 1) to support Conv1D layers.
            - Four 1D convolutional layers with ReLU activations, each followed by max pooling.
            - Flattening layer to transition to dense layers.
            - Three fully connected dense layers with ReLU activations and a dropout layer for regularisation.
            - Output dense layer with sigmoid activation for multi-label output.

        Compilation details:
            - Optimiser: Adam with a learning rate of 0.001.
            - Loss: Binary cross-entropy (appropriate for multi-label classification tasks).
            - Metrics: Jaccard accuracy, Hamming loss, Example-based F1 score, and Root Mean Squared Error (RMSE).

        Returns:
            keras.models.Sequential: The compiled 1D CNN model ready for training.
        """

        self.model = Sequential()

        # Input layer: (208,) reshape to (208, 1) for Conv1D
        self.model.add(Input(shape=(self.num_voltages, 1)))

        # 1D Convolutional layers with ReLU activations 
        # Conv Layer 1, with max pooling
        self.model.add(Conv1D(filters=8, kernel_size=21, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=1))

        # Conv Layer 2, with max pooling
        self.model.add(Conv1D(filters=8, kernel_size=13, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=1))
        
        # Conv Layer 3, with max pooling
        self.model.add(Conv1D(filters=16, kernel_size=7, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=1))

        # Conv Layer 4, with max pooling
        self.model.add(Conv1D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=1))

        # Flatten output before dense layers
        self.model.add(Flatten())

        # Fully connected layers
        # Dense 1
        self.model.add(Dense(self.num_voltages, activation='relu'))

        # Dense 2
        self.model.add(Dense(self.num_voltages * 2, activation='relu'))

        # Dense 3
        self.model.add(Dense(self.num_voltages * 3, activation='relu'))
        self.model.add(Dropout(0.4)) 
        
        # Output Layer 
        self.model.add(Dense(self.num_classes, activation='sigmoid')) 
        
        # Compile using the binary cross entropy loss function for multi-label classification.

        self.model.compile(optimizer=Adam(learning_rate=1e-3), 
                    loss='binary_crossentropy', 
                    metrics=[jaccard_accuracy, hamming_loss, example_based_f1, RootMeanSquaredError()])
        
        return self.model

    def create_1d_cnn_model_regression(self):
        """
        Builds and compiles a 1D Convolutional Neural Network (CNN) model for regression 
        with EIT boundary voltages to predict relative force at sample points.  
        Input is the EIT boundary voltages. Output is the size of number of sample points,
        where each index of the output is a float value representing the relative pressure.

        The model architecture includes:
            - Input reshaped to (pairs_electrodes, 1) for Conv1D processing.
            - Four 1D convolutional layers with ReLU activation, each followed by max pooling.
            - Flattening layer to transition to fully connected dense layers.
            - Three fully connected dense layers with ReLU activations and a dropout layer for regularisation.
            - Output dense layer with sigmoid activation to predict continuous values.

        Compilation details:
            - Optimiser: Adam with a learning rate of 1e-5.
            - Loss: Mean Squared Error (MSE), suitable for regression tasks.
            - Metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

        Returns:
            keras.models.Sequential: The compiled 1D CNN model ready for training.
        """

        self.model = Sequential()

        # Input layer: reshape (pairs_electrodes,) to (pairs_electrodes, 1) for Conv1D
        self.model.add(Input(shape=(self.num_voltages, 1)))

        # 1D Convolutional layers with ReLU activations 
        # Conv Layer 1 with max pooling
        self.model.add(Conv1D(filters=8, kernel_size=21, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=1))

        # Conv Layer 2 with max pooling
        self.model.add(Conv1D(filters=8, kernel_size=13, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=1))

        # Conv Layer 3 with max pooling
        self.model.add(Conv1D(filters=16, kernel_size=7, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=1))

        # Conv Layer 4 with max pooling
        self.model.add(Conv1D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=1))

        # Flatten output before dense layers
        self.model.add(Flatten())

        # Fully connected dense layers
        self.model.add(Dense(self.num_voltages, activation='relu'))
        self.model.add(Dense(self.num_voltages * 2, activation='relu'))
        self.model.add(Dense(self.num_voltages * 3, activation='relu'))
        
        # Dropout layer for regularisation
        self.model.add(Dropout(0.4))  

        # Output layer with sigmoid activation for continuous outputs
        self.model.add(Dense(self.num_classes, activation='sigmoid'))

        # Compile model with Mean Squared Error loss for regression and Adam optimiser
        self.model.compile(
            optimizer=Adam(1e-5),
            loss='mse',
            metrics=['mae', RootMeanSquaredError()]
        )

        return self.model

    
    def create_dnn_model_regression(self):
        """
        Builds and compiles a Deep Neural Network (DNN) model for regression tasks 
        with EIT boundary voltages to predict relative force at sample points.
        Input is the EIT boundary voltages. Output is the size of number of sample points,
        where each index of the output is a float value representing the relative pressure.

        The model architecture includes:
            - Dense input layer matching the number of electrode pairs.
            - Multiple fully connected layers with ELU activation, L2 regularisation, batch normalisation, 
            and dropout for improved generalisation.
            - Output dense layer with sigmoid activation to predict continuous values.

        Compilation details:
            - Optimiser: Adam with a learning rate of 1e-5.
            - Loss: Mean Squared Error (MSE), suitable for regression problems.
            - Metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

        Returns:
            keras.models.Sequential: The compiled DNN model ready for training.
        """

        self.model = Sequential()

        # Input layer
        self.model.add(Dense(self.num_voltages, input_shape=(self.num_voltages,)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        # Dense Layer 1, elu
        self.model.add(Dense(self.num_voltages * 2, activation='elu', kernel_regularizer=l2(1e-3)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        # Dense Layer 2, elu
        self.model.add(Dense(self.num_voltages, activation='elu', kernel_regularizer=l2(1e-3)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        # Dense Layer 3, elu
        self.model.add(Dense(self.num_classes * 2, activation='elu'))
        self.model.add(BatchNormalization())

        # Output dense layer with sigmoid activation
        self.model.add(Dense(self.num_classes, activation='sigmoid'))

        # Compile the model with MSE loss for regression and Adam optimiser
        self.model.compile(
            optimizer=Adam(1e-5),
            loss='mse',
            metrics=['mae', RootMeanSquaredError()]
        )

        return self.model

    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=2000):
        history = self.model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=128, 
                    callbacks=[])

        return history

    def evaluate_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test)
        return results

    def load_model(self, file_name):
        self.model.load_weights(file_name)

    def save_model(self, file_name):
        self.model.save_weights(file_name)
 

    def transfer_learning_model(self, new_voltages, new_outputs, epochs=100, cnn=False):
        # Make a copy of the model to modify for transfer learning
        transfer_model = self.model
        
        # Freeze the last few layers of the DNN
        if not cnn:
            for layer in transfer_model.layers[:-3]:
                layer.trainable = False  # Freeze layers
            transfer_model.layers[0].trainable = True
            transfer_model.layers[-1].trainable = True
            transfer_model.layers[-3].trainable = True

        # Freeze the last few layers of the CNN
        else:
            for layer in transfer_model.layers[:-3]:
                layer.trainable = False  # Freeze layers
            transfer_model.layers[-1].trainable = True
            transfer_model.layers[-3].trainable = True
            transfer_model.layers[-4].trainable = True
        

        # Compile the model with binary cross-entropy for classification
        transfer_model.compile(optimizer=Adam(learning_rate=1e-3), 
                            loss='binary_crossentropy', 
                            metrics=[jaccard_accuracy, 
                                     hamming_loss, 
                                     example_based_f1, 
                                     RootMeanSquaredError()])
        
        # Train the model on the new data (only the last layers will be trained)
        history = transfer_model.fit(new_voltages, new_outputs, 
                                    epochs=epochs, 
                                    batch_size=16) # smaller number of batch size to account for small dataset
        
        
        return transfer_model, history
        