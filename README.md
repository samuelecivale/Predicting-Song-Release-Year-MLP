# Music Year Prediction with MLP

This project predicts the release year and decade of songs using a Multi-Layer Perceptron (MLP) neural network implemented with PyTorch. The model is trained on the **YearPredictionMSD** dataset from the UCI Machine Learning Repository, which includes audio features extracted from the Million Song Dataset.

## Overview

The goal of this project is to predict a song's release year based on its audio features. The project includes two classification tasks:

1. **Year Classification**: Predicting the exact year of release (1922-2011).
2. **Decade Classification**: Predicting the decade of release (e.g., 1920s, 1930s, ..., 2010s).

## Technologies Used

- **Python**: Programming language used for development
- **PyTorch**: Deep learning framework for building and training the MLP model
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical operations
- **Scikit-Learn**: Data preprocessing, feature scaling, and dataset splitting
- **Matplotlib**: Data visualization
- **TensorBoard**: Tracking and visualizing training metrics

## Dataset

The **YearPredictionMSD** dataset contains 515,345 samples with 90 audio features extracted from each song. The target is the year of release.

- **Features**: 90-dimensional vector extracted from each song.
- **Target**: Year of the song's release, ranging from 1922 to 2011.

## Preprocessing

1. **Data Cleaning**: Removed null values.
2. **Sampling**: Retained 50% of the data for faster training.
3. **Normalization**: Standardized features using `StandardScaler` from Scikit-Learn.
4. **Splitting**: Divided data into training, validation, and test sets (80%, 10%, 10%).

## Model Architecture

The MLP model is implemented using PyTorch with the following architecture:

- **Input Layer**: 90 neurons (for the 90 audio features).
- **Hidden Layers**: Three fully connected layers with sizes [128, 64, 32].
- **Output Layer**: For year classification: 90 neurons (representing years 1922-2011). For decade classification: 10 neurons (representing decades).
- **Activation Function**: ReLU for hidden layers, log softmax for the output layer.

## Training

- **Loss Function**: CrossEntropyLoss (for multi-class classification).
- **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.1.
- **Epochs**: 10 epochs.
- **Batch Size**: 64 samples per batch.

## Results

- **Year Classification**: Achieved a validation accuracy of around 9% after 10 epochs, indicating the challenge of predicting the exact year from audio features.
- **Decade Classification**: Achieved better performance, with accuracy reflecting more generalized predictions based on broader time periods.

## Conclusion

The project demonstrates the complexity of predicting exact years from audio features due to overlaps in feature patterns across different years. Predicting decades instead of specific years resulted in better performance, suggesting that more generalized classification tasks may be more suitable for this dataset.

