# Music Year Prediction with MLP

This project aims to predict the release year of songs based on their audio features using a Multi-Layer Perceptron (MLP) neural network implemented with PyTorch. The dataset used for training and testing is the YearPredictionMSD dataset from the UCI Machine Learning Repository, which contains features extracted from the Million Song Dataset.

## Project Overview

The goal of this project is to build a machine learning model that can predict the year of a song's release based on its audio features. The model takes input features derived from audio signals and outputs a prediction for the song's release year, which is treated as a classification problem with multiple classes corresponding to each year from 1922 to 2011.

## Technologies Used

- **Python**: The programming language used for implementing the project.
- **PyTorch**: The deep learning framework used to build and train the MLP model.
- **Pandas**: Used for data manipulation and preprocessing.
- **NumPy**: Utilized for numerical operations.
- **Scikit-Learn**: Used for data preprocessing, scaling features, and splitting the dataset.
- **Matplotlib**: Used for visualizing the data distribution and training results.
- **TensorBoard**: Used for tracking the training process and visualizing metrics.

## Dataset

The dataset used is the **YearPredictionMSD** dataset, which is part of the UCI Machine Learning Repository. It contains 515,345 samples with 90 audio features extracted from each song, and the target variable is the year of release.

- **Features**: 90-dimensional vector extracted from each song.
- **Target**: Year of the song's release, ranging from 1922 to 2011.

## Preprocessing

1. **Data Cleaning**: Removed any null values from the dataset.
2. **Sampling**: Used a random sampling to retain 50% of the data for faster training.
3. **Normalization**: Standardized the features using `StandardScaler` from Scikit-Learn.
4. **Splitting**: Divided the data into training, validation, and test sets with proportions 80%, 10%, and 10%, respectively.

## Model Architecture

The MLP model is implemented using PyTorch. The architecture consists of the following:

- Input layer: 90 neurons (corresponding to the 90 audio features).
- Hidden layers: Three fully connected layers with sizes [128, 64, 32].
- Output layer: 90 neurons (each representing a year from 1922 to 2011).
- Activation function: ReLU for hidden layers, and log softmax for the output layer.

## Training

- **Loss Function**: CrossEntropyLoss, which is suitable for multi-class classification.
- **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.1.
- **Epochs**: Trained for 10 epochs.
- **Batch Size**: 64 samples per batch.

## Results

- The model achieved a validation accuracy of around 9% after 10 epochs. The low accuracy suggests that predicting the exact release year based on the given audio features is a challenging problem, potentially due to the overlap and similarities in audio features across different years.
- Training and validation loss decreased over time, indicating that the model was learning, but the complexity of the task and the dataset might require more sophisticated architectures or additional features for better performance.


