# Proj_12A_Mobiles_Prices_Neural_Network

## Introduction
This project is designed to predict the price range of mobile phones. It uses various mobile features such as battery, RAM, internal memory, screen size, camera, etc., and applies a neural network model for prediction.

## Purpose
The main objectives of this project are:
- To accurately predict the mobile price range based on given features.
- To understand and implement applications of neural networks in data science and machine learning.
- To assist in mobile market analysis.

## Features
- Uses technical features of mobiles like battery_power, ram, int_memory, px_height, px_width, four_g, three_g, dual_sim, etc.
- Data is stored in a CSV file.
- Machine learning model (Neural Network) used for prediction.
- Model prepared for training, testing, and validation.
- Results can be visualized.

## Usage Instructions
 **Prerequisites**  
   - Python should be installed.
   - Required libraries: numpy, pandas, scikit-learn, tensorflow/keras (if using neural networks).

 **Load Data**  
   - Load `mobile_prices.csv` using pandas:
     ```python
     import pandas as pd
     data = pd.read_csv('mobile_prices.csv')
     ```

 **Data Processing**  
   - Scale/normalize all features.
   - Split data into train and test sets.

 **Model Training**  
   - Build a Neural Network model (e.g., using keras/tensorflow).
   - Train the model:
     ```python
     # Example
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Dense

     model = Sequential()
     model.add(Dense(64, activation='relu', input_dim=20))
     model.add(Dense(4, activation='softmax'))  # 4 price_range classes
     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
     # Data X, y should be defined
     model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
     ```

 **Results and Analysis**  
   - Check model performance on test data.
   - View confusion matrix, accuracy, etc.

 **Make Predictions**  
   - Input new mobile features to predict its price range.
