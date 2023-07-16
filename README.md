# Diabetes_classification_using_neural_network

This repository contains code for training an Artificial Neural Network (ANN) model to predict the presence of diabetes in patients based on certain diagnostic measurements. The model achieves an accuracy of 80% on the provided dataset.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database. It is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to predict whether or not a patient has diabetes based on the following diagnostic measurements:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skinfold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function (a measure of the genetic influence)
- Age: Age in years

The dataset consists of samples from female patients who are at least 21 years old and of Pima Indian heritage. The goal is to build a model that can predict whether a patient has diabetes or not based on these features.

## Model

The model used in this project is an Artificial Neural Network (ANN) implemented using the PyTorch framework. The ANN architecture consists of three hidden layers with 40, 100, and 20 neurons respectively. The output layer has two neurons representing the binary classification of "Diabetic" and "No Diabetic" outcomes.

The model is trained using the Adam optimizer and the Cross Entropy Loss function. The training is performed for 500 epochs with a learning rate of 0.01. During training, the loss is monitored and plotted to observe the model's learning progress.

## Evaluation

The trained model is evaluated on a separate testing set. The evaluation metrics used are a confusion matrix and accuracy score. The confusion matrix provides insights into the model's performance by showing the true positive, true negative, false positive, and false negative predictions. The accuracy score represents the percentage of correct predictions made by the model.

The model achieves an accuracy of 80% on the testing set, indicating its effectiveness in predicting the presence of diabetes based on the given diagnostic measurements.

## Usage

To use the trained model for making predictions on new data, the saved model file "diabetes.pt" can be loaded and the ANN model can be invoked with the input data. The model will output the predicted label, indicating whether the patient is diabetic or not.

Please refer to the provided code and comments for more details on how to train the model, evaluate its performance, and use it for making predictions.

Note: It is recommended to install the required dependencies before running the code. The necessary libraries can be installed using the `requirements.txt` file.

## Conclusion

The ANN model trained on the Pima Indians Diabetes Database achieves an accuracy of 80% in predicting the presence of diabetes based on diagnostic measurements. The model can be used as a tool for early detection and risk assessment of diabetes in patients. However, further analysis and validation may be required before deploying the model in a real-world setting.

Feel free to explore the code and dataset provided in this repository for a better understanding of the project and to further improve the model's performance.
