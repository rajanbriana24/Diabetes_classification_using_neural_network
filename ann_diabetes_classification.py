# Importing necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data processing, CSV file I/O

# Reading the dataset from CSV file
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

# Checking for missing values
df.isnull().sum()

# Data visualization
import seaborn as sns  # For data visualization

# Mapping the 'Outcome' column to meaningful labels
df['Outcome'] = np.where(df['Outcome'] == 1, "Diabetic", "No Diabetic")

# Creating a pair plot of the dataset
sns.pairplot(df, hue='Outcome')

# Resampling the dataset using SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=10)

# Creating an Artificial Neural Network (ANN) model using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_Model(nn.Module):
    def __init__(self, input_features=8, hidden1=40, hidden2=100, hidden3=20, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.f_connected3 = nn.Linear(hidden2, hidden3)
        self.out = nn.Linear(hidden3, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = F.relu(self.f_connected3(x))
        x = self.out(x)
        return x

torch.manual_seed(20)
model = ANN_Model()

# Defining the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
epochs = 500
final_losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)  # Forward pass
    loss = loss_function(y_pred, y_train)  # Calculating the loss
    final_losses.append(loss.item())

    if i % 10 == 1:
        print("Epoch number: {} and the loss: {}".format(i, loss.item()))

    optimizer.zero_grad()  # Clearing the gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Updating the weights

# Plotting the loss function
import matplotlib.pyplot as plt

plt.plot(range(epochs), final_losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')

# Evaluating the model on the testing set
with torch.no_grad():
    predictions = []
    for data in X_test:
        y_pred = model(data)
        pred = y_pred.argmax().item()
        predictions.append(pred)

from sklearn.metrics import confusion_matrix, accuracy_score

# Creating a confusion matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

# Calculating the accuracy score
score = accuracy_score(y_test, predictions)
print("Accuracy Score:", score)

# Saving the trained model
torch.save(model, 'diabetes.pt')

# Making predictions on new data
new_data = torch.tensor([6.0, 148.0, 24.0, 35.0, 0.0, 33.6, 0.927, 50.0])

with torch.no_grad():
    y_pred = model(new_data)
    print("Predicted Label:", y_pred.argmax().item())
