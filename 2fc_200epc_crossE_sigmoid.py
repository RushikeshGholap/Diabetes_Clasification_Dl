import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from Layers.FullyConnectedLayer import FullyConnectedLayer
from Objective_Functions.CrossEntropy import CrossEntropy
from Activation_Layers.LogisticSigmoidLayer import LogisticSigmoidLayer
import numpy as np
import matplotlib.pyplot as plt
from Metrics.Metrics import *
from Preprocess.preprocessing import load_and_preprocess_diabetes_dataset

np.random.seed(13)

def train_test_split(X, y, test_size=0.3, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]



# Load dataset
X, y, num_classes = load_and_preprocess_diabetes_dataset(r'data\diabetes_prediction_dataset.csv')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)


# Training loop
loss_history = []
validation_loss_history = []  
epochs = 200
learning_rate = 0.1

# Initialize layers
no_layer = 2
input_size = X.shape[1] 
fc_layer1 = FullyConnectedLayer(sizeIn=input_size, sizeOut=32)
sigmoid_layer1 = LogisticSigmoidLayer()
fc_layer2 = FullyConnectedLayer(sizeIn=32, sizeOut=1)
sigmoid_layer_output = LogisticSigmoidLayer()  
crossentropy = CrossEntropy()

for epoch in range(epochs):
    epoch_loss = 0
    for x, y_true in zip(X_train, y_train):
        x = x.reshape(1, -1)  
        y_true = y_true.reshape(1, -1) 
        
        # Forward pass
        x = fc_layer1.forward(x)
        x = sigmoid_layer1.forward(x)
        x = fc_layer2.forward(x)
        y_pred = sigmoid_layer_output.forward(x)
        
        # Compute loss
        loss = crossentropy.eval(y_pred, y_true)
        
        # Backward pass
        d_loss = crossentropy.gradient(y_pred, y_true)
        d_loss = sigmoid_layer_output.backward(d_loss)
        d_loss = fc_layer2.backward(d_loss)
        d_loss = sigmoid_layer1.backward(d_loss)
        d_loss = fc_layer1.backward(d_loss)
        
        # Update weights
        fc_layer1.updateWeights(learning_rate)
        fc_layer2.updateWeights(learning_rate)
        
        epoch_loss += loss
  
    val_loss = 0
    for x_val, y_val_true in zip(X_test, y_test):
        x_val = x_val.reshape(1, -1)
        y_val_true = y_val_true.reshape(1, -1)
        
        # Forward pass only, no backward pass or weight update
        x_val = fc_layer1.forward(x_val)
        x_val = sigmoid_layer1.forward(x_val)
        x_val = fc_layer2.forward(x_val)
        y_val_pred = sigmoid_layer1.forward(x_val)
     
        
        # Compute validation loss
        val_loss += crossentropy.eval(y_val_pred, y_val_true)
    
    avg_val_loss = val_loss / len(y_test)
    validation_loss_history.append(avg_val_loss)


    avg_epoch_loss = epoch_loss / len(y)
    loss_history.append(avg_epoch_loss)
    print(f'Epoch {epoch+1}, Training Loss: {avg_epoch_loss}, Validation Loss: {avg_val_loss}')

plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.plot(validation_loss_history, label='Validation Loss')
plt.title(f'Train and Valid Loss over Epochs; {no_layer} FC Lyr with LR: {learning_rate} & sigmoid')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./plots/{no_layer}fc_{epochs}epc_training_validation_loss_sigmoid.png')


def predict(X):
    predictions = []
    for x in X:
        x = x.reshape(1, -1)
        x = fc_layer1.forward(x)
        x = sigmoid_layer1.forward(x)
        x = fc_layer2.forward(x)
        y_pred = sigmoid_layer_output.forward(x)
        predictions.append(y_pred.flatten()[0])
    return np.array(predictions)

# Predict and evaluate
y_pred = (predict(X_test)> 0.5).astype(int)
classification  = classification_report(y_test, y_pred)
print("Confusion Matrix:\n", classification)


