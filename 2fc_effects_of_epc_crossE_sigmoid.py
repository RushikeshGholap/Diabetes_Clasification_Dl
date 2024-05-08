import numpy as np
import pandas as pd
from Layers.FullyConnectedLayer import FullyConnectedLayer
from Objective_Functions.CrossEntropy import CrossEntropy
from Activation_Layers.LogisticSigmoidLayer import LogisticSigmoidLayer
import numpy as np
import matplotlib.pyplot as plt
from Preprocess.preprocessing import load_and_preprocess_diabetes_dataset

np.random.seed(13)

def train_test_split(X, y, test_size=0.3, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Load  dataset
X, y, num_classes = load_and_preprocess_diabetes_dataset(r'data\diabetes_prediction_dataset.csv')
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)


def train_and_evaluate(X_train, y_train, epochs, learning_rate):
    # Initialize layers with the specified learning rate where applicable
    input_size = X_train.shape[1]
    fc_layer1 = FullyConnectedLayer(sizeIn=input_size, sizeOut=32)
    sigmoid_layer1 = LogisticSigmoidLayer()
    fc_layer2 = FullyConnectedLayer(sizeIn=32, sizeOut=1)
    sigmoid_layer_output = LogisticSigmoidLayer()
    crossentropy = CrossEntropy()

    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y_true in zip(X_train, y_train.reshape(-1, 1)): 
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

        avg_epoch_loss = epoch_loss / len(y_train)
        loss_history.append(avg_epoch_loss)
        print(f'LR: {learning_rate} Epoch {epoch+1}, Average Loss: {avg_epoch_loss}')
    
    return loss_history

epochs = 20

plt.figure(figsize=(10, 6))

# Iterate over learning rates 
for lr in [0.01, 0.001, 0.0001,2,4]:
    loss_history = train_and_evaluate(X_train, y_train, epochs, lr)
    plt.plot(range(1, epochs + 1), loss_history, label=f'LR: {lr}')

plt.title('Effect of Learning Rate on Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.savefig('./plots/effects_of_lr.png')