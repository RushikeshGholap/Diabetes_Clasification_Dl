# Diabetes Prediction using Deep Learning

## Project Overview

This project explores the application of deep learning techniques, specifically Multi-Layer Perceptron (MLP) classifiers, to predict diabetes. Our goal is to develop an accurate and robust model that can assist healthcare professionals in early diabetes detection and intervention.

MLP Architecture
*Figure 1: Multi-Layer Perceptron Architecture*

## Key Features

- Utilizes Multi-Layer Perceptron (MLP) architecture
- Implements Xavier Initialization and Adam optimization
- Explores various activation functions and error metrics
- Achieves high accuracy in diabetes prediction (90.79%)

## Dataset

The project uses a comprehensive dataset of health parameters, including:
- Age
- Body Mass Index (BMI)
- Blood Pressure
- Cholesterol Levels

The dataset is preprocessed, cleaned, and balanced using random under-sampling for the majority class.

Data Distribution
*Figure 2: Distribution of key features in the dataset*

## Model Architecture

Our best-performing model features:
- 4 layers MLP
- Hidden layers with 15, 32, and 16 neurons respectively
- Xavier Initialization
- Adam Optimizer
- Learning rate of 0.1

```python
class DiabetesMLP(nn.Module):
    def __init__(self):
        super(DiabetesMLP, self).__init__()
        self.fc1 = nn.Linear(8, 15)
        self.fc2 = nn.Linear(15, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
```

## Results

The model achieves:
- Accuracy: 90.79%
- Precision: 91%
- Recall: 88%
- F1-Score: 90%

Confusion Matrix
*Figure 3: Confusion Matrix of the best-performing model*

## Comparison with Other Models

Our MLP model outperforms other machine learning models such as Decision Trees, Naive Bayes, KNN, and Ensemble voting in terms of accuracy and overall performance.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MLP   | 90.79%   | 91%       | 88%    | 90%      |
| SVM   | 85.23%   | 86%       | 84%    | 85%      |
| Random Forest | 87.45% | 88% | 86%    | 87%      |

## How to Use

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/diabetes-prediction-mlp.git
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the main script to train and evaluate the model:
   ```
   python main.py
   ```

4. Use the trained model for diabetes prediction on new data:
   ```python
   from model import DiabetesMLP
   
   model = DiabetesMLP()
   model.load_state_dict(torch.load('best_model.pth'))
   
   # Make predictions
   predictions = model(new_data)
   ```

## Future Work

- Incorporate temporal analysis for dynamic healthcare data
- Explore advanced architectures like RNNs and attention mechanisms
- Integrate multimodal data sources for enhanced prediction

## Contributors

- Rushikesh Gholap
- Tushar Jain
- Sushmitha Rajeswari Muppa
- Vedavarshita Nunna
