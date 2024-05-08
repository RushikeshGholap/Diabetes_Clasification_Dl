
import pandas as pd
import numpy as np 

def load_and_preprocess_diabetes_dataset(dataset_path):
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'])
    
    # Standardize numerical columns
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean(axis=0)) / df[numerical_cols].std(axis=0)
    
    # Separate features and target
    X = df.drop(columns=['diabetes']).values
    y = df['diabetes'].values

    # Determine class counts and identify minority class
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    minority_size = class_counts[minority_class]
    

    majority_size = int((50 / 50) * minority_size)  
    
    # Indices of each class' observations
    indices_minority = np.where(y == minority_class)[0]
    indices_majority = np.where(y != minority_class)[0]
    
    # Randomly sample from majority class to achieve the desired ratio
    np.random.seed(13)  # For reproducibility
    indices_majority_downsampled = np.random.choice(indices_majority, size=majority_size, replace=False)
    
    # Combine minority class with downsampled majority class
    indices_new = np.concatenate([indices_minority, indices_majority_downsampled])
    
    # Extract new balanced dataset
    X_new = X[indices_new]
    y_new = y[indices_new]
    
    return X_new, y_new, len(np.unique(y_new)) 
