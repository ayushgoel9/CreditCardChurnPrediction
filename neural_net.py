#{'activation': 'relu', 'hidden_layer_sizes': (19, 19), 'solver': 'adam'}
#Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
# from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from imblearn.combine import SMOTEENN

# Load the dataset from the CSV file
dataset = pd.read_csv('result.csv')
labels = dataset['Attrition_Flag']
features = dataset.drop(['Attrition_Flag'], axis=1)#Assuming 'Attrition_Flag' is the label column so dropping that column

# Convert categorical variables to numerical representations using Label Encoding
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    features[column] = label_encoders[column].fit_transform(features[column])

#Normailizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Doing sampling and resampling
# smote_enn = SMOTEENN(random_state=42)
# features_resampled, labels_resampled = smote_enn.fit_resample(features_scaled, labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# create neural network model
clf = MLPClassifier(hidden_layer_sizes=(19,19), activation='relu', solver='adam', max_iter=1000)

# Fit model
model=clf.fit(X_train, y_train)

# Save the best model
import joblib
joblib.dump(model,'model.pkl')