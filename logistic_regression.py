import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('C:/Users/ayush/OneDrive/UB/DIC/Project/result.csv')
labels = dataset['Attrition_Flag']
features = dataset.drop(['Attrition_Flag'], axis=1)

# Label Encoding
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    features[column] = label_encoders[column].fit_transform(features[column])

# Normalizing features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

train_features, test_features, train_labels, test_labels = train_test_split(features_normalized, labels, test_size=0.20, random_state=42)

#Logistic Regression model with weighted training
best_model = LogisticRegression()
# Grid search with cross-validation
best_model = best_model.fit(train_features, train_labels)

# Save the best model
import joblib
joblib.dump(best_model, 'best_model.pkl')
