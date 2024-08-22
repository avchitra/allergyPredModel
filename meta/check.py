import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import numpy as np

# Load the model
model = load_model('allergy_model.h5')

# Print model summary
print("Model Summary:")
model.summary()

# Load the new data
new_data = pd.read_csv('data.csv')

# Print the columns of the new data
print("\nColumns in the new data:")
print(new_data.columns)

# Create the target variable
new_data['HAS_ALLERGY'] = new_data[['SHELLFISH_ALG_START', 'FISH_ALG_START', 'MILK_ALG_START', 
                        'SOY_ALG_START', 'EGG_ALG_START', 'WHEAT_ALG_START', 
                        'PEANUT_ALG_START', 'SESAME_ALG_START', 'TREENUT_ALG_START', 
                        'WALNUT_ALG_START', 'PECAN_ALG_START', 'PISTACH_ALG_START', 
                        'ALMOND_ALG_START', 'BRAZIL_ALG_START', 'HAZELNUT_ALG_START', 
                        'CASHEW_ALG_START']].notna().any(axis=1).astype(int)

# Drop original allergy columns
new_data.drop(columns=['SHELLFISH_ALG_START', 'FISH_ALG_START', 'MILK_ALG_START', 
                 'SOY_ALG_START', 'EGG_ALG_START', 'WHEAT_ALG_START', 
                 'PEANUT_ALG_START', 'SESAME_ALG_START', 'TREENUT_ALG_START', 
                 'WALNUT_ALG_START', 'PECAN_ALG_START', 'PISTACH_ALG_START', 
                 'ALMOND_ALG_START', 'BRAZIL_ALG_START', 'HAZELNUT_ALG_START', 
                 'CASHEW_ALG_START'], inplace=True)

# Convert categorical variables to dummy variables
new_data = pd.get_dummies(new_data, columns=['GENDER_FACTOR', 'RACE_FACTOR', 'ETHNICITY_FACTOR', 'PAYER_FACTOR'], drop_first=True)

# Define features
X = new_data.drop(columns=['HAS_ALLERGY'])

# Print the shape and columns of X
print("\nShape of X:")
print(X.shape)
print("\nColumns of X:")
print(X.columns)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nShape of X_scaled:")
print(X_scaled.shape)

# Don't try to make predictions yet, just print the model's input shape
print("\nModel's input shape:")
print(model.input_shape)

# Stop here for now
print("\nScript completed without making predictions.")