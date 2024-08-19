import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the saved model
model = load_model('allergy_model.h5')

# Load the test data
test_data = pd.read_csv('Input.csv')

# Preprocess the test data
def preprocess_data(df):
    # Create a binary target variable (if present in the data)
    allergy_cols = ['SHELLFISH_ALG_START', 'FISH_ALG_START', 'MILK_ALG_START', 
                    'SOY_ALG_START', 'EGG_ALG_START', 'WHEAT_ALG_START', 
                    'PEANUT_ALG_START', 'SESAME_ALG_START', 'TREENUT_ALG_START', 
                    'WALNUT_ALG_START', 'PECAN_ALG_START', 'PISTACH_ALG_START', 
                    'ALMOND_ALG_START', 'BRAZIL_ALG_START', 'HAZELNUT_ALG_START', 
                    'CASHEW_ALG_START']
    
    if all(col in df.columns for col in allergy_cols):
        df['HAS_ALLERGY'] = df[allergy_cols].notna().any(axis=1).astype(int)
        df = df.drop(columns=allergy_cols)
    
    # Convert categorical variables to dummy variables
    cat_cols = ['GENDER_FACTOR', 'RACE_FACTOR', 'ETHNICITY_FACTOR', 'PAYER_FACTOR']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

# Preprocess the test data
test_data_processed = preprocess_data(test_data)

# Select features (excluding 'HAS_ALLERGY' if it exists)
X_test = test_data_processed.drop(columns=['HAS_ALLERGY', 'SUBJECT_ID'], errors='ignore')

# Print column names and shape for debugging
print("Columns in X_test:")
print(X_test.columns)
print(f"Shape of X_test: {X_test.shape}")

# Normalize the features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Make predictions
try:
    predictions_prob = model.predict(X_test_scaled)
    predictions = (predictions_prob > 0.5).astype(int).flatten()

    # Add predictions to the original dataframe
    test_data['Predicted_Allergy'] = predictions

    # Print results
    print("\nPredictions:")
    print(test_data[['SUBJECT_ID', 'Predicted_Allergy']])

    # If you want to save the results
    test_data.to_csv('predictions_output.csv', index=False)
    print("\nPredictions have been added to the dataframe and saved to 'predictions_output.csv'")

    # If 'HAS_ALLERGY' column exists in the input, calculate accuracy
    if 'HAS_ALLERGY' in test_data.columns:
        from sklearn.metrics import accuracy_score
        actual = test_data['HAS_ALLERGY']
        accuracy = accuracy_score(actual, predictions)
        print(f"\nAccuracy on test data: {accuracy:.2f}")

except Exception as e:
    print(f"An error occurred during prediction: {str(e)}")
    print(f"Model input shape: {model.input_shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")