import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the model
model = load_model('allergyPredModel/allergy_model.h5')

# Load the new data
new_data = pd.read_csv('allergyPredModel/Input.csv')

# One-hot encode categorical variables
categorical_cols = ['GENDER_FACTOR', 'RACE_FACTOR', 'ETHNICITY_FACTOR', 'PAYER_FACTOR']
new_data = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

# Create the target variable
new_data['HAS_ALLERGY'] = new_data[['SHELLFISH_ALG_START', 'FISH_ALG_START', 'MILK_ALG_START', 
                        'SOY_ALG_START', 'EGG_ALG_START', 'WHEAT_ALG_START', 
                        'PEANUT_ALG_START', 'SESAME_ALG_START', 'TREENUT_ALG_START', 
                        'WALNUT_ALG_START', 'PECAN_ALG_START', 'PISTACH_ALG_START', 
                        'ALMOND_ALG_START', 'BRAZIL_ALG_START', 'HAZELNUT_ALG_START', 
                        'CASHEW_ALG_START']].notna().any(axis=1).astype(int)

# Split data into features (X) and target (y)
X = new_data.drop(['SHELLFISH_ALG_START', 'FISH_ALG_START', 'MILK_ALG_START', 
                 'SOY_ALG_START', 'EGG_ALG_START', 'WHEAT_ALG_START', 
                 'PEANUT_ALG_START', 'SESAME_ALG_START', 'TREENUT_ALG_START', 
                 'WALNUT_ALG_START', 'PECAN_ALG_START', 'PISTACH_ALG_START', 
                 'ALMOND_ALG_START', 'BRAZIL_ALG_START', 'HAZELNUT_ALG_START', 
                 'CASHEW_ALG_START', 'HAS_ALLERGY'], axis=1)

# Ensure that the new data has the same columns as the model's input
model_input_shape = model.input_shape[1]  # Get the expected number of input features
missing_cols = set(X.columns) - set(X.columns)

# Add any missing columns as zero
for col in missing_cols:
    X[col] = 0

# Reorder columns to match the model's input shape
X = X.reindex(sorted(X.columns), axis=1)

# Check if all expected columns are present
if X.shape[1] != model_input_shape:
    raise ValueError(f"Expected {model_input_shape} input features, but got {X.shape[1]} features. Check your preprocessing steps.")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Make predictions
new_predictions = model.predict(X_scaled)
new_predictions = (new_predictions > 0.5).astype(int).flatten()

# Print the predictions
print(new_predictions)
