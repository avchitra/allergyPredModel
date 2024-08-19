import pandas as pd
import numpy as np
<<<<<<< HEAD
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
import joblib
import os
from typing import List, Tuple, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AllergyPredictor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.label_encoders = {}
        self.allergy_types = [
            'SHELLFISH', 'FISH', 'MILK', 'SOY', 'EGG', 'WHEAT', 'PEANUT', 'SESAME', 
            'TREENUT', 'WALNUT', 'PECAN', 'PISTACH', 'ALMOND', 'BRAZIL', 'HAZELNUT', 'CASHEW'
        ]

    def load_data(self) -> None:
        """Load the CSV data."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully from {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self) -> None:
        """Preprocess the data including handling missing values and encoding."""
        try:
            # Handle missing values
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    self.data[col].fillna('Unknown', inplace=True)
                else:
                    self.data[col].fillna(self.data[col].median(), inplace=True)

            # Encode categorical variables
            categorical_columns = ['GENDER_FACTOR', 'RACE_FACTOR', 'ETHNICITY_FACTOR', 'PAYER_FACTOR', 'ATOPIC_MARCH_COHORT']
            for col in categorical_columns:
                self.label_encoders[col] = LabelEncoder()
                self.data[f'{col}_ENCODED'] = self.label_encoders[col].fit_transform(self.data[col])

            # Create binary indicators for allergies
            for allergy in self.allergy_types:
                self.data[f'{allergy}_ALG_BINARY'] = (self.data[f'{allergy}_ALG_START'].notna()).astype(int)

            # Feature engineering
            self.data['TOTAL_ALLERGIES'] = self.data[[f'{allergy}_ALG_BINARY' for allergy in self.allergy_types]].sum(axis=1)
            self.data['AGE_RANGE'] = self.data['AGE_END_YEARS'] - self.data['AGE_START_YEARS']

            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def prepare_features(self) -> None:
        """Prepare features and target variable."""
        try:
            features = [
                'GENDER_FACTOR_ENCODED', 'RACE_FACTOR_ENCODED', 'ETHNICITY_FACTOR_ENCODED',
                'PAYER_FACTOR_ENCODED', 'ATOPIC_MARCH_COHORT_ENCODED', 'BIRTH_YEAR',
                'AGE_START_YEARS', 'AGE_END_YEARS', 'AGE_RANGE', 'TOTAL_ALLERGIES'
            ]
            self.X = self.data[features]
            self.y = self.data['TOTAL_ALLERGIES'] > 0  # Predicting any allergy
            logger.info("Features and target variable prepared successfully")
        except Exception as e:
            logger.error(f"Error in preparing features: {str(e)}")
            raise

    def train_model(self) -> None:
        """Train the model using GridSearchCV for hyperparameter tuning."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', RandomForestClassifier(random_state=42))
            ])

            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }

            grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
            logger.info(f"ROC AUC Score: {roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])}")

            # Feature importance
            importance = self.model.named_steps['classifier'].feature_importances_
            for i, v in enumerate(importance):
                logger.info(f'Feature: {self.X.columns[i]}, Score: {v}')

            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Save the trained model to a file."""
        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        """Load a trained model from a file."""
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        try:
            return self.model.predict(features)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

def main():
    try:
        data_path = 'path/to/your/data.csv'
        model_path = 'path/to/save/model.joblib'

        predictor = AllergyPredictor(data_path)
        predictor.load_data()
        predictor.preprocess_data()
        predictor.prepare_features()
        predictor.train_model()
        predictor.save_model(model_path)

        # Example of loading the model and making predictions
        new_data = pd.DataFrame(...)  # Your new data here
        predictor.load_model(model_path)
        predictions = predictor.predict(new_data)
        logger.info(f"Predictions: {predictions}")

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    main()
=======
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
>>>>>>> tmp
