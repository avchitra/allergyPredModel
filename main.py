import pandas as pd
import numpy as np
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