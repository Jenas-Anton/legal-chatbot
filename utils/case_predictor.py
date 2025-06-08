import os
import PyPDF2
import docx
import google.generativeai as genai
import json
import shap
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
import re

# Suppress warnings
warnings.filterwarnings('ignore')

class CasePredictor:
    def __init__(self, api_key):
        """Initialize the predictor with API key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a new pipeline for each instance
        print("Creating new model...")
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, 
                                         strip_accents='unicode',
                                         lowercase=True,
                                         analyzer='word',
                                         stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

    def _preprocess_text(self, text):
        """Preprocess text for model input"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def train_model(self, summarized_cases):
        """Train the model using summarized cases"""
        if not summarized_cases:
            raise ValueError("No training data provided")
        
        print(f"Training with {len(summarized_cases)} cases...")
        
        # Extract text and labels from summarized cases
        texts = []
        labels = []
        
        for case in summarized_cases:
            if isinstance(case, dict) and 'summary' in case:
                summary = case['summary']
                # Try to extract outcome from summary
                if isinstance(summary, dict):
                    texts.append(self._preprocess_text(summary.get('reasoning', '')))
                    labels.append(summary.get('outcome', 'Unknown'))
                else:
                    # If summary is not a dict, use the whole text
                    texts.append(self._preprocess_text(str(summary)))
                    # Try to extract outcome from text
                    if 'dismissed' in str(summary).lower():
                        labels.append('Appeal Dismissed')
                    elif 'allowed' in str(summary).lower():
                        labels.append('Appeal Allowed')
                    else:
                        labels.append('Unknown')
        
        if texts and labels:
            self.pipeline.fit(texts, labels)
            print("Model training completed.")
        else:
            raise ValueError("No valid training data extracted from summarized cases")

    def explain_prediction(self, text, prediction, confidence):
        """Generate SHAP explanation for prediction"""
        try:
            # Get vectorizer and classifier from pipeline
            vectorizer = self.pipeline.named_steps['vectorizer']
            classifier = self.pipeline.named_steps['classifier']
            
            # Transform text to sparse matrix
            X = vectorizer.transform([text])
            
            # Convert sparse matrix to dense array
            X_dense = X.toarray()
            
            # Create explainer
            explainer = shap.TreeExplainer(classifier)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_dense)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For multi-class
                class_idx = np.where(classifier.classes_ == prediction)[0][0]
                shap_values_class = shap_values[class_idx][0]
            else:
                # For binary classification
                shap_values_class = shap_values[0]
            
            # Convert to numpy array and ensure it's 1D
            shap_values_class = np.array(shap_values_class).flatten()
            
            # Create feature importance pairs, handling one value at a time
            feature_importance = []
            for idx, (feature, importance) in enumerate(zip(feature_names, shap_values_class)):
                # Convert each importance value to a Python scalar
                imp_value = importance.item() if hasattr(importance, 'item') else float(importance)
                if not np.isnan(imp_value):
                    feature_importance.append((feature, imp_value))
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top 10 features
            top_features = feature_importance[:10]
            
            explanation = {
                "top_features": [
                    {
                        "feature": str(feature),  # Ensure feature name is string
                        "importance": float(importance),  # Ensure importance is float
                        "effect": "positive" if importance > 0 else "negative"
                    }
                    for feature, importance in top_features
                ]
            }
            
            return explanation
            
        except Exception as e:
            print(f"Debug - Error in explain_prediction: {str(e)}")
            print(f"Debug - Type of shap_values: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"Debug - Shape of first shap_values: {np.array(shap_values[0]).shape}")
            print(f"Debug - Classifier classes: {classifier.classes_}")
            print(f"Debug - Prediction value: {prediction}")
            print(f"Debug - SHAP values shape: {np.array(shap_values).shape if hasattr(shap_values, 'shape') else 'no shape'}")
            raise Exception(f"Explanation generation failed: {str(e)}")

    @staticmethod
    def extract_text_from_pdf(file_path):
        """Extract text from a PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    @staticmethod
    def extract_text_from_doc(file_path):
        """Extract text from a DOC/DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading DOC/DOCX file: {str(e)}")

    def predict_case(self, case_text, summarized_cases):
        """Predict case outcome using the provided text and similar cases"""
        try:
            # Always train a new model with the current cases
            self.train_model(summarized_cases)
            
            # Preprocess the case text
            processed_text = self._preprocess_text(case_text)
            
            # Make prediction
            prediction_proba = self.pipeline.predict_proba([processed_text])[0]
            prediction = self.pipeline.predict([processed_text])[0]
            confidence = float(max(prediction_proba))
            
            # Generate explanation
            try:
                explanation = self.explain_prediction(processed_text, prediction, confidence)
            except Exception as e:
                print(f"Warning: Could not generate detailed explanation: {str(e)}")
                explanation = {"top_features": []}
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "explanation": explanation
            }
                
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}") 