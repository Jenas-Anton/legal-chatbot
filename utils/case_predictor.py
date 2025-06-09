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
        
        # Create pipeline
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
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def train_model(self, summarized_cases):
        """Train the model using summarized cases"""
        if not summarized_cases:
            raise ValueError("No training data provided")
        
        texts = []
        labels = []
        
        for case in summarized_cases:
            if isinstance(case, dict) and 'summary' in case:
                summary = case['summary']
                if isinstance(summary, dict):
                    texts.append(self._preprocess_text(summary.get('reasoning', '')))
                    labels.append(summary.get('outcome', 'Unknown'))
                else:
                    texts.append(self._preprocess_text(str(summary)))
                    if 'dismissed' in str(summary).lower():
                        labels.append('Appeal Dismissed')
                    elif 'allowed' in str(summary).lower():
                        labels.append('Appeal Allowed')
                    else:
                        labels.append('Unknown')
        
        if texts and labels:
            self.pipeline.fit(texts, labels)
        else:
            raise ValueError("No valid training data extracted from summarized cases")

    def explain_prediction(self, text, prediction, confidence):
        """Generate SHAP explanation for prediction"""
        try:
            vectorizer = self.pipeline.named_steps['vectorizer']
            classifier = self.pipeline.named_steps['classifier']
            
            X = vectorizer.transform([text])
            X_dense = X.toarray()
            
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_dense)
            
            feature_names = vectorizer.get_feature_names_out()
            
            if isinstance(shap_values, list):
                class_idx = np.where(classifier.classes_ == prediction)[0][0]
                shap_values_class = shap_values[class_idx][0]
            else:
                shap_values_class = shap_values[0]
            
            shap_values_class = np.array(shap_values_class).flatten()
            
            feature_importance = []
            for idx, (feature, importance) in enumerate(zip(feature_names, shap_values_class)):
                imp_value = importance.item() if hasattr(importance, 'item') else float(importance)
                if not np.isnan(imp_value):
                    feature_importance.append((feature, imp_value))
            
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feature_importance[:10]
            
            explanation = {
                "top_features": [
                    {
                        "feature": str(feature),
                        "importance": float(importance),
                        "effect": "positive" if importance > 0 else "negative"
                    }
                    for feature, importance in top_features
                ]
            }
            
            return explanation
            
        except Exception as e:
            return {"top_features": []}

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
            self.train_model(summarized_cases)
            
            processed_text = self._preprocess_text(case_text)
            
            prediction_proba = self.pipeline.predict_proba([processed_text])[0]
            prediction = self.pipeline.predict([processed_text])[0]
            confidence = float(max(prediction_proba))
            
            try:
                explanation = self.explain_prediction(processed_text, prediction, confidence)
            except Exception:
                explanation = {"top_features": []}
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "explanation": explanation
            }
                
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}") 