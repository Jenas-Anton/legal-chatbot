import os
from utils.case_predictor import CasePredictor

def main():
    # Replace with your actual API key
    api_key = "YOUR_GOOGLE_API_KEY"
    
    # Sample training data
    sample_cases = [
        {
            "summary": {
                "reasoning": """
                The appellant challenged the lower court's decision regarding property rights.
                The court found insufficient evidence to support the appellant's claims.
                The documentation provided was incomplete and inconsistent.
                """,
                "outcome": "Appeal Dismissed"
            }
        },
        {
            "summary": {
                "reasoning": """
                The appellant provided strong evidence of procedural irregularities.
                The lower court failed to consider key witness testimonies.
                New evidence emerged that substantially supported the appellant's position.
                """,
                "outcome": "Appeal Allowed"
            }
        },
        {
            "summary": {
                "reasoning": """
                The court found multiple violations of statutory requirements.
                The respondent failed to comply with regulatory guidelines.
                Clear evidence of negligence was established through documentation.
                """,
                "outcome": "Appeal Allowed"
            }
        }
    ]

    try:
        # Initialize the predictor
        print("Initializing Case Predictor...")
        predictor = CasePredictor(api_key)

        # Test case text
        test_case = """
        The appellant filed an appeal against the decision of the lower court
        regarding a contract dispute. The evidence presented shows that the
        contract terms were clearly violated, with substantial documentation
        supporting the breach. Multiple witnesses testified to the respondent's
        failure to fulfill contractual obligations. The appellant has provided
        comprehensive financial records demonstrating damages.
        """

        # Make prediction
        print("\nMaking prediction for test case...")
        result = predictor.predict_case(test_case, sample_cases)

        # Print results
        print("\nPrediction Results:")
        print(f"Predicted Outcome: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        print("\nTop Contributing Factors:")
        for feature in result['explanation']['top_features']:
            effect = "+" if feature['effect'] == "positive" else "-"
            print(f"{effect} {feature['feature']}: {abs(feature['importance']):.4f}")

        # Test PDF functionality (if you have a PDF file)
        """
        pdf_path = "path/to/your/case.pdf"
        if os.path.exists(pdf_path):
            print("\nTesting PDF extraction...")
            pdf_text = CasePredictor.extract_text_from_pdf(pdf_path)
            pdf_result = predictor.predict_case(pdf_text, sample_cases)
            print(f"PDF Case Prediction: {pdf_result['prediction']}")
        """

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 