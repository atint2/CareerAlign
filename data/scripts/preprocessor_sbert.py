import re

class SBERTPreprocessor:
    """Data Preprocessor for job descriptions and resumes -- SBERT model"""
    
    @staticmethod
    def clean_text_sbert(text):
        """Clean text for SBERT model with minimal preprocessing"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', "", text)
        
        # Remove phone numbers
        text = re.sub(r'\+?\d[\d\s\-\(\)]{8,}\d', "", text)

        # Clean up leftover artifacts from URLs, emails, phone numbers
        text = re.sub(r'\b(email|contact|visit|call|phone):\s*', '', text, flags=re.IGNORECASE)
        
        # Remove pipe characters
        text = text.replace('|', ' ')

        # Remove excessive whitespace (but preserve single spaces)
        text = " ".join(text.split())
        
        return text.strip()