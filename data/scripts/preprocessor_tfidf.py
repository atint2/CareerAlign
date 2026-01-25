import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json

# Download required NLTK data (run once)
#nltk.download('stopwords')
#nltk.download('punkt_tab')

class TFIDFPreprocessor:
    """Data Preprocessor for job descriptions and resumes -- TF-IDF model"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words -= {'senior', 'junior', 'lead', 'principal', 'staff', 
                           'full', 'part', 'remote', 'hybrid', 'not', 'no'}

    def clean_text_tfidf(self, text):
        """Clean text for TF-IDF model with comprehensive preprocessing"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert text to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', "", text)
        
        # Remove phone numbers
        text = re.sub(r'\+?\d[\d\s\-\(\)]{8,}\d', "", text)

        # Clean up leftover artifacts from URLs, emails, phone numbers
        text = re.sub(r'\b(email|contact|visit|call|phone):\s*', '', text, flags=re.IGNORECASE)
        
        # Remove stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        filtered_text = " ".join(filtered_words)

        return filtered_text.strip()