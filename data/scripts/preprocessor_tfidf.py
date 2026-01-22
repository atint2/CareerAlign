import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
#nltk.download('stopwords')
#nltk.download('punkt_tab')

class TFIDFPreprocessor:
    """Data Preprocessor for job descriptions and resumes -- TF-IDF model"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Remove domain-specific words from stopwords
        self.stop_words -= {'senior', 'junior', 'lead', 'principal', 'staff', 
                           'full', 'part', 'remote', 'hybrid', 'not', 'no'}
        self.stemmer = PorterStemmer()
        
        # Technical terms to protect
        self.technical_patterns = {
            r'\bC\+\+\b': 'cpluspluslang',
            r'\bC#\b': 'csharplang',
            r'\b\.NET\b': 'dotnetframework',
            r'\bNode\.js\b': 'nodejsframework',
            r'\bReact\.js\b': 'reactjsframework',
            r'\bVue\.js\b': 'vuejsframework',
        }
    
    def _protect_technical_terms(self, text):
        """Replace technical terms with placeholders"""
        protected = {}
        for pattern, placeholder in self.technical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in protected.values():
                    text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
                    protected[placeholder] = match
        return text, protected
    
    def _restore_technical_terms(self, text, protected):
        """Restore original technical terms"""
        for placeholder, original in protected.items():
            text = text.replace(placeholder, original.lower())
        return text
    
    def clean_text_tfidf(self, text):
        """Clean text for TF-IDF model with comprehensive preprocessing"""
        if not text or not isinstance(text, str):
            return ""
        
        # Protect technical terms first
        text, protected = self._protect_technical_terms(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', "", text)
        
        # Remove phone numbers
        text = re.sub(r'\+?\d[\d\s\-\(\)]{8,}\d', "", text)
        
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens (but preserve protected terms)
        tokens = [
            word for word in tokens 
            if (word not in self.stop_words and len(word) > 2) 
            or any(placeholder in word for placeholder in protected.keys())
        ]
        
        # Stem tokens (except protected terms)
        stemmed_tokens = []
        for token in tokens:
            if any(placeholder in token for placeholder in protected.keys()):
                stemmed_tokens.append(token)
            else:
                stemmed_tokens.append(self.stemmer.stem(token))
        
        text = " ".join(stemmed_tokens)
        
        # Restore technical terms
        text = self._restore_technical_terms(text, protected)
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text.strip()