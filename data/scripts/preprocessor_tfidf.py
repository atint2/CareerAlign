import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import json

# Download required NLTK data (run once)
#nltk.download('stopwords')
#nltk.download('punkt_tab')

class TFIDFPreprocessor:
    """Data Preprocessor for job descriptions and resumes -- TF-IDF model"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words -= {'senior', 'junior', 'lead', 'principal', 'staff', 
                           'full', 'part', 'remote', 'hybrid'}
        self.stemmer = PorterStemmer()

    def stem_word(self, word):
        special_characters = ['.', '#', '++']

        if (len(word) <= 3):
            return word
        if any(char in special_characters for char in word):
            return word
        return self.stemmer.stem(word)

    def clean_text_tfidf(self, text):
        """Clean text for TF-IDF model with comprehensive preprocessing"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert text to lowercase
        text = text.lower()

        # Protect C++ and C#
        text = re.sub(r'c\+\+', 'CPLUSPLUS', text)
        text = re.sub(r'c#', 'CSHARP', text)

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
        
        # Split by 'slash' to handle terms like 'Python/Django'
        words = [subword for word in words for subword in word.split('/')]
        filtered_words = [word for word in words if word not in self.stop_words]
        filtered_text = " ".join(filtered_words)

        # Remove digits
        filtered_text = re.sub(r'\d+', '', filtered_text)
        
        # Remove special characters and punctuation
        # Preserve words that either start with periods or contain periods (except if just at the end)
        filtered_text = re.sub(r'[^a-zA-Z0-9\.\s]', ' ', filtered_text)
        filtered_text = re.sub(r'(?<!\w)\.(?!\w)', ' ', filtered_text)  # Remove standalone periods
        # Remove periods at end of words
        filtered_text = re.sub(r'(?<=\w)\.(?!\w)', ' ', filtered_text)

        # Restore C++ and C#
        filtered_text = re.sub(r'CPLUSPLUS', 'C++', filtered_text)
        filtered_text = re.sub(r'CSHARP', 'C#', filtered_text)

        # Stemming
        stemmed_words = [self.stem_word(word) for word in filtered_text.split()]
        filtered_text = " ".join(stemmed_words)

        # Remove extra whitespace
        filtered_text = " ".join(filtered_text.split())

        return filtered_text.strip()