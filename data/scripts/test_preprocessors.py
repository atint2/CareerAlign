import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessor_sbert import SBERTPreprocessor
from preprocessor_tfidf import TFIDFPreprocessor

def test_tfidf_preprocessor():
    """Test TF-IDF preprocessor with various inputs"""
    print("=" * 60)
    print("Testing TF-IDF Preprocessor")
    print("=" * 60)
    
    tfidf_prep = TFIDFPreprocessor()
    
    test_cases = [
        "Senior C++ Developer with 5+ years experience",
        "Looking for Python/Django expert. Email: hr@test.com",
        "React.js and Node.js development",
        "Ph.D. in Computer Science preferred",
        "Full-stack engineer specializing in .NET framework",
        "",  # Empty string
        None,  # None value
    ]
    
    for i, test in enumerate(test_cases, 1):
        try:
            result = tfidf_prep.clean_text_tfidf(test)
            print(f"\nTest Case {i}:")
            print(f"Input:  {repr(test)}")
            print(f"Output: {repr(result)}")
        except Exception as e:
            print(f"\nTest Case {i}:")
            print(f"Input:  {repr(test)}")
            print(f"ERROR: {e}")


def test_sbert_preprocessor():
    """Test SBERT preprocessor with various inputs"""
    print("\n" + "=" * 60)
    print("Testing SBERT Preprocessor")
    print("=" * 60)
    
    test_cases = [
        "Senior C++ Developer with 5+ years experience",
        "Looking for Python/Django expert. Email: hr@test.com",
        "React.js and Node.js development",
        "Visit https://company.com/careers for more info",
        "Call us at +1-555-123-4567",
        "",  # Empty string
        None,  # None value
    ]
    
    for i, test in enumerate(test_cases, 1):
        try:
            result = SBERTPreprocessor.clean_text_sbert(test)
            print(f"\nTest Case {i}:")
            print(f"Input:  {repr(test)}")
            print(f"Output: {repr(result)}")
        except Exception as e:
            print(f"\nTest Case {i}:")
            print(f"Input:  {repr(test)}")
            print(f"ERROR: {e}")


def test_comparison():
    """Compare TF-IDF vs SBERT preprocessing on same input"""
    print("\n" + "=" * 60)
    print("Comparison: TF-IDF vs SBERT")
    print("=" * 60)
    
    tfidf_prep = TFIDFPreprocessor()
    
    sample_text = """
    We are seeking a Senior Software Engineer with 5+ years of experience 
    in C++ and Python development. Must have expertise in Node.js and React.js.
    Contact: jobs@company.com | Visit: https://company.com/careers
    """
    
    print("\nOriginal Text:")
    print(sample_text)
    
    print("\n" + "-" * 60)
    print("TF-IDF Preprocessed:")
    print(tfidf_prep.clean_text_tfidf(sample_text))
    
    print("\n" + "-" * 60)
    print("SBERT Preprocessed:")
    print(SBERTPreprocessor.clean_text_sbert(sample_text))


if __name__ == "__main__":
    # Run all tests
    test_tfidf_preprocessor()
    test_sbert_preprocessor()
    test_comparison()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)