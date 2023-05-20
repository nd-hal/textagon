import os
import nltk

def main():
    os.system("python -m spacy download en_core_web_sm")

    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('sentiwordnet')
    nltk.download('vader_lexicon')
    nltk.download('punkt')
