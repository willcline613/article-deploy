import pickle
import os
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
import nltk
import string
import re 
from re import search
import spacy

def get_model(data):
    #CATEGORIZER
    #Make imports
    nltk.download('stopwords')
    ps = nltk.PorterStemmer()
    wn = nltk.WordNetLemmatizer()
    snow_stemmer = nltk.SnowballStemmer('english')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    import pickle
    import string
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    print('\n\n', os.getcwd(), '\n\n')
    with open('./src/models/encoder.pkl','rb') as f:
        encoder = pickle.load(f)
    
    #load test sample variable(s)
    string_= data["article"].item()
    
    
    from tqdm import tqdm
    stop = stopwords.words('english')
    
    #Call clean function that is created below this function
    string_ = clean_history(string_, stop, wn)
    
    #Tokenizer
    with open('./src/models/tokenizer.pkl','rb') as f:
        tokenizer = pickle.load(f)
    text_seq = tokenizer.texts_to_sequences([string_])
    text_seq = pad_sequences(text_seq, maxlen=1000, truncating='post')
    
    from tensorflow.keras.models import model_from_yaml
    # load YAML and create model
    yaml_file = open('./src/models/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("./src/models/model.h5")
    print("Loaded model from disk")
    
    #make prediction
    prediction = loaded_model.predict_classes(text_seq)
    #decode
    genre = str(encoder.inverse_transform(prediction))
    #Reformat prediction
    genre = str(genre)[2:-2].lower().title()
    
    
    #troubleshooting
    print("within get_model:")
    print(data["headline"].item())
    
    
    
    #SUMMARIZER
    summarizer_model = un_pickle_model("./src/models/summarizer.pkl")
    ARTICLE = data['article']
    summary_pre = summarizer_model(ARTICLE.item(), max_length=120, min_length=30, do_sample=False)
    #return this processed summary var
    summary = summary_pre[0]['summary_text']
#     summary = "Mockup Summary"
    
    #TROUBLESHOOT AND RETURN
    print(genre)
    print(summary)
    
    return genre, summary

def clean_history(text, stop, wn):
    text = re.sub('<a\b[^>]*>(.*?)</a>', 
       '',text)
    punct_translator=str.maketrans('','',string.punctuation.replace('.', '') + '―“”’')
    digit_translator=str.maketrans('','',string.digits)
    text=text.translate(punct_translator)
    text=text.translate(digit_translator)
    split = text.split()
    text = " ".join([wn.lemmatize(word.lower()) for word in split if word.lower() not in stop and '.com' not in word and 'pictwitter' not in word and 'http' not in word and 'www' not in word])   
    return text

def un_pickle_model(model_path):
    """ Load the model from the .pkl file """
    with open(model_path, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model