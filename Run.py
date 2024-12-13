import os
import emoji
import joblib
import pandas as pd
import re, nltk
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Firstly, Preprocessing
def preprocessing(df, context='text'):
    clean_tokens_list = []
    for text in df[context].fillna('').astype(str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = emoji.replace_emoji(text, replace='')
        lower_text = text.lower()
        tokenize_text = word_tokenize(lower_text)
        clean_tokens = []
        stop_words = set(stopwords.words('english'))
        for token in tokenize_text:
            if token not in stop_words and token.replace(' ', '') != '' and len(token) > 1:
                clean_tokens.append(token)
        clean_tokens_list.append(clean_tokens)
    return clean_tokens_list

# Lemmatization and Stemming
def stem_and_lem(tokens_list):
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    result = []
    for tokens in tokens_list:
        processed_tokens = [lemmatizer.lemmatize(ps.stem(token)) for token in tokens]
        result.append(processed_tokens)
    return result

# Embedding
def tfidf_test(text_vector, vectorizer):
    untokenized_data = [' '.join(text) for text in tqdm(text_vector, "Vectorizing...")]
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors

# Loading vectorizer
vectorizer_path = "./models/tfidf_vectorizer.pkl"
vectorizer = joblib.load(vectorizer_path)
print("Loaded TF-IDF vectorizer successfully.")

# Loading Text data and Preprocessing
df = pd.read_csv(r"./offensive-test.csv")
clean_tokens = preprocessing(df, context='text')
preprocess_tokens = stem_and_lem(clean_tokens)
text_label = df['label'].astype(int)

# Converting test data to TF-IDF features
X_test_tfidf = tfidf_test(preprocess_tokens, vectorizer)

# Model Path
models_dir = "./models"
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and f != "tfidf_vectorizer.pkl"]

# Test every model
for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    model_name = os.path.splitext(model_file)[0]  #Extract the model name (minus the .pkl extension)
    print(f"Evaluating model: {model_name}")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(text_label, y_pred)
    conf_matrix = confusion_matrix(text_label, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 50)
