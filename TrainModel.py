import emoji
import joblib
import pandas as pd
import re, nltk
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

#Preprocessing
def preprocessing(df, context='text'):

    #Remove the noises
    clean_tokens_list = []
    for text in df[context].fillna('').astype(str):
        #Remove noises (URLs, HTML tags, hashtags, etc.)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = emoji.replace_emoji(text, replace='')

        #Transfer to lower and Tokenize
        lower_text = text.lower()
        tokenize_text = word_tokenize(lower_text)

        #Remove stop words
        clean_tokens = []
        stop_words = set(stopwords.words('english'))
        for token in tokenize_text:
            if token not in stop_words and token.replace(' ', '') != '' and len(token) > 1:
                clean_tokens.append(token)
        
        clean_tokens_list.append(clean_tokens)
    
    return clean_tokens_list
    
#Lemmatization and Stemming
def stem_and_lem(tokens_list):
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    result = []
    for tokens in tokens_list:
        processed_tokens = [lemmatizer.lemmatize(ps.stem(token)) for token in tokens]
        result.append(processed_tokens)
    return result

#Embedding
def tfidf(text_vector):
    vectorizer = TfidfVectorizer()
    untokenized_data = [' '.join(text) for text in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors, vectorizer


df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 2\Coursework2_Model2\offensive-test.csv')
clean_tokens = preprocessing(df, context = 'text')
preprocess_tokens = stem_and_lem(clean_tokens)
text_label = df['label'].astype(int)
X_train_tfidf, vectorizer = tfidf(preprocess_tokens)

vectorizer_path = "tfidf_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_path)
print(f"TF-IDF vectorizer saved as {vectorizer_path}")

def train(model_type):
    classifier = None
    if model_type == "MNB":
        classifier = MultinomialNB(alpha=0.7)
        classifier.fit(X_train_tfidf, text_label)
    elif model_type == "KNN":
        classifier = KNeighborsClassifier(n_jobs=4)
        params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
        classifier.fit(X_train_tfidf, text_label)
        classifier = classifier.best_estimator_
    elif model_type == "DT":
        classifier = DecisionTreeClassifier(max_depth=800, min_samples_split=5)
        params = {'criterion': ['gini', 'entropy']}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
        classifier.fit(X_train_tfidf, text_label)
        classifier = classifier.best_estimator_
    elif model_type == "RF":
        classifier = RandomForestClassifier(max_depth=800, min_samples_split=5)
        params = {'n_estimators': [50, 100, 150], 'criterion': ['gini', 'entropy']}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
        classifier.fit(X_train_tfidf, text_label)
        classifier = classifier.best_estimator_
    elif model_type == "LR":
        classifier = LogisticRegression(multi_class='auto', solver='newton-cg')
        params = {"C": np.logspace(-3, 3, 7), "penalty": ["l2"]}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
        classifier.fit(X_train_tfidf, text_label)
        classifier = classifier.best_estimator_
    else:
        print("Unsupported Model Type!")
        return None
    
    print(f"Model: {model_type}")
    train_accuracy = accuracy_score(text_label, classifier.predict(X_train_tfidf))
    print(f"Training Accuracy: {train_accuracy:.4f}")

    #Saveing models
    model_path = f"{model_type}_model.pkl"
    joblib.dump(classifier, model_path)
    print(f"Model saved as {model_path}")

for model in ["MNB", "KNN", "DT", "RF", "LR"]:
    train(model)









