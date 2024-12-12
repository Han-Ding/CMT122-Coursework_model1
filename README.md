# Task 1:
Each group is given a specific dataset. Each group's task is developing a whole machine
learning pipeline that tries to solve the task based on what you have studied in class. You
will have to implement two solutions:
1. Based on a machine learning algorithm such as logistic regression, naive bayes
model, LSTM, RNN and use embeddings, etc.
2. Using language models. E.g.,
- Fine-tune a pretrained-language model.
- Use a large language model and in-context learning, for example by
employing zero-shot, one-shot and/or few shot.

# Five Machine Learning Models

| **Model**                 | **Advantages**                                             | **Disadvantages**                               |
|---------------------------|------------------------------------------------------------|-------------------------------------------------|
| **Multinomial Naive Bayes (MNB)**       | small computational effort, suitable for large-scale data. MNB can make good use of conditional probability for classification in the case of insufficient data.| Assumption that features are independent of each other, which usually does not hold true in real data. |
| **K-Nearest Neighbors (KNN)**           | Direct use of similarities between data points without complex assumptions. | As the amount of data increases, the overhead of computing nearest neighbors during prediction increases. |
| **Decision Tree (DT)**                  | Can fit complex decision boundaries well. | Prone to overfitting training data in deep trees or when data dimensionality is high. |
| **Random Forest (RF)**                  | Reduced dependence on individual feature segmentation by integrating multiple trees. | Compared to a single decision tree, the decision-making process of a random forest is difficult to explain intuitively. |
| **Logistic Regression (LR)**            | Often combined with TF-IDF features, it performs well in text categorization tasks. | Assuming a linear relationship between features and categories makes it difficult to capture complex patterns. |

# Train Models Approach

---

## Step 1:  Import the libraries that we need using "import".

- **emoji**: Used to delete emoji in texts.
- **joblib**: Used to save models that could be used directly.
- **re**: Delete information we don't want to use.
- **MultinomialNB** : For training MNB model
- **KNeighborsClassifier** : For training polynomial plain Bayesian models.
- **DecisionTreeClassifier** : For training decision tree models.
- **RandomForestClassifier** : For training random forest models.
- **LogisticRegression** : For training logistic regression models.
- **WordNetLemmatizer, PorterStemmer** : Reducing a word to its lemma and truncating the endings of words to extract the stem, without concern for semantics or lexical properties.
- **GridSearchCV** : The best combination of hyperparameters for the model is found through an exhaustive search. Perform cross-validation for each parameter combination and return the model with the best performance.

---

## Step 2:  Preprocessing

### Step 2.1: Removing noises:
This function cleans the text by:
1. Remove URLs, HTML tags, hashtags, extra spaces, and emoticons.
2. Lowercasing the text.
3. Tokenizing sentences and deleting extra space.
4. Deleting stopwords.

```
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

```

### Step 2.2: Stemming and morphological reduction:
• Stemming and morphological reduction is performed for each word in order to standardise the form of the word and thus reduce the diversity of the vocabulary.
**For Example : ‘running’ → stem extraction → ‘run’ → morphological reduction → ‘run’**

```
def stem_and_lem(tokens_list):
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    result = []
    for tokens in tokens_list:
        processed_tokens = [lemmatizer.lemmatize(ps.stem(token)) for token in tokens]
        result.append(processed_tokens)
    return result
```

### Step 2.3: Embedding:

• TF-IDF Feature Extraction and Vectorisation of Segmented Text Data.

```
#Embedding
def tfidf(text_vector):
    vectorizer = TfidfVectorizer()
    untokenized_data = [' '.join(text) for text in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors, vectorizer
```

## Step 3:  Loading data and saving tf-idf model:

**Load the training data and save the vectorised model, which can be called directly.**

```
df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 2\Coursework2_Model2\offensive-test.csv')
clean_tokens = preprocessing(df, context = 'text')
preprocess_tokens = stem_and_lem(clean_tokens)
text_label = df['label'].astype(int)
X_train_tfidf, vectorizer = tfidf(preprocess_tokens)

vectorizer_path = "tfidf_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_path)
print(f"TF-IDF vectorizer saved as {vectorizer_path}")
```

## Step 4:  Training Models:

- **polynomial plain Bayesian models**
```
lassifier = MultinomialNB(alpha=0.7)
```
**Note ：alpha is a smoothing parameter, the larger the value of the parameter, the smoother the model is and the more sensitive it is to low-frequency words.**

- **Decision tree models**
```
params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
```
**Note ：Specify the range of K values to be considered and have all neighbours contribute equally to the prediction, with closer neighbours having higher weights. Use 3-fold cross-validation. Parallel processing, using 4 threads to accelerate computation.**

- **Random forest models**
 ``` 
classifier = DecisionTreeClassifier(max_depth=800, min_samples_split=5)
        params = {'criterion': ['gini', 'entropy']}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
```
**Note :**
- **Limit the maximum depth of the tree to 800 to prevent overfitting. Each branch node needs to contain at least 5 samples to split further.**
- **gini : gini coefficient, used to measure the purity of nodes.**
- **entropy : information gain, used to measure the reduction of information after splitting.**

- **Random forest models**
```
classifier = RandomForestClassifier(max_depth=800, min_samples_split=5)
        params = {'n_estimators': [50, 100, 150], 'criterion': ['gini', 'entropy']}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
```
**Note : The number of trees in a random forest, ranging from 50 to 150.**

- **Logistic regression models**
```
classifier = LogisticRegression(multi_class='auto', solver='newton-cg')
        params = {"C": np.logspace(-3, 3, 7), "penalty": ["l2"]}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
```
**Note :**
- **auto : automatic selection of binary or multiclassification strategy.**
- **newton-cg : solver for optimisation, suitable for multiclassification tasks and regularisation.**
- **penalty : use L2 regularisation (ridge regression) to apply a squared penalty to the weights.**

 ## Step 5:  Testing data and Results:
 
- **Preprocessing test data : Same treatment of data as for training.**

- **Calling the model and producing results**

```
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
```






