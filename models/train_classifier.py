import sys
import pandas as pd
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import gzip
import bz2

def load_data(database_filepath):
    """
    Load data from SQLite database and split into features and labels.

    Parameters:
    - database_filepath (str): Path to SQLite database file.

    Returns:
    - X (Series): Messages for classification.
    - y (DataFrame): Labels for each category.
    - category_names (Index): Names of categories in y.
    """
    #reloading the tables data into dataframe for further processing
    #remember we have learned about engine and its parameters in process_code.py file
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('response_table', engine)

    #this will be our feature, as the message is the most important column for understanding disaster responses.
    X = df['message']
    #we are using slicing operator with iloc
    #iloc uses integer-location based indexing to select all rows and columns from 4th index till end
    # we can also write specifically as df.iloc[:, 4:40] which will take columns from index 4 till 39
    y = df.iloc[:, 4:]
    #we have been asked to return the column names of our target so entity.columns gives us <class 'pandas.core.indexes.base.Index'>
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    Tokenizes text data by normalizing, lemmatizing, and removing URLs.

    Parameters:
    - text (str): Raw text data.

    Returns:
    - list: List of clean tokens.
    """
    # Replace URLs with a placeholder
    # reason why we are considering url?, because we have practiced this code in our modules and it is good to write the code
    # for revision
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, "urlplaceholder", text)

    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    #this all tokenization can also be done in 1 line like below
    #tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word not in stop_words]
    #but avoid to use as while coding i faced lot of errors when used stop_words not sure the exact reason but there might be an issue with loading some nltk resources whne joblib is getting called.

    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline and GridSearchCV object.

    Returns:
    - cv (GridSearchCV): Grid search object with pipeline.
    """
    #we learned 3 important concepts in ML:
    # Pipeline - which reduces lot of our code
    # FeatureUnion - to run parallel esitmators on similar features.
    # GridSearchCV - to utlize the hyperparameters for enhancing your model
    # we will not use FeatureUnion as the questions is straight forward
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    #size of pickle file generated post model creation was 1GB due to which the github uploade was failing
    #as the limit is of 100MB size per file on github, so took different steps to 
    #have the same efficiency & reduce the size
    # removed estimators 100 which is usually the tree size.
    #A smaller number of trees can reduce the model size without significantly affecting performance
    #reduced the depth size as well
    parameters = {
        'clf__estimator__n_estimators': [None,50],
        'clf__estimator__min_samples_split': [2, 5]
    }

    #cv=5: Specifies that we want to use 5-fold cross-validation.
    #this param helps in breaking the dataset into multiple parts or "folds," and then the model is trained on some folds while being tested on others.
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1, verbose=2)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate model on test data and print classification report.

    Parameters:
    - model: Trained model.
    - X_test (Series): Test messages.
    - y_test (DataFrame): True labels for test data.
    - category_names (Index): Category names.

    Returns:
    - None: Prints classification metrics for each category.
    """
    # classification_report helps in displaying output in our required format
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test.values, y_pred_test, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save trained model as a pickle file.

    Parameters:
    - model: Trained model.
    - model_filepath (str): File path to save the model.

    Returns:
    - None: Saves model to specified filepath.
    """
    #another step taken to reduce the pickle size by 
    #Remove training data references
    #used del to remove large, unnecessary attributes here (cv.best_estimator_._estimators_).
    if hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, '_estimators_'):
        del model.best_estimator_._estimators_

    with  bz2.BZ2File(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database as the first argument and the filepath of the pickle file '
              'to save the model to as the second argument.\n\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
