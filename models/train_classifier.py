import sys
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, punkt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_data', engine)
    X = df['message']
    y = df.iloc[:,4:40]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    stop_words=set(stopwords.words('english'))
    lemmatizer= WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word not in stop_words]

    return tokens

def build_model():
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfid', TfidfTransformer()),
    ('multi', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    parameters = {
        'vect__ngram_range': ((1, 1),(1,2)),
        'multi__estimator__n_estimators':[10,20],
        'multi__estimator__min_samples_split': [2,3]
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred_test = model.predict(X_test)
    for idx, col in enumerate(Y_test.columns):
        print(f"Metrics for {col}:")
        print(classification_report(Y_test[col], Y_pred_test[:, idx]))
    

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()