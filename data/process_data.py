import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Parameters:
    - messages_filepath (str): File path for the messages CSV.
    - categories_filepath (str): File path for the categories CSV.

    Returns:
    - DataFrame: Merged DataFrame of messages and categories.
    """
    #converting all messages into dataframe from csv file
    messages = pd.read_csv(messages_filepath)
    #loading all categories into dataframe from csv file
    categories = pd.read_csv(categories_filepath)
    #function merges two DataFrames based on a key or multiple keys.
    df = pd.merge(messages, categories, how='inner', left_on='id', right_on='id')

    return df

def clean_data(df):
    """
    Clean merged DataFrame by splitting categories and removing duplicates.

    Parameters:
    - df (DataFrame): Merged DataFrame containing messages and categories.

    Returns:
    - DataFrame: Cleaned DataFrame with separate category columns.
    """
    # Split categories into separate columns
    # converting each column value into a string and then spliting based on -, which returns a Series.
    categories = df['categories'].str.split(';', expand=True)

    # Extract category names from the first row of data
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Convert category values to 0 or 1, we will handle non 1/0 values later for one column
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # Drop original categories column and merge new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates, we can either drop it from the main directly or use another variable to refer a new df with
    # dropped duplicate columns liek new_df=df.drop_duplicates()
    df.drop_duplicates(inplace=True)

    # Replace '2' with '1' in 'related' column due to data inconsistency
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    return df

def save_data(df, database_filename):
    """
    Save cleaned data to SQLite database.
>>>>>>> origin/master

    Parameters:
    - df (DataFrame): Cleaned data.
    - database_filename (str): Filename for the SQLite database.

    Returns:
    - None: Saves the DataFrame into an SQLite database.
    """
    #utilzing the code provided in jupyter notebook, adding important parameters:
    # engine is refering to the param con=engine, it is basically SQLAlchemy engine that connects to the database
    # index=False, argument prevents the df's index from being written as a column in the SQL table
    # if_exists='replace', parameter specifies what to do if a table with the same name already exists.
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('response_table', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the file paths for the messages and categories datasets as the first and second arguments, respectively,'
              ' as well as the file path for the database to save the cleaned data as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv DisasterResponse.db')

if __name__ == '__main__':
    main()
