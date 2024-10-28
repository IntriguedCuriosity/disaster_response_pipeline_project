**Disaster Response Pipeline Project**
- This project processes and classifies disaster response messages, enabling better disaster management by categorizing messages into relevant response types.


**Project Components**

-process_data.py: Extracts, cleans, and transforms data for modeling.

-train_classifier.py: Trains a machine learning model using the cleaned data.

-run.py: A Flask web app that allows users to input messages and view category predictions.


**Installation**
-Clone the repository, and install the necessary packages:
> pip install -r requirements.txt
 

**File Descriptions**

-process_data.py: Loads messages and categories, merges datasets, and cleans the data.

-train_classifier.py: Builds a model pipeline, trains, and tunes the model using GridSearchCV.

-run.py: Launches a web interface for disaster response predictions.



**Instructions**
-Run ETL Pipeline: Process and clean data.
- go to data directory 
>python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

**Train ML Model:**
- you can go to models directory and run below command:
>python train_classifier.py ../data/DisasterResponse.db classifier.pkl


**Run the Web App:**
- go to app directory:
>python run.py


-Access the web app at:

http://0.0.0.0:3000.

**Visualizations**

-The web app provides visual insights into genre and category distributions. Suggestions for further visualizations include:

-**Word Cloud** for frequently appearing words.

-**Heatmap** of category correlations.


**License**
MIT License
