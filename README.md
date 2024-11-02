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

-process_data.py: this file reads two CSV files, merges them, cleans the data, and then saves it to an SQLite database. you can get more details about every function by leveraging the docstrings.

-train_classifier.py: this file loads data, tokenizes, builds a model, trains it, evaluates it, and then saves the trained model. We have utilized RandomForestClassifier as our classifier with detailed docstrings.

-run.py: Launches a web interface for disaster response predictions.



**Instructions**
-Run ETL Pipeline: Process and clean data.
- from the home directory of your code package run below.
>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

**Train ML Model:**
- from the home directory of the repository you can run the below command:
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


**Run the Web App:**
- go to app directory (cd app) and run below:
>python run.py


-Access the web app at:

http://0.0.0.0:3001.

**Visualizations**

-The web app provides visual insights into genre and category distributions. Suggestions for further visualizations include:

-**Distribution of Message Genres**.
<img width="834" alt="image" src="https://github.com/user-attachments/assets/979f6d43-e937-412d-ab1a-772185222b30">


-**Distribution of Message Categories**.
<img width="814" alt="image" src="https://github.com/user-attachments/assets/ee2a307c-5954-4697-a011-96e8be08811c">


-**Top 5 categories of Genres**
<img width="849" alt="image" src="https://github.com/user-attachments/assets/d670e786-12c5-4140-ab70-bf860ce9dc92">


-**Pie chart of Percentage of Messages Per Category**
<img width="795" alt="image" src="https://github.com/user-attachments/assets/ad17bae1-d4bc-4ccc-8c70-797b693cac57">


![Uploading image.png…]()



**License**
MIT License
