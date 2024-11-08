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
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

**Train ML Model:**
- from the home directory of the repository you can run the below command:
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl.bz2

-- You would see that the classifier is with bz2 extension because github does not allow files with size > 100mb and our tuned model with efficient grid search was always crossing 100mb so, in the code it self we are zipping and unzipping the classifer.



**Run the Web App:**
- go to app directory (cd app) and run below:
>python run.py


-Access the web app at:

http://0.0.0.0:3001.
- if there is an error that the port is not available on your local machine, you could choose any other like 5000/5001/6000

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




- at the top you will find the input space, where you can put any prompt that you would feel someone would send as a message. Could be information, request, a panic alert, weather alert, and our classifier would predict what kind of message was it based on which we can identify the next steps, below is the sample image:

- <img width="880" alt="image" src="https://github.com/user-attachments/assets/24aa390c-8a35-4d37-9472-34af9823c866">
  ![Uploading image.png…]()




- One of the major concerns that i would like to point here is the imbalanced dataset, in this project we have some labels (e.g., "water") with far fewer examples than others, it poses challenges for training the model effectively. Lets try to understand the drawbacks and how can we overcome it:
-  The model tends to perform well on majority classes (labels with more samples) but struggles with minority classes. This happens because the model “learns” to prioritize predictions for classes it sees more often, leading to lower accuracy for underrepresented labels. Tuning the hyperparameters would be very helpful to handle these, for example using scoring param with f1_weighted:
--  cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_weighted', cv=5)
   --And we need to consider optimizing the F1 score, which balances precision and recall, especially in categories where both false positives and false negatives have adverse effects.


- You can play with the code of train_classifier.py to play with several other params, that would help us in creating a robust model.


**License**
MIT License



