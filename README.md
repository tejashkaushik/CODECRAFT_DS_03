# CODECRAFT_DS_03

# 📊 Customer Purchase Prediction using Decision Tree

This project uses the `bank-full.csv` dataset to build a **Decision Tree Classifier** that predicts whether a customer will purchase a product or service based on their **demographic and behavioral data**. This work is submitted as part of an internship task under **CodeCraft**.

---

## 📁 Files in this Repository

- `bank-full.csv` — Dataset used for training the model  
- `decision_tree_predict.py` — Python script that:
  - Loads and preprocesses the dataset
  - Trains a Decision Tree model
  - Evaluates its performance
  - Predicts new customer responses from user input  
- `README.md` — This documentation file  
- `requirements.txt` — List of Python libraries needed to run the project  

---

## 🚀 How to Run This Project

### 🔧 Step 1: Install the required libraries

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt


▶️ Step 2: Run the script
bash

python decision_tree_model.py


✍️ Step 3: Enter customer information
The program will ask you to input details like age, job, marital status, etc. It will then display a prediction:

bash

Will the customer purchase the product/service? yes

🔢 Features Used in the Model
The following features were used to train the Decision Tree model:

age

job

marital

education

default

balance

housing

loan

contact

day

month

duration

campaign

pdays

previous

poutcome


🧪 Sample Prediction
Here's a sample interaction:

text

Enter age: 45
Enter job: technician
Enter marital status: married
...
Prediction: yes

📊 Model Info
Algorithm: Decision Tree Classifier

Criterion: Entropy (Information Gain)

Max Depth: 5

Accuracy on Test Data: ~89.5%

Issue Noted: Slight class imbalance — performs better on 'no' class

📝 Notes
The dataset used (bank-full.csv) is from a marketing campaign dataset used by a Portuguese bank.

Categorical variables are label-encoded using sklearn.preprocessing.LabelEncoder.

The prediction logic uses the same encoders to transform user input before passing it to the trained model.

This project was developed as part of a learning task during my internship at CodeCraft.

📦 Requirements
These libraries are needed:

text

pandas
scikit-learn
matplotlib
They are listed in requirements.txt.

🙌 Author
Tejash Kaushik
Intern at CodeCraft
GitHub: [tejashkaushik]


