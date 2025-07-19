import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt


# Loading dataset
df = pd.read_csv("bank-full.csv", sep=';')
df.head()


# Made a copy of the dataset
df_clean = df.copy()

# Encoding of all categorical columns
label_encoders = {}
for col in df_clean.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Features and target
X = df_clean.drop('y', axis=1)
y = df_clean['y']

# Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Creation and training of the model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
clf.fit(X_train, y_train)


# Make predictions
y_pred = clf.predict(X_test)

# Check accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Visualize the tree
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, feature_names=X.columns, class_names=label_encoders['y'].classes_, filled=True)
plt.show()

def get_customer_input():
    return {
        'age': int(input("Enter age: ")),
        'job': input("Enter job (e.g., technician, management): "),
        'marital': input("Enter marital status (single, married, divorced): "),
        'education': input("Enter education (primary, secondary, tertiary, unknown): "),
        'default': input("Has credit in default? (yes/no): "),
        'balance': int(input("Enter balance: ")),
        'housing': input("Has housing loan? (yes/no): "),
        'loan': input("Has personal loan? (yes/no): "),
        'contact': input("Contact type (cellular, telephone, unknown): "),
        'day': int(input("Day of last contact: ")),
        'month': input("Month of last contact (jan, feb, ..., dec): "),
        'duration': int(input("Duration of contact in seconds: ")),
        'campaign': int(input("Number of contacts in this campaign: ")),
        'pdays': int(input("Days since last contact (-1 if never): ")),
        'previous': int(input("Number of previous contacts: ")),
        'poutcome': input("Outcome of previous campaign (success, failure, unknown): ")
    }
def predict_new_customer(input_dict, model, encoders):
    # Convert to DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Encode categorical columns
    for col in df_input.columns:
        if col in encoders:
            df_input[col] = encoders[col].transform(df_input[col])
    
    # Predict
    pred = model.predict(df_input)
    result = encoders['y'].inverse_transform(pred)
    
    return result[0]

while True:
    new_customer = get_customer_input()
    result = predict_new_customer(new_customer, clf, label_encoders)
    print("Prediction:", result)
    
    again = input("Predict for another customer? (yes/no): ")
    if again.lower() != 'yes':
        break


#Predict
result = predict_new_customer(new_customer, clf, label_encoders)
print("Prediction:", result)

