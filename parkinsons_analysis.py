# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def display_symptoms():
    print("List of Symptoms:")
    print("1. Tremors")
    print("2. Rigidity")
    print("3. Bradykinesia")
    print("4. Postural Instability")
    print("5. Sleep disturbances")
    print("6. Mood disorders (Depression, Anxiety)")
    print("7. Cognitive Impairment")

def predict_symptoms():
    # Take input from the user
    age = float(input("Enter age: "))
    blood_pressure = float(input("Enter blood pressure: "))
    cholesterol_levels = float(input("Enter cholesterol levels: "))
    exercise_hours_per_week = float(input("Enter exercise hours per week: "))

    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'age': [age],
        'blood_pressure': [blood_pressure],
        'cholesterol_levels': [cholesterol_levels],
        'exercise_hours_per_week': [exercise_hours_per_week]
    })

    # Standardize the user input
    user_data_scaled = scaler.transform(user_data)

    # Make predictions
    prediction = rf_classifier.predict(user_data_scaled)

    # Print the prediction
    if prediction[0] == 1:
        print("The model predicts that the individual may have Parkinson's Disease.")
    else:
        print("The model predicts that the individual may not have Parkinson's Disease.")

# Function to display the menu
def display_menu():
    print("\nPlease select an option:")
    print("1. View the list of symptoms")
    print("2. Enter symptoms for prediction")
    print("3. Add a new person's data")
    print("4. Delete a person's data")
    print("5. Edit a person's data")
    print("6. Fetch all data")
    print("7. Exit")

# Function to add a new person's data
def add_person_data(data):
    new_data = {}
    for column in data.columns[:-1]:
        value = input(f"Enter {column}: ")
        new_data[column] = [float(value)]
    new_data['target'] = [int(input("Enter target (1 or 0): "))]
    new_person_data = pd.DataFrame(new_data)
    return pd.concat([data, new_person_data], ignore_index=True)

# Function to delete a person's data
def delete_person_data(data):
    display_data(data)
    index_to_delete = int(input("Enter the index to delete: "))
    return data.drop(index=index_to_delete).reset_index(drop=True)

# Function to edit a person's data
def edit_person_data(data):
    display_data(data)
    index_to_edit = int(input("Enter the index to edit: "))
    for column in data.columns[:-1]:
        value = input(f"Enter new {column} (or press Enter to keep the existing value '{data.at[index_to_edit, column]}'): ")
        if value:
            data.at[index_to_edit, column] = float(value)
    target_value = input(f"Enter new target value (or press Enter to keep the existing value '{data.at[index_to_edit, 'target']}'): ")
    if target_value:
        data.at[index_to_edit, 'target'] = int(target_value)
    return data


# Function to fetch all data with descriptive target labels
def fetch_all_data(data):
    # Make a copy so the original DataFrame isn't changed
    display_df = data.copy()
    
    # Map target values to descriptive labels
    display_df['target'] = display_df['target'].map({
        0: "The person has Parkinson’s Disease",
        1: "The person does NOT have Parkinson’s Disease"
    })
    
    display_data(display_df)

# Function to display data
def display_data(data):
    print("\nCurrent Data:")
    print(data)
# Load the dataset from "parkinsons_data.csv"
data = pd.read_csv("parkinsons_data.csv")

# Data preprocessing
X = data.drop("target", axis=1)  # Features
y = data["target"]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Display menu
while True:
    display_menu()
    choice = input("Enter your choice (1-7): ")

    if choice == '1':
        display_symptoms()
    elif choice == '2':
        predict_symptoms()
    elif choice == '3':
        data = add_person_data(data)
    elif choice == '4':
        data = delete_person_data(data)
    elif choice == '5':
        data = edit_person_data(data)
    elif choice == '6':
        fetch_all_data(data)
    elif choice == '7':
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please enter a number between 1 and 7.")
