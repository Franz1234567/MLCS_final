import pandas as pd

# Define the data
models = [
    {"model": "KNN", "time_taken": 0.0204007625579834, "f1_score": 0.9354518371400199, "accuracy": 0.9272116461366181},
    {"model": "Naive Bayes", "time_taken": 0.02700972557067871, "f1_score": 0.9257028112449799, "accuracy": 0.9171332586786114},
    {"model": "Logistic Regression", "time_taken": 0.11501097679138184, "f1_score": 0.9595959595959596, "accuracy": 0.9552071668533034},
    {"model": "Decision Tree", "time_taken": 0.07430100440979004, "f1_score": 0.9498997995991983, "accuracy": 0.9440089585666294},
    {"model": "Random Forest", "time_taken": 0.4822201728820801, "f1_score": 0.9604863221884499, "accuracy": 0.9563269876819709},
    {"model": "SVM", "time_taken": 1.0566859245300293, "f1_score": 0.966497461928934, "accuracy": 0.9630459126539753},
    {"model": "Gradient Boosting", "time_taken": 3.4946980476379395, "f1_score": 0.9626639757820383, "accuracy": 0.9585666293393057}
]

# Create a DataFrame
df = pd.DataFrame(models)

# Round all numerical values to 4 decimal places
df = df.round({'time_taken': 4, 'f1_score': 4, 'accuracy': 4})

# Sort by f1_score in descending order
df = df.sort_values(by='f1_score', ascending=True)

# Rearrange columns so f1_score is the last column
df = df[['model', 'time_taken', 'accuracy', 'f1_score']]

# Print the DataFrame
print(df.to_string(index=False))
