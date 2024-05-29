from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import joblib
from analysis.result_analysis import analyse_error

######################################################
CSV_INHALE_PATH = 'data/inhale.csv'
CSV_EXHALE_PATH = 'data/exhale.csv'
CSV_SILENCE_PATH = 'data/silence.csv'
MODEL_PATH = 'model/trained_model_rf.pkl'
######################################################

# load training data
X_inhale = pd.read_csv(CSV_INHALE_PATH)
X_exhale = pd.read_csv(CSV_EXHALE_PATH)
X_silence = pd.read_csv(CSV_SILENCE_PATH)

# combining training data into one DataFrame
X = pd.concat([X_inhale, X_exhale, X_silence], ignore_index=True)

# labeling training data
Y = [0] * len(X_inhale) + [1] * len(X_exhale) + [2] * len(X_silence)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# creating and training classifier with regularization
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)

# saving the model to a file
joblib.dump(rf_classifier, MODEL_PATH)

# cross-validation
scores = cross_val_score(rf_classifier, X_train, Y_train, cv=3, scoring="accuracy")
print("Cross-validation accuracy:", scores)

# accuracy of train
start = time.time()
predictions_train = rf_classifier.predict(X_train)
end = time.time()

print("Time:", end - start)
print("Predictions on training set:", predictions_train.tolist())
print("True labels for training set:", Y_train)

# calculate the accuracy of the model on the training set
train_accuracy = accuracy_score(Y_train, predictions_train)
print("Train accuracy:", train_accuracy)


# accuracy of test
start = time.time()
predictions = rf_classifier.predict(X_test)
end = time.time()

print("Time:", end - start)
print("Predictions:", predictions.tolist())
print("True labels:", Y_test)

# calculate the accuracy of the model on the test set
accuracy = accuracy_score(Y_test, predictions)
print("Test accuracy:", accuracy)

# analyse results
analyse_error(rf_classifier, X_test, Y_test)

"""

# Hyperparameter tuning (checking which hyperparameters perform best)
param_grid = {
    'n_estimators': [10, 30, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'ccp_alpha': [0.0, 0.01, 0.1]
}

grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Best parameters:  {'ccp_alpha': 0.0, 'max_depth': None, 'n_estimators': 100}

"""
