from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import time
import result_analysis
import joblib

######################################################
CSV_INHALE_TRAIN_PATH = 'data/train/csv/inhale.csv'
CSV_EXHALE_TRAIN_PATH = 'data/train/csv/exhale.csv'
CSV_SILENCE_TRAIN_PATH = 'data/train/csv/silence.csv'
CSV_INHALE_TEST_PATH = 'data/test/csv/inhale.csv'
CSV_EXHALE_TEST_PATH = 'data/test/csv/exhale.csv'
CSV_SILENCE_TEST_PATH = 'data/test/csv/silence.csv'
MODEL_PATH = 'model/trained_model_rf.pkl'
######################################################

# load training data
X_train_inhale = pd.read_csv(CSV_INHALE_TRAIN_PATH)
X_train_exhale = pd.read_csv(CSV_EXHALE_TRAIN_PATH)
X_train_silence = pd.read_csv(CSV_SILENCE_TRAIN_PATH)

# combining training data into one DataFrame
X_train = pd.concat([X_train_inhale, X_train_exhale, X_train_silence], ignore_index=True)

# labeling training data
Y_train = [0] * len(X_train_inhale) + [1] * len(X_train_exhale) + [2] * len(X_train_silence)

X_test_exhale = pd.read_csv(CSV_EXHALE_TEST_PATH)
X_test_inhale = pd.read_csv(CSV_INHALE_TEST_PATH)
X_test_silence = pd.read_csv(CSV_SILENCE_TEST_PATH)
X_test = pd.concat([X_test_inhale, X_test_exhale, X_test_silence], ignore_index=True)

Y_test = [0] * len(X_test_inhale) + [1] * len(X_test_exhale) + [2] * len(X_test_silence)

# creating and training classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, Y_train)

# cross-validation
scores = cross_val_score(rf_classifier, X_train, Y_train, cv=3, scoring="accuracy")
print("Cross-validation accuracy:", scores)

# error analysis
# result_analysis.analyse_error(rf_classifier, X_train, y_train)

# TODO : check accuracy on test data
# prediction
start = time.time()
predictions = rf_classifier.predict(X_test)
print("Time", time.time() - start)
print(predictions.tolist())
print(Y_test)

X_train = pd.concat([X_train_inhale, X_train_exhale, X_train_silence, X_test_inhale, X_test_exhale, X_test_silence], ignore_index=True)
Y_train = [0] * len(X_train_inhale) + [1] * len(X_train_exhale) + [2] * len(X_train_silence) + [0] * len(X_test_inhale) + [1] * len(X_test_exhale) + [2] * len(X_test_silence)
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, Y_train)
joblib.dump(rf_classifier, MODEL_PATH)