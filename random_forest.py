from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import time
import result_analysis

######################################################
CSV_EXHALE_TRAIN_PATH = 'data/train/exhale/exhale.csv'
CSV_INHALE_TRAIN_PATH = 'data/train/inhale/inhale.csv'
CSV_INHALE_TEST_PATH = 'data/test/exhale/exhale.csv'
CSV_EXHALE_TEST_PATH = 'data/test/inhale/inhale.csv'
######################################################

# Wczytanie danych treningowych
X_train_inhale = pd.read_csv(CSV_INHALE_TRAIN_PATH)
X_train_exhale = pd.read_csv(CSV_EXHALE_TRAIN_PATH)

# Połączenie danych treningowych w jeden DataFrame
X_train = pd.concat([X_train_inhale, X_train_exhale], ignore_index=True)

# Tworzenie etykiet dla danych treningowych
y_train = [0] * len(X_train_inhale) + [1] * len(X_train_exhale)

X_test_ex = pd.read_csv(CSV_EXHALE_TEST_PATH)
X_test_in = pd.read_csv(CSV_INHALE_TEST_PATH)
X_test = pd.concat([X_test_ex, X_test_in], ignore_index=True)
# Tworzenie i trenowanie klasyfikatora
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

#Kroswalidacja
scores = cross_val_score(rf_classifier, X_train, y_train, cv=3, scoring="accuracy")
print("Cross-validation accuracy:", scores)

#Analiza błedow
result_analysis.analyse_error(rf_classifier, X_train, y_train)

# Predykcja
start = time.time()
predictions = rf_classifier.predict(X_test)
print("Time", time.time() - start)
print(predictions)
