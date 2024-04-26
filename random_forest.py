from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time

# Wczytanie danych treningowych
X_train_inhale = pd.read_csv('inhale.csv')
X_train_exhale = pd.read_csv('exhale.csv')

# Połączenie danych treningowych w jeden DataFrame
X_train = pd.concat([X_train_inhale, X_train_exhale], ignore_index=True)

# Tworzenie etykiet dla danych treningowych
y_train = [0] * len(X_train_inhale) + [1] * len(X_train_exhale)

X_test_ex = pd.read_csv('exhale_test.csv')
X_test_in = pd.read_csv('inhale_test.csv')
X_test = pd.concat([X_test_ex, X_test_in], ignore_index=True)
# Tworzenie i trenowanie klasyfikatora
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Predykcja
start = time.time()
predictions = rf_classifier.predict(X_test)
print("Time", time.time() - start)
print(predictions)
