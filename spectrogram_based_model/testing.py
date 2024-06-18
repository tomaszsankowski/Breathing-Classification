import matplotlib.pyplot as plt
import numpy as np

# Twoje dane
x = np.arange(0, 100, 1)
y = np.sin(x / 10)

# Twoje klasyfikacje (przykładowe)
classification = np.random.choice([0, 1, 2], size=len(x))

# Tworzenie wykresu
fig, ax = plt.subplots()

# Rysowanie linii
line, = ax.plot(x, y, color='black')

# Wypełnianie obszarów różnymi kolorami w zależności od klasyfikacji
ax.fill_between(x, y, where=(classification == 0), color='red', alpha=0.3)
ax.fill_between(x, y, where=(classification == 1), color='green', alpha=0.3)
ax.fill_between(x, y, where=(classification == 2), color='blue', alpha=0.3)

plt.show()