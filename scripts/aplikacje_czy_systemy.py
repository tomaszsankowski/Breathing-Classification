import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Tworzenie figur i osi
fig, ax = plt.subplots()
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.5, 1.5)

# Linia sinusoidy
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x), color='black')

print("x:", len(x))

# Kolorowanie obszaru pod krzywą
fill_red = ax.fill_between(x, 0, np.sin(x), where=(np.sin(x) >= 0), color='red', alpha=0.3)
fill_green = ax.fill_between(x, 0, np.sin(x), where=(np.sin(x) <= 0), color='green', alpha=0.3)

# Funkcja animacji
def update(frame):
    global fill_red, fill_green
    line.set_ydata(np.sin(x + frame/10))
    fill_red.remove()
    fill_green.remove()
    fill_red = ax.fill_between(x, -6000, 6000, where=(np.sin(x + frame/10) >= 0), color='red', alpha=0.3)
    fill_green = ax.fill_between(x, -6000, 6000, where=(np.sin(x + frame/10) <= 0), color='green', alpha=0.3)
    return line,

# Tworzenie animacji
ani = FuncAnimation(fig, update, frames=range(200), interval=50)

plt.show()
