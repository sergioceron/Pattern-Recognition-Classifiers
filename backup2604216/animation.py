"""
A simple example of an animated plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    print i
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,


ani = animation.FuncAnimation(fig, animate, np.arange(1, 100),
                              interval=25, blit=False, repeat=False)
plt.show()
