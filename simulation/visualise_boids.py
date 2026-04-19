import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')  # or 'Qt5Agg'

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from boid_config import X_BOUND, Y_BOUND
from make_boid_history import make_history

make_history()

fig, ax = plt.subplots()
scat = ax.scatter([], [])
ax.set_xlim(-X_BOUND, X_BOUND)
ax.set_ylim(-Y_BOUND, Y_BOUND)

df = pd.read_csv('data/boid_history.csv')
grouped = df.groupby('time')

def update(frame):
    data = grouped.get_group(frame)
    x = data["x"]
    y = data["y"]

    scat.set_offsets(np.c_[x, y])
    return scat,

ani = FuncAnimation(
    fig,
    update,
    frames=sorted(df["time"].unique())[::10],
    interval=20,
    blit=True
)

plt.show()