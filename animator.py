"""
Goal: Take output from different schemes and create plots and make a movie
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-o')

n = 400

solverlist = ["ex", "im", "cn"]
solver_method = solverlist[1]
loaded_matrix = np.loadtxt(f'{solver_method}-solver/{solver_method}_output_' \
                           + f'{n+1}_nodes.txt', dtype='f', delimiter=' ')

x = np.linspace(0.0, 2.0, len(loaded_matrix[1]))

with writer.saving(fig, f"{solver_method}_{len(loaded_matrix[1])}_node_solution.mp4", 100):    
    for i in range(len(loaded_matrix)):
        y = loaded_matrix[i]
        plt.plot(x,y)
        plt.title(f"Time Evolution of Heat Equation Solver")
        writer.grab_frame()
        plt.clf()
        



