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

#solverlist = ["ex", "im", "cn"]
#solver_method = solverlist[0]
#loaded_matrix = np.loadtxt(f'{solver_method}-solver/{solver_method}_output_' \
#                           + f'{n+1}_nodes.txt', dtype='f', delimiter=' ')
#
#x = np.linspace(0.0, 2.0, len(loaded_matrix[1]))
#
#with writer.saving(fig, f"{solver_method}_{len(loaded_matrix[1])}_node_solution.mp4", 100):    
#    for i in range(len(loaded_matrix)):
#        y = loaded_matrix[i]
#        plt.plot(x,y)
#        plt.title(f"Time Evolution of Heat Equation Solver")
#        writer.grab_frame()
#        plt.clf()

#explicit case
loaded_matrix_ex = np.loadtxt(f'ex-solver/ex_output_{n+1}_nodes.txt', dtype='f', delimiter=' ')
#since this is our first matrix, we must set x values
x = np.linspace(0.0, 2.0, len(loaded_matrix_ex[0]))
with writer.saving(fig, f"ex_{len(loaded_matrix_ex[1])}_node_solution.mp4", 100):    
    for i in range(len(loaded_matrix_ex)):
        y1 = loaded_matrix_ex[i]
        plt.plot(x,y1,'r')
        plt.title(f"Time Evolution of Heat Equation Solver - EX")
        writer.grab_frame()
        plt.clf()


loaded_matrix_im = np.loadtxt(f'im-solver/im_output_{n+1}_nodes.txt', dtype='f', delimiter=' ')
with writer.saving(fig, f"im_{len(loaded_matrix_im[1])}_node_solution.mp4", 100):    
    for i in range(len(loaded_matrix_im)):
        y2 = loaded_matrix_im[i]
        plt.plot(x,y2,'b')
        plt.title(f"Time Evolution of Heat Equation Solver - IM")
        writer.grab_frame()
        plt.clf()



loaded_matrix_cn = np.loadtxt(f'cn-solver/cn_output_{n+1}_nodes.txt', dtype='f', delimiter=' ')
with writer.saving(fig, f"cn_{len(loaded_matrix_cn[1])}_node_solution.mp4", 100):    
    for i in range(len(loaded_matrix_cn)):
        y3 = loaded_matrix_cn[i]
        plt.plot(x,y3,'g')
        plt.title(f"Time Evolution of Heat Equation Solver - CN")
        writer.grab_frame()
        plt.clf()
        

#%% For all three methods at once
with writer.saving(fig, f"all_solutions_at_once.mp4", 100):    
    for i in range(len(loaded_matrix_ex)):
        y1 = loaded_matrix_ex[i]
        y2 = loaded_matrix_im[i]
        y3 = loaded_matrix_cn[i]
        plt.plot(x,y1,"r", label = "EX")
        plt.plot(x,y2,"b--", label ="IM")
        plt.plot(x,y3,"g",label="CN")
        plt.title(f"Time Evolution of Heat Equation Solver - All Three Methods")
        plt.legend()
        writer.grab_frame()
        plt.clf()
















