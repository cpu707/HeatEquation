#%%Solving the Heat Equation with changing boundary conditions using the 
# Crank-Nicolson Method to Simulating a block of ice floating in seawater

#%%import needed libraries
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from scipy import sparse
from scipy.sparse import linalg


#%% Physical Constants in SI Units
alpha = 0.3; # albedo
eps_ice = 0.96; # emissivity
rho_i = 916.7; # density of ice, kg/m^3
T_w = 273.15; # temperature of bulk water, K
c_pi = 2027; # specific heat of ice, J/(kgK)
kappa_ice = 2.25; # thermal conductivity of ice, W/(mK)
alpha_ice = (kappa_ice)/(c_pi*rho_i); # thermal diffusivity of ice, m^2/s

# Define function to calculate temperature based on time
def air_temp(t): #t is in seconds, so dt*i would be evaluated
    t_hours = (t/3600.0)%24 #convert to 24 hour clock
    temp = 7.0*np.sin((np.pi/12.0)*(t_hours-13.0))+268
    return temp
# the initial value for temperature would be air_temp(0.0)

#%% Numerical Parameters

# Space mesh
L = 2.0; # depth of sea ice
n = 400; # number of spaces
x = np.linspace(0.0,L,n+1); #space mesh
dx = L/n; # length between nodes

# Time parameters
dt = 0.5; # time between iterations, in seconds
nt = 6000; # amount of iterations
t_days = (dt*nt)/86400.0

r = ((alpha_ice)*(dt))/(2*dx*dx); # stability condition
print("The value of r is ", r)

#ND Parameters
diff_time_scale = (float(L**2))/(alpha_ice) #in seconds

#%% Set up the scheme

#Create the matrices in the C_N scheme
#these do not change during the iterations

A = sparse.diags([-r, 1+2*r, -r], [-1, 0, 1], shape = (n+1,n+1),format='lil')
B = sparse.diags([r, 1-2*r, r], [-1, 0, 1], shape = (n+1,n+1),format='lil')

#now we pad the matrices with one in the TL and BR corners
A[0,[0,1]] = [1.0,0.0]
A[n,[n-1,n]] = [0.0,1.0]
B[0,[0,1]] = [1.0,0.0]
B[n,[n-1,n]] = [0.0,1.0]

#now convert todifferent format

A = A.tocsc()
B = B.tocsc()

#some inital profile
def T_init(x):
    return -14.0*np.sin(np.pi*x/L) + ((air_temp(0.0)*(L-x))+(T_w*x))/L

u = T_init(x)
#set initial BC as well
u.shape = (len(u),1)
u[0]=air_temp(0.0)
u[-1]=273.15


#%% Initial and boudnary conditions

# Now we have a initial linear distribution of temperature in the sea ice
plt.plot(x,u.transpose()[0],"g-",label="Initial Profile")
plt.title("Initial Distribution of Temperature in Sea Ice")
plt.savefig("init_profile.png")
plt.close()

#initially solve right hand side of matrix equation
rhs = B.dot(u)

#Create an empty list for outputs and plots
top_ice_temp_list = []
air_temp_list = []

#set initial conditions to the matrix as the first row
u_soln = u

#%% Start Iteration and prepare plots

for i in range(0,nt):
    
    # time in seconds to hours on a 24-hour clock will be used for air temp function
    print(f"i={i}/{nt}, %={(i/nt)*100:.3f}, hr={(i*dt/3600)%24:.4f}")

    # Run through the CN scheme for interior points
    u = sparse.linalg.spsolve(A,rhs)
    
    #force to be column vector
    u.shape = (len(u),1)

    #update u top boundary
    u[0]=air_temp(i*dt)
    
    #Make sure the bottom BC is still 0 degrees C
    u[-1]=273.15
    
    #update rhs with new interior nodes
    rhs = B.dot(u)
    
    # Now add the surface temp to its list
    top_ice_temp_list.append(u[0])
    
    #append this array to solution file
    if (i*dt)%120 == 0: #every 60 seconds
        u_soln = np.append(u_soln, u, axis=1)
    

#%% write the solution matrix to a file
np.savetxt(f"cn_output_{n+1}_nodes.txt",u_soln.transpose(), fmt = '%.10f',delimiter=' ')

#%% Plotting Main Results
locs, labels = plt.yticks()
    
# Plot the figure after nt iterations with initial profile
plt.plot(x,u,"g",label="Initial Profile")
plt.plot(x,u,"k",label=f"After {t_days:.2f} days")
title1=f"Distribution of Temperature in Sea Ice after {t_days:.2f} days"
plt.title(title1)
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.legend()
plt.tight_layout()
plt.savefig("ice_temp_distribution.png")
plt.close()

#%% Some more output

locs, labels = plt.yticks()

#Create Time Array
time_list = dt*(np.array(list(range(1,nt+1)))) #in seconds, can convert later
time_hours = time_list/3600.0

#Plot time evolution of surface temperature
title_T_it=f"Surface Temperature Evolution after {t_days:.2f} days"
plt.plot(time_hours,top_ice_temp_list,label="Top of Ice Surface Temperature")
plt.title(title_T_it)
plt.xlabel("Time (hr)")
#plt.yticks(locs, map(lambda x: "%.3f" % x, locs*1e0))
plt.ylabel('Temperature (K)')
plt.legend()
plt.tight_layout()
plt.savefig("surface_temp_temporal.png")
plt.close()