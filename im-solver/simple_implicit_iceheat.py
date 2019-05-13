#%%Solving the Heat Equation with changing boundary conditions using the 
# Crank-Nicolson Method to Simulating a block of ice floating in seawater

#%%import needed libbraries
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os #shutil
from scipy import sparse

#%% Physical Constants in SI Units
alpha = 0.3; # albedo
eps_ice = 0.96; # emissivity
rho_i = 916.7; # density of ice, kg/m^3
T_w = 272.15; # temperature of bulk water, K
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
n = 400; # number of nodes
x = np.linspace(0.0,L,n+1); #space mesh
dx = L/n; # length between nodes

# Time parameters
dt = 0.5; # time between iterations, in seconds
nt = 6000; # amount of iterations
t_days = (dt*nt)/86400.0

r = ((alpha_ice)*(dt))/(dx*dx); # stability condition
print("The value of r is ", r)

#ND Parameters
diff_time_scale = (float(L**2))/(alpha_ice) #in seconds

#%% Set up the scheme

#Create the matrices in the C_N scheme
#these do not change during the iterations

A = sparse.diags([-r, 1+2*r, -r], [-1, 0, 1], shape = (n+1,n+1), format='lil')

#now we pad the matrices with one in the TL and BR corners
A[0,[0,1]] = [1.0,0.0]
A[1,0] = -r
A[n,[n-1,n]] = [0.0,1.0]
A[n-1,n] = -r

A = A.tocsc()

#some inital profile
def T_init(x):
    return -14.0*np.sin(np.pi*x/L) + ((air_temp(0.0)*(L-x))+(T_w*x))/L
u = T_init(x)
#set initial BC as well
u.shape = (len(u),1)
u[0]=air_temp(0.0)
u[-1]=273.15

# Now we have a initial distribution of temperature in the sea ice
#plt.plot(x,Tsoln_pr,"g-",label="Initial Profile")
#plt.title("Initial Distribution of Temperature in Sea Ice")
#plt.close()

#Create an empty list for outputs and plots
top_ice_temp_list = []
air_temp_list = []

#set initial conditions to the matrix as the first row
u_soln = u

#%% Start Iteration and prepare plots

for i in range(0,nt):
    
    print(f"i={i}/{nt}, hr={(i*dt/3600)%24:.4f}")
    
    # Run through the CN scheme for interior points
    u = sparse.linalg.spsolve(A,u)
    
    #force to be column vector
    u.shape = (len(u),1)

    #update u top boundary
    u[0]=air_temp(i*dt)
  
    # Now add the values to their respective lists
    air_temp_list.append(air_temp(i*dt))
    top_ice_temp_list.append(u[0])
    
    #append this array to solution file
    if (i*dt)%120 == 0: #every 60 seconds
        u_soln = np.append(u_soln, u, axis=1)

# write the solution matrix to a file
u_soln = u_soln.transpose()
np.savetxt(f"im_output_{n+1}_nodes.txt",u_soln, fmt = '%.10f',delimiter=' ')

#%% Plotting Main Results
locs, labels = plt.yticks()
    
# Plot the figure after nt iterations with initial profile
plt.plot(x,u,"k",label=f"After {t_days:.2f} days")
title1=f"Distribution of Temperature in Sea Ice after {t_days:.2f} days"
plt.title(title1)
plt.xlabel("x (m)")
#plt.yticks(locs, map(lambda x: "%.1f" % x, locs*1e0))
plt.ylabel("Temperature (K)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/ice_temp_distribution.png")
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
plt.savefig("figures/surface_temp_temporal.png")
plt.close()
