#%%Solving the Heat Equation with changing boundary conditions using a 
# simple Explicit Method to Simulating a block of ice floating in seawater

# 1D Heat Equation solver for block of ice floating on seawater using a
# forward time centered space (FTCS) finite difference method

#import needed libbraries
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#%% Constants in SI Units
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

#%% Initial and Boundary Conditions

# Space mesh
L = 2.0; # depth of sea ice
n = 400; # number of nodes
dx = L/n; # length between nodes
x = np.linspace(0.0,L,n+1);

# Time parameters
<<<<<<< HEAD
dt = 20; # time between iterations, in seconds
nt = 20000; # amount of iterations
=======
dt = 0.5; # time between iterations, in seconds
nt = 1000; # amount of iterations
>>>>>>> c4ff7d3953c8e77b99f485fc52d3ca5fcd68909a
t_days = (dt*nt)/86400.0

# Calculate r, want ~0.25, must be < 0.5 (for explicit method)
r = ((alpha_ice)*(dt))/(dx*dx);# stability condition
print("The value of r is ", r)

#ND Parameters
diff_time_scale = (float(L**2))/(alpha_ice) #in seconds

#%% More numerical parameters

#some inital profile
def T_init(x):
    return -14.0*np.sin(np.pi*x/L) + ((air_temp(0.0)*(L-x))+(T_w*x))/L

u = T_init(x)
#set initial BC as well
u.shape = (len(u),1)
u[0]=air_temp(0.0)
u[-1]=273.15

#set initial conditions to the matrix as the first row
u_soln = u

# Now we have a initial linear distribution of temperature in the sea ice
plt.plot(x,u,"g-",label="Initial Profile")
plt.title("Initial Distribution of Temperature in Sea Ice")
plt.savefig("init_profile.png")
plt.close()

#Create an empty list for outputs and plots
top_ice_temp_list = []
air_temp_list = []


#%% Start Iteration and prepare plots

for i in range(0,nt):
    
    # time in seconds to hours on a 24-hour clock will be used for air temp function
    print(f"i={i}/{nt}, %={(i/nt)*100:.3f}, hr={(i*dt/3600)%24:.4f}")
    
    # Run through the FTCS with these BC
#    Tsoln[1:n] = Tsoln[1:n]+r*(Tsoln_pr[2:n+1]-2*Tsoln_pr[1:n]+Tsoln_pr[0:n-1])   
    for j in range(1,n):
        u[j] = u[j] + r*(u[j+1]-2*u[j]+u[j-1])
    
    #force to be column vector
    u.shape = (len(u),1)

    #update u top boundary
    u[0]=air_temp(i*dt)
    
    # Now add the surface temp to its list
    top_ice_temp_list.append(float(u[0]))
    
    #append this array to solution file
    if (i*dt)%120 == 0: #every 60 seconds
        u_soln = np.append(u_soln, u, axis=1)
    
# write the solution matrix to a file
u_soln = u_soln.transpose()
np.savetxt(f"ex_output_{n+1}_nodes.txt",u_soln, fmt = '%.10f',delimiter=' ')

#print(u[40])

#%% Plotting Main Results
locs, labels = plt.yticks()
    
# Plot the figure after nt iterations with initial profile
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
plt.ylabel('Temperature (K)')
plt.legend()
plt.savefig("surface_temp_temporal.png")
plt.close()