#%%Solving the Heat Equation with changing boundary conditions using a 
# simple Explicit Method to Simulating a block of ice floating in seawater

# 1D Heat Equation solver for block of ice floating on seawater using a
# forward time centered space (FTCS) finite difference method

#import needed libbraries
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os #shutil

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

#ND Parameters
diff_time_scale = (float(L**2))/(alpha_ice) #in seconds

# Create two boundary conditions (top and bottom of ice)
T_it = 272.65
T_ib = 273.15

#Create Solution Vector and initial profile
Tsoln = np.zeros(n+1)
Tsoln_pr = np.full(n+1, 272.65)
init = np.full(n+1, 272.65)

# Now we have a initial linear distribution of temperature in the sea ice
plt.plot(x,Tsoln_pr,"g-",label="Initial Profile")
plt.title("Initial Distribution of Temperature in Sea Ice")
plt.close()

# Time parameters
dt = 0.5; # time between iterations, in seconds
nt = 200000; # amount of iterations
t_days = (dt*nt)/86400.0

# Calculate r, want ~0.25, must be < 0.5 (for explicit method)
r = ((alpha_ice)*(dt))/(dx*dx); # stability condition
print("The value of r is ", r)

#Create an empty list for outputs and plots
top_ice_temp_list = []
air_temp_list = []


#%% Start Iteration and prepare plots

#first, clear the folder with the images to make room for new ones
folder = "figures/giffiles"
filelist = [f for f in os.listdir(folder)]
for f in filelist:
    os.remove(os.path.join(folder, f))

for i in range(0,nt):
    # Run through the FTCS with these BC
    for j in range(1,n):
        Tsoln[j] = Tsoln_pr[j] + r*(Tsoln_pr[j+1]-2*Tsoln_pr[j]+Tsoln_pr[j-1])
    
    # time in seconds to hours on a 24-hour clock will be used for radiation function
    
    print(f"i={i}/{nt}, hr={(i*dt/3600)%24:.4f}")
    
    #Now set the top root as the new BC for Tsoln
    Tsoln[0]=air_temp((i+1)*dt)
    
    #Make sure the bottom BC is still 0 degrees C
    Tsoln[-1]=273.15
    
    # Now add the values to their respective lists
    air_temp_list.append(air_temp(i*dt))
    
    # Let's make a movie!
    if (i*dt)%60 == 0: #every 30 seconds
        title = str(int((i*dt)//60))
        plt.close()
        plt.plot(x,Tsoln,"k",label = f"{(i*dt/3600.0)%24:.2f} hours")
        plt.legend(loc=4)
        title1=f"Distribution of Temperature in Sea Ice after {t_days:.2f} days"
        plt.title(title1)
        plt.xlabel("x (m)")
        plt.ylabel("Temperature (K)")
        plt.tight_layout()
        plt.savefig("figures/giffiles/plot"+title+".png")
        plt.close()
    
    #update Tsoln before next time step
    Tsoln_pr = Tsoln

#%% Movie Time

png_dir = 'figures/giffiles/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('figures/icemovie.gif',images)

#%% Plotting Main Results
locs, labels = plt.yticks()
    
# Plot the figure after nt iterations with initial profile
plt.plot(x,init,"g",label="Initial Profile")
plt.plot(x,Tsoln,"k",label=f"After {t_days:.2f} days")
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
