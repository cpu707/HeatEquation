#%%Solving the Heat Equation with changing boundary conditions using the 
# Crank-Nicolson Method to Simulating a block of ice floating in seawater

#%%import needed libbraries
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os #shutil
import matplotlib.animation as manimation
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from scipy import sparse
from scipy import linalg
import stat

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
n = 401; # number of nodes
x = np.linspace(0.0,L,n); #space mesh
dx = L/(n-1); # length between nodes

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

A = sparse.diags([-r, 1+2*r, -r], [-1, 0, 1], shape = (n-2,n-2)).toarray()
B = sparse.diags([r, 1-r, r], [-1, 0, 1], shape = (n-2,n-2)).toarray()
#plt.matshow(A)
#plt.matshow(B)

#some inital profile
u = np.full(n, 272.65)
#set initial BC as well
u[0]=air_temp(0.0)
u[-1]=273.15

#this here is only defined to plot initial profile, not used anywhere else
init = np.full(n,272.65)
init[0]=air_temp(0.0)
init[-1]=273.15
#%% Initial and boudnary conditions

# Now we have a initial linear distribution of temperature in the sea ice
plt.plot(x,u,"g-",label="Initial Profile")
plt.title("Initial Distribution of Temperature in Sea Ice")
plt.savefig("init_profile.png")
plt.close()

#initially solve right hand side of matrix equation
rhs = B.dot(u[1:-1])

#Create an empty list for outputs and plots
top_ice_temp_list = []
air_temp_list = []

#%% Start Iteration and prepare plots

folder = "giffiles"
os.chmod(folder, 0o777)
filelist = [f for f in os.listdir(folder)]
for f in filelist:
    os.remove(os.path.join(folder, f))

for i in range(0,nt):
    
    # time in seconds to hours on a 24-hour clock will be used for air temp function
    print(f"i={i}/{nt}, %={(i/nt)*100:.3f}, hr={(i*dt/3600)%24:.4f}")

    # Run through the CN scheme for interior points
    u[1:-1] = sparse.linalg.spsolve(A,rhs)

    #update u top boundary
    u[0]=air_temp(i*dt)
    
    #Make sure the bottom BC is still 0 degrees C
    u[-1]=273.15
    
    #update rhs with new interior nodes
    rhs = B.dot(u[1:-1])
    
    # Now add the surface temp to its list
    top_ice_temp_list.append(u[0])
    
    # Let's make a movie!
    if (i*dt)%120 == 0: #every 60 seconds
        title = str(int((i*dt)//60))
        plt.close()
        plt.plot(x,init,"g-",label="Initial Profile")
        plt.plot(x,u,"k",label = f"{(i*dt/3600.0)%24:.2f} hours")
        plt.legend(loc=4)
        title1=f"Distribution of Temperature in Sea Ice after {t_days:.2f} days"
        plt.title(title1)
        plt.xlabel("x (m)")
        plt.ylabel("Temperature (K)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("giffiles/plot"+title+".png")
        plt.close()
    

#%% Movie Time

png_dir = 'giffiles/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('icemovie.gif',images)

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

#finally, clear the folder with the images to make room for new ones

folder = "giffiles"
os.chmod(folder, 0o777)
filelist = [f for f in os.listdir(folder)]
for f in filelist:
    os.remove(os.path.join(folder, f))