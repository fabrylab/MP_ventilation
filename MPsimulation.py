# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:36:21 2024
@author: Ben Fabry
To cite this software in publications, please cite:
Ben Fabry https://github.com/fabrylab/MP_ventilation
You can find more information in the following publication:
B Fabry, "How to minimize mechanical power during controlled mechanical ventilation"
Int Care Med Exp 2024, DOI 10.1186/s40635-024-00699-4 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.optimize import minimize


E = 10  #total respiratory elastance in units of mbar / L
R = 8 # total airway resistance, including tube resistance, in units of mbar / L / s
VD = 0.2 # anatomical dead space in units of L
PEEP = 0 # externally applied PEEP in units of mbar

tau = R/E # time constant of passive expiration


# Objective function to minimize MP with respect to I:E ratio
def objective(IEr, VT, rr, tau, E, R, PEEP):
    tin = 60 / rr * IEr / (1 + IEr)  # inspiratory time
    tex = 60 / rr - tin  # expiratory time
    EELV = PEEP / E + VT * np.exp(- tex / tau) / (1 - np.exp(- tex / tau))  # end-expiratory lung volume above FRC
    iPEEP = EELV * E  # intrinsic PEEP
    MP = 0.1 * rr * (VT * VT * (0.5 * E + rr * R * (1 + IEr) / (60 * IEr)) + iPEEP * VT)
    return MP

#----------general fonts and font sizes for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('axes', titlesize=18)


Vminrange = [5,6,7,8,9,10,11,12,13,14,15] # range of minute ventilation to be explored
VTrange = np.arange(VD + 0.05, 1.6, 0.0005) # range of tidal volumes to be explored
    
# Plotting
fig1 = plt.figure(1, (7, 5))
border_width = 0.2
ax_size = [0 + border_width, 0 + border_width,
           1 - 2 * border_width, 1 - 2 * border_width]
ax1 = fig1.add_axes(ax_size)

VTbest = []
MPbest = []
rrbest = []
    
for Vmin in Vminrange:
    print(f'computing power for a minute ventilation of {Vmin:.1f} L/min')
    MPrange = []
    IEr_range = []
    rr_range = []
    for VT in VTrange:
        rr = Vmin / (VT - VD)  # compute respiratory rate
        
        # Minimize MP with respect to I:E ratio, starting with I:E=1
        result = minimize(objective, x0=1, args=(VT, rr, tau, E, R, PEEP), bounds=[(0.01, 10)], method='L-BFGS-B')
        optimal_IEr = result.x[0]  # Optimal I:E ratio

        IEr_range.append(optimal_IEr)
        rr_range.append(rr)
        MP = result.fun
        MPrange.append(MP)
        
    MPrange = np.asarray(MPrange)
    IEr_range = np.asarray(IEr_range)
    min_index = np.argmin(MPrange)
    VTbest.append(VTrange[min_index])
    MPbest.append(MPrange[min_index])
    rrbest.append(rr_range[min_index])
    
    points = np.array([VTrange, MPrange]).T.reshape(-1, 1, 2) # Create segments for LineCollection
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
    lc = LineCollection(segments, cmap='viridis', norm=mcolors.LogNorm(vmin=0.7, vmax=4)) # Create a LineCollection from the segments
    lc.set_array(IEr_range)
    lc.set_linewidth(2)
    
    # Add the LineCollection to the plot
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.text(np.max(VTrange) + 0.14 - 0.2/(Vmin-3), MPrange[-1], Vmin, fontsize=10, ha='right', va='center') # display the V'min
    #ax1.text(VTrange[min_index]-0.1, MPrange[min_index], int(rr_range[min_index]), fontsize=10, ha='right') # display the rr at the minimum MP

    
ax1.text(np.max(VTrange) + 0.12, MPrange[-1] + 1.7, r'$\dot{V}_{\mathrm{alv}}$' + ' (L/min)', fontsize=10, ha='right', bbox=dict(facecolor='white', edgecolor='none', pad=0))

cbar = plt.colorbar(lc, shrink=0.6, orientation='vertical', pad=0.1) # Add a colorbar to show the I:E ratio
cbar.ax.set_title('I:E', fontsize=18, pad=10)  

cbar.ax.tick_params(labelsize=18)
cbar.set_ticks([0.7, 1.0, 1.5, 2, 3, 4])
cbar.set_ticklabels(['0.7', '1.0', '1.5', '2', '3', '4']) 

ax1.plot(VTbest, MPbest, 'o-', color='gray', markerfacecolor='red', markersize=6, linewidth=2)
plt.ylim([0, 43])
plt.xlim([-0.15, np.max(VTrange) + 0.15])
ax1.set_xticks([0, 0.5, 1.0, 1.5])

plt.grid(color=(0.4, 0.4, 0.4), linestyle='-', linewidth=0.2)
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(axis="x", direction="in")
plt.xlabel( r'$V_{\mathrm{T}}$' + ' (L)')
plt.ylabel('MP (J/min)')
plt.show()
#filename = f'Fig2_E{E}_R{R}.png'
#plt.savefig(filename, dpi = 600, format='png')    

    
