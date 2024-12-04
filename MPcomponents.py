# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:36:21 2024
@author: Ben Fabry
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.optimize import minimize
from matplotlib.patches import Patch



E = 10  #total respiratory elastance in units of mbar / L
R = 8 # total ariway resistance, including tube resistance, in units of mbar / L / s
tau = R/E # time constant of passive expiration
VD = 0.2 # anatomical dead space in units of L
PEEP = 0 # externally applied PEEP in units of mbar
Vmin = 10 # minute ventilation in units of L/min

# Objective function to minimize MP with respect to IEr
def objective(IEr, VT, rr, tau, E, R, PEEP):
    tin = 60 / rr * IEr / (1 + IEr)  # inspiratory time
    tex = 60 / rr - tin  # expiratory time
    EELV = PEEP / E + VT * np.exp(- tex / tau) / (1 - np.exp(- tex / tau))  # end-expiratory lung volume above FRC
    iPEEP = EELV * E  # intrinsic PEEP
    MP = 0.098 * rr * (VT * VT * (0.5 * E + rr * R * (1 + IEr) / (60 * IEr)) + iPEEP * VT)
    return MP

#----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=14)
plt.rc('axes', titlesize=18)


Vminrange = [8]
VTrange = np.arange(VD + 0.001, 1.6, 0.0005)
    
# Plotting
fig1 = plt.figure(1, (7, 5))
border_width = 0.2
ax_size = [0 + border_width, 0 + border_width,
           1 - 2 * border_width, 1 - 2 * border_width]
ax1 = fig1.add_axes(ax_size)

VTbest = []
MPbest = []
rrbest = []
    
MPrange = []
IEr_range = []
rr_range = []
MP_res = []
MP_el = []
MP_PEEP = []
for VT in VTrange:
    rr = Vmin / (VT - VD)  # compute respiratory rate
    
    # Minimize MP with respect to IEr, starting with IEr=1
    result = minimize(objective, x0=1, args=(VT, rr, tau, E, R, PEEP), bounds=[(0.01, 10)], method='L-BFGS-B')
    IEr = result.x[0]  # Optimal I:E ratio

    rr_range.append(rr)
    MP = result.fun
    MPrange.append(MP)
    MP_el.append(0.098 * rr * VT * VT * 0.5 * E) 
    MP_res.append(0.098 * rr * VT * VT * rr * R * (1 + IEr) / (60 * IEr))
    tin = 60 / rr * IEr / (1 + IEr)  # inspiratory time
    tex = 60 / rr - tin  # expiratory time
    EELV = PEEP / E + VT * np.exp(- tex / tau) / (1 - np.exp(- tex / tau))  # end-expiratory lung volume above FRC
    iPEEP = EELV * E  # intrinsic PEEP
    MP_PEEP.append(0.098 * rr * iPEEP * VT)

# pick global MP minimum
MPrange = np.asarray(MPrange)
min_index = np.argmin(MPrange)
VTbest.append(VTrange[min_index])
MPbest.append(MPrange[min_index])
rrbest.append(rr_range[min_index])

# pick  MP_el minimum
MP_el = np.asarray(MP_el)
min_index = np.argmin(MP_el)
VTbest_el = VTrange[min_index]
MPbest_el = MP_el[min_index]

MP_res = np.asarray(MP_res)
MP_PEEP = np.asarray(MP_PEEP)

ax1.fill_between(VTrange, 0, MP_el, color='orange', alpha=0.3, edgecolor='gray', zorder=0)  
ax1.fill_between(VTrange, MP_el, MP_res+MP_el, color='blue', alpha=0.3, edgecolor='gray', zorder=0)  
ax1.fill_between(VTrange, MP_el+MP_res, MP_PEEP+MP_res+MP_el, color='green', alpha=0.3, edgecolor='gray', zorder=0)  

ax1.plot(VTrange,MPrange,'-', color='darkgray', linewidth=2)
ax1.plot([VD,VD],[0,23],'--', color='black', linewidth=1)

# Add the legend for the patches
legend_elements = [
    Patch(facecolor='green', alpha=0.3, edgecolor='gray', label=r'$MP_{iPEEP}$'),
    Patch(facecolor='blue', alpha=0.3, edgecolor='gray', label=r'$MP_{resistive}$'),
    Patch(facecolor='orange', alpha=0.3, edgecolor='gray', label=r'$MP_{elastic}$')

]
ax1.legend(handles=legend_elements, loc='upper right')    

legend = ax1.legend(
    handles=legend_elements, 
    loc='upper right', 
    frameon=False, 
    framealpha=1, 
    edgecolor='gray', 
    handleheight=1,   # Adjust the height of the handle to make it square
    handlelength=1,   # Adjust the length to match the height (square)
)

ax1.plot(VTbest, MPbest, 'o-', color='gray', markerfacecolor='red', markersize=8, linewidth=2)
ax1.plot(VTbest_el, MPbest_el, 'o-', color='gray', markerfacecolor='yellow', markersize=9, linewidth=2)

plt.ylim([0, 23])
plt.xlim([-0.15, np.max(VTrange) + 0.15])
ax1.set_xticks([0, 0.5, 1.0, 1.5])

plt.grid(color=(0.4, 0.4, 0.4), linestyle='-', linewidth=0.2)
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(axis="x", direction="in")
plt.xlabel( r'$V_{\mathrm{T}}$' + ' (L)')
plt.ylabel('MP (J/min)')
plt.show()
filename = f'FigComp_E{E}_R{R}.png'
plt.savefig(filename, dpi = 600, format='png')    

    