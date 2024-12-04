Simple script to compute the ventilator parameters (VT, rr, I:E ratio) that minimize mechanical power for a given alveolar minute ventilation. There is no user interface. Users should specify the following parameters directly in the script:

``` python
E = 10  # total respiratory elastance in units of mbar / L
R = 8 # total airway resistance, including tube resistance, in units of mbar / L / s
VD = 0.2 # anatomical dead space in units of L
PEEP = 0 # externally applied PEEP in units of mbar
```

The script computes the ventilator parameters (VT, rr, I:E ratio) that minimize mechanical power for a range of alveolar minute ventilation; this range can also be edited directly in the script: 

```python
Vminrange = [5,6,7,8,9,10,11,12,13,14,15] # range of minute ventilation to be explored
```
The range of tidal volumes that are tested for their effect of mechanical power can also be specified, if needed:
```python
VTrange = np.arange(VD + 0.05, 1.6, 0.0005) # range of tidal volumes to be explored
```
