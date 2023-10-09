"""
ELEC-H311
Lucas Placentino
Salman Houdaibi
R R
"""

import matplotlib.pyplot as plt
#import math
#import pandas as pd
import numpy as np
import scipy as sp

#t = np.arange(0, 5, 0.01)
T = 5
t = np.linspace(0, 5, T*1000, endpoint=True)

B_freq_range = 200e6 # 200 MHz
f_c = 24*10e9 # Carrier frequency 24 GHz
F_sampling_freq = 512e6  # 512 MHz
Num_samples = 2**18
# chirp duration T = 0.1 - 0.4 ms
T_chirp_duration = 2e-4 # 0.2 ms
f_c = 10 # ???
Beta_slope = B_freq_range/T # B = Beta*T

def f_i(t):
    return Beta_slope*t

def phi_i(t):
    return np.pi*Beta_slope*t**2

def s(t):
    return np.cos(2*np.pi*f_c*t + phi_i(t))

#f_i = [sp.signal.sawtooth(2 * np.pi * 5 * t) for t in t]
#plt.plot(t, f_i)
plt.plot(t, B*sp.signal.sawtooth(2 * np.pi * t))

# math.cos(2*math.pi*t)
plt.plot(t, np.cos(2 * np.pi * 5 * t))
