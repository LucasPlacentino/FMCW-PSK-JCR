"""
ELEC-H311
Lucas Placentino
Salman Houdaibi
R R [possiblement, probablement, éventuellement, il se pourrait, il y a une possibilité, surtout qu'Anthony est parti (pas pour de vrai, il a quitté l'option t'as capté chacal) en vrai y a grave moyen, sauf s'il fait avec Louis]
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
#import math
#import pandas as pd
import numpy as np
import scipy as sp

#t = np.arange(0, 5, 0.01)
# Chirp's end time
T_chirp_duration = 1e-4  # 0.1 ms - 0.4 ms
#T = 0.1 # 100ms
Number_of_samples = 2**18 # 2**18 # 262144 samples
t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)
#t = np.linspace(0, T, Number_of_samples, endpoint=True)

B_freq_range = 200e6 # 200 MHz
f_c = 24*10e9 # Carrier frequency 24 GHz
F_sampling_freq = 512e6  # 512 MHz
# chirp duration T = 0.1 - 0.4 ms

Beta_slope = B_freq_range/T_chirp_duration # B = Beta*T

def f_i(t):
    return Beta_slope*t

def phi_i(t):
    return np.pi*Beta_slope*(t**2)

def s_with_carrier(t):
    return np.cos(2*np.pi*f_c*t + phi_i(t))

def s(t):
    return np.cos(phi_i(t))

#f_i = [sp.signal.sawtooth(2 * np.pi * 5 * t) for t in t]
freq = f_i(t)
plt.plot(t, freq)
plt.grid()
plt.legend(['$f_i$'])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Instantaneous frequency')
#go.Figure(data=go.Scatter(x=t, y=f_i(t))).show()

#plt.plot(t, B*sp.signal.sawtooth(2 * np.pi * t))

# math.cos(2*math.pi*t)
#plt.plot(t, np.cos(2 * np.pi * 5 * t))

plt.show() # display plot

# Signal plot in time domain:
plt.plot(t, s(t))
plt.grid()
plt.legend(['$s(t)$'])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('FM transmitted signal (radar chirp in time domain - real part)')
#go.Figure(data=go.Scatter(x=t, y=s(t))).show()

plt.show() # display plot

# FFT of signal => signal (FT) plot in frequency domain:
plt.plot(freq, np.fft.fft(s(t))) #? s(t) or s_with_carrier(t)
plt.grid()
plt.legend(['$S(f)$'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Fourier transform of the signal (radar chirp in frequency domain - modulus)')
#go.Figure(data=go.Scatter(x=t, y=np.fft.fft(s(t)))).show()

plt.show() # display plot

#f = np.fft.fftshift(f)

# Bandwidth of the signal:
B = B_freq_range # ?
# or 15 MHz ? found
print('Bandwidth of the signal: B =', B, 'Hz')

