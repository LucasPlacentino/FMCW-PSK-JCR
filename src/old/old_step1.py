"""
ELEC-H311 - Joint Communications and Radar simulation project
Lucas Placentino
Salman Houdaibié
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
#import math
#import pandas as pd
import numpy as np
import scipy as sp

"""
    Step 1: FMCW signal
"""

#t = np.arange(0, 5, 0.01)
T_chirp_duration = 2e-4  # chirp duration: 0.1 ms to 0.4 ms
#T = 2*T_chirp_duration # window duration (we first see the transmitting of the message than nothing)
Number_of_samples = 2**18 # 2**18 # 262144 samples
t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)
t_extended = np.linspace(0, 2*T_chirp_duration, 2*Number_of_samples, endpoint=True)

B_freq_range = 200e6 # 200 MHz
f_c = 24*10e9 # Carrier frequency 24 GHz
F_sampling_freq = 512e6  # 512 MHz

Beta_slope = B_freq_range/T_chirp_duration # B = Beta*T

def f_i(t):
    #return Beta_slope*(t%T_chirp_duration)
    #return Beta_slope*(t*(1-int(t/T_chirp_duration))) # does not work
    return Beta_slope*t

def phi_i(t):
    return np.pi*Beta_slope*(t**2) # ? should integrate rather

def s_with_carrier(t):
    return np.cos(2*np.pi*f_c*t + phi_i(t))

def s_baseband(t):
    return np.exp(1j*phi_i(t))

#f_i = [sp.signal.sawtooth(2 * np.pi * 5 * t) for t in t]
freq = f_i(t)
print('f_i: ', freq, 'Hz')
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

# Transmitted signal in baseband plot in time domain:
print("s_baseband(t): ", s_baseband(t))
plt.plot(t, s_baseband(t))
plt.grid()
plt.legend(['$s(t)$'])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('FM transmitted signal (radar chirp in time domain - real part)')
#go.Figure(data=go.Scatter(x=t, y=s(t))).show()

plt.show() # display plot


### TODO: ----------------------- FIX THE FFT ? ----------------------- ###
# FFT of signal (in baseband) => signal (FT) plot in frequency domain:
# ? shift the FFT ?
#plt.plot(freq, np.fft.fftshift(s(t))) # ? not freq, but -freq/2 to freq/2 nahhh
freq_range = np.fft.fftshift(np.fft.fftfreq(Number_of_samples, d=1/F_sampling_freq))
print('freq_range: ', freq_range, 'Hz')
print('fft: ', np.fft.fft(s_baseband(t)))
plt.plot(freq_range, np.fft.fftshift(s_baseband(t)))
plt.grid()
plt.legend(['$S(f)$'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Fourier transform of the signal (radar chirp in frequency domain - modulus)')
#go.Figure(data=go.Scatter(x=t, y=np.fft.fft(s(t)))).show()

plt.show() # display plot

#f = np.fft.fftshift(f)

# Bandwidth of the signal:
Bandwidth = 2*B_freq_range+2*(1/T_chirp_duration) # Car(l)son's rule
# or 15 MHz ? found
print('Bandwidth of the signal: B =', Bandwidth, 'Hz')

chirp_durations = np.linspace(1e-4,4e-4,endpoint=True)
plt.plot(chirp_durations,2*B_freq_range+2*(1/chirp_durations))
plt.grid()
plt.title('Bandwidth depending on the chirp duration (between .1ms and .4ms)')
plt.xlabel('Chirp duration (s)')
plt.ylabel('Bandwidth (Hz)')

plt.show()



# repliement spectral ? (aliasing) => fenêtre de Hamming
