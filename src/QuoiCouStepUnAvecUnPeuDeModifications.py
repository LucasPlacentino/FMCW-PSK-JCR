"""
ELEC-H311 - Joint Communications and Radar simulation project
LPulcaacsentino
SHaolumdaanibi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft

"""
    Step 1: FMCW signal
"""

# Parameters
T_chirp_duration = 2e-4  # chirp duration: 0.1 ms to 0.4 ms
Number_of_samples = 2**18  # 2**18 samples
B_freq_range = 200e6  # 200 MHz
f_c = 24e9  # Carrier frequency 24 GHz
F_sampling_freq = 512e6  # 512 MHz
Beta_slope = B_freq_range / T_chirp_duration  # B = Beta*T

# Time vector
t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)


# Instantaneous frequency
def f_i(t):
    return Beta_slope * t


# Instantaneous phase
def phi_i(t):
    return np.pi * Beta_slope * (t**2)


# Baseband signal
def s_baseband(t):
    return np.exp(1j * phi_i(t))


# Calculate instantaneous frequency and baseband signal
freq_instantaneous = f_i(t)
signal_baseband = s_baseband(t)

# Fourier Transform
fft_signal = sft.fft(signal_baseband)
fft_freq = sft.fftfreq(Number_of_samples, d=1 / F_sampling_freq)
fft_shifted = sft.fftshift(fft_signal)
fft_freq_shifted = sft.fftshift(fft_freq)

def main():
    # Instantaneous frequency plot
    plt.figure(figsize=(12, 4))
    plt.plot(t, freq_instantaneous)
    plt.grid()
    plt.title("Instantaneous frequency $f_i(t)$")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

    # Baseband signal plot
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal_baseband.real, label="Real")
    plt.grid()
    plt.title("Real part of the baseband signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # FFT plot
    plt.figure(figsize=(12, 4))
    plt.plot(fft_freq_shifted, 20 * np.log10(np.abs(fft_shifted)/np.max(np.abs(fft_shifted))))
    plt.grid()
    plt.title("Fourier Transform of Baseband Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.show()

    # Bandwidth calculation #! correct?
    Bandwidth = 2 * B_freq_range + 2 * (1 / T_chirp_duration)
    print("Bandwidth of the signal: "+Bandwidth+" Hz")

    ##! not correct? AND NOT NECESSARY
    ## Bandwidth as a function of chirp duration
    #chirp_durations = np.linspace(1e-4, 4e-4, 100, endpoint=True)
    #bandwidths = 2 * B_freq_range + 2 * (1 / chirp_durations)
    #plt.figure(figsize=(12, 4))
    #plt.plot(chirp_durations, bandwidths)
    #plt.grid()
    #plt.title("Bandwidth depending on the chirp duration (between 0.1ms and 0.4ms)")
    #plt.xlabel("Chirp duration (s)")
    #plt.ylabel("Bandwidth (Hz)")
    #plt.show()

if __name__ == "__main__":
    main()
