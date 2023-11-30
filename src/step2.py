"""
ELEC-H311 - Joint Communications and Radar simulation project
Lucas Placentino
Salman Houdaibi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft

import step1

"""
    Step 2: PSK Modulation of FMCW Signal
"""

# Paramètres existants
T_chirp_duration = 2e-4  # Durée du chirp
Number_of_samples = 2**18
B_freq_range = 200e6
f_c = 24e9
F_simulation_sampling_freq = 512e6
F_radar_sampling_freq = 2e6
Beta_slope = B_freq_range / T_chirp_duration
t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)
N_fast_time_fft_size = 512
K_slow_time_fft_size = 256


target_range = 10  #! arbitrary, in m
target_velocity = 1  #! arbitrary, in m/s


## Signal in baseband
#def s_baseband(t):
#    return np.exp(1j * np.pi * Beta_slope * (t**2))

# --- 1. --- Generate the FMCW signal composed of K chirps
single_chirp = step1.signal_baseband  # used with step1.t
FMCW_over_K_chirps = K_slow_time_fft_size * single_chirp

## Fourier Transform
#fft_fmcw_signal = sft.fft(FMCW_over_K_chirps)
#fft_freq = sft.fftfreq(Number_of_samples, d=1 / #F_simulation_sampling_freq)
#fft_fmcw_signal_shifted = sft.fftshift(fft_fmcw_signal)
#fft_freq_shifted = sft.fftshift(fft_freq)


c = 3e8  # speed of light
delay = (2 * target_range) / c
freq_shift = 2 * target_velocity * f_c / c
tau_max = delay  # ? maximum possible delay
T_r = T_chirp_duration - tau_max # duration
N_samples_per_chirp = int(T_r * F_simulation_sampling_freq)
# K chirps are observed:
# samples can be organised in a N x K matrix

maximum_estimated_range = F_radar_sampling_freq*T_chirp_duration*c/(2*B_freq_range)
beat_freq_estimation = F_radar_sampling_freq/N_fast_time_fft_size # = 1/T_r
range_estimation_resolution = c/(2*B_freq_range)
doppler_freq_estimation = 1/(K_slow_time_fft_size*T_chirp_duration)
speed_estimation_resolution = (1/(K_slow_time_fft_size*T_chirp_duration))*(c/(2*f_c))

# The DFT over the chirp time index k (slow time) results in a Dirac pulse at the Dopppler frequency.

# N-FFT and K-FFT can be combined into a single 2D FFT of the N x K matrix of samples => Range Doppler Map (RDM)



def main():
    ## Affichage
    #plt.figure(figsize=(12, 10))
    ## Signal FMCW
    #plt.subplot(3, 1, 1)
    #plt.plot(t, FMCW_over_K_chirps.real, label="Real")
    #plt.plot(t, FMCW_over_K_chirps.imag, label="Imaginary", #linestyle="--")
    #plt.grid()
    #plt.title("PSK Modulated FMCW Signal")
    #plt.xlabel("Time (s)")
    #plt.ylabel("Amplitude")
    #plt.legend()
    ## Spectre du signal FMCW
    #plt.subplot(3, 1, 2)
    #plt.plot(fft_freq_shifted, np.abs#(fft_fmcw_signal_shifted))
    #plt.grid()
    #plt.title("Spectrum of PSK Modulated FMCW Signal")
    #plt.xlabel("Frequency (Hz)")
    #plt.ylabel("Magnitude")
    #plt.tight_layout()
    #plt.show()

    #! TODO: Simulate the impact of the single-target channel on the FMCW signal (the extension to a multi-target channel is obviously the sum of the target contributions)

    #! TODO: Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions

    #! TODO: RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correc targets positions on the RDM.

    #! TODO: Compute the range and Doppler resolutions and discuss the relevance of the radar parameters for the considered scenario.


if __name__ == "__main__":
    main()
