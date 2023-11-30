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

max_range = 100  #! arbitrary, in m

## Signal in baseband
# def s_baseband(t):
#    return np.exp(1j * np.pi * Beta_slope * (t**2))

# --- 1. --- Generate the FMCW signal composed of K chirps
single_chirp = step1.signal_baseband  # used with step1.t
FMCW_over_K_chirps = K_slow_time_fft_size * single_chirp

## Fourier Transform
# fft_fmcw_signal = sft.fft(FMCW_over_K_chirps)
# fft_freq = sft.fftfreq(Number_of_samples, d=1 / #F_simulation_sampling_freq)
# fft_fmcw_signal_shifted = sft.fftshift(fft_fmcw_signal)
# fft_freq_shifted = sft.fftshift(fft_freq)

c = 3e8  # speed of light
tau_max = (2 * max_range) / c  # maximum possible delay
T_r = T_chirp_duration - tau_max  # duration
N_samples_per_chirp = int(T_r * F_radar_sampling_freq)
# K chirps are observed:
# samples can be organised in a N x K matrix

maximum_estimated_range = (
    F_radar_sampling_freq * T_chirp_duration * c / (2 * B_freq_range)
)
beat_freq_estimation = F_radar_sampling_freq / N_fast_time_fft_size  # = 1/T_r
speed_estimation_resolution = (1 / (K_slow_time_fft_size * T_chirp_duration)) * (
    c / (2 * f_c)
)

# The DFT over the chirp time index k (slow time) results in a Dirac pulse at the Dopppler frequency.

# N-FFT and K-FFT can be combined into a single 2D FFT of the N x K matrix of samples => Range Doppler Map (RDM)


def target_contribution(target_range, target_velocity):
    delay = (2 * target_range) / c
    freq_shift = 2 * target_velocity * f_c / c
    received_signal = np.exp(1j * 2 * np.pi * freq_shift * t) * np.exp(
        1j * 2 * np.pi * Beta_slope * (t - delay) ** 2
    )
    return received_signal


# single target:
target_range = 10  #! arbitrary, in m
target_velocity = 1  #! arbitrary, in m/s
target_signal = target_contribution(target_range, target_velocity)

# multiple targets:
target_ranges = [12, 26, 30]  #! arbitrary, in m
target_velocities = [1, 3, 2]  #! arbitrary, in m/s
target_signals = [
    target_contribution(target_ranges[i], target_velocities[i])
    for i in range(len(target_ranges))
]
target_signals_sum = sum(target_signals)

# --- 2. --- Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions

# mixing with the transmitted signal
mixed_signal = target_signals_sum * FMCW_over_K_chirps  # ? np.conj() needed?

# sampling at F_s
sampled_signal = mixed_signal[
    :: int(F_radar_sampling_freq * T_chirp_duration / N_samples_per_chirp)
]  #! ???

# S/P conversion
sp_conversion = sampled_signal.reshape((N_samples_per_chirp, -1))

# Fast time FFT
fast_time_fft = sft.fft(sp_conversion, axis=0)

# Slow time FFT
slow_time_fft = sft.fft(fast_time_fft, axis=1)

# --- 3. --- RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correct targets positions on the RDM.

#! TODO:
# generate multiple random scenarios



# --- 4. --- Compute the range and Doppler resolutions and discuss the relevance of the radar parameters for the considered scenario.

range_estimation_resolution = c / (2 * B_freq_range)
doppler_freq_estimation_resolution = 1 / (K_slow_time_fft_size * T_chirp_duration)

print("Range resolution: ", range_estimation_resolution)
print("Doppler resolution: ", doppler_freq_estimation_resolution)

# Relevance of radar parameters for the considered scenario:
# The frequency range B is the bandwidth of the transmitted signal. It directly determines the range resolution, inversely-proportional. For our case, it is set to 200 MHz.
# The chirp duration T_chirp_duration is the time duration of a chirp. It directly determines the Doppler frequency resolution, inversely-proportional. For our case, it is set between 0.1 and 0.4 ms.
# The number of chirps K_slow_time_fft_size is the number of chirps in a sequence. It directly determines the Doppler frequency resolution, inversely-proportional. For our case, it is set to 256.


def plot():
    ## Affichage
    # plt.figure(figsize=(12, 10))
    ## Signal FMCW
    # plt.subplot(3, 1, 1)
    # plt.plot(t, FMCW_over_K_chirps.real, label="Real")
    # plt.plot(t, FMCW_over_K_chirps.imag, label="Imaginary", #linestyle="--")
    # plt.grid()
    # plt.title("PSK Modulated FMCW Signal")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    ## Spectre du signal FMCW
    # plt.subplot(3, 1, 2)
    # plt.plot(fft_freq_shifted, np.abs#(fft_fmcw_signal_shifted))
    # plt.grid()
    # plt.title("Spectrum of PSK Modulated FMCW Signal")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude")
    # plt.tight_layout()
    plt.show()

    #! TODO: Simulate the impact of the single-target channel on the FMCW signal (the extension to a multi-target channel is obviously the sum of the target contributions)

    #! TODO: Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions

    #! TODO: RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correc targets positions on the RDM.

    #! TODO: Compute the range and Doppler resolutions and discuss the relevance of the radar parameters for the considered scenario.


if __name__ == "__main__":
    plot()
