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
targets = [(11, 3), (20, 5), (3, 0.4), (34, -2), (25, 0)]
signal_multiple_targets = sum(
    target_contribution(range, speed) for range, speed in targets
)

# --- 2. --- Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions

# mixing with the transmitted signal
mixed_signal = signal_multiple_targets * FMCW_over_K_chirps  # ? np.conj() needed?

# sampling at F_s
sampled_signal = mixed_signal[
    :: int(F_radar_sampling_freq * T_chirp_duration / N_samples_per_chirp)
]  #! TODO: ayo what is this sh*t ???

# S/P conversion
# sp_conversion = sampled_signal.reshape((N_samples_per_chirp, -1))

# Fast time FFT
# fast_time_fft = sft.fft(sp_conversion, axis=0)

# Slow time FFT
# slow_time_fft = sft.fft(fast_time_fft, axis=1)

# --- 3. --- RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correct targets positions on the RDM.

#! TODO: should use the FFTs above (---2---)
# generate multiple random scenarios
Rx = signal_multiple_targets

Nr = 512  #! number of range cells / OR number of samples on each chirp ?
Nd = 256  # number of doppler cells / number of chirps in one sequence
# t_rdm = np.linspace(0, Nr * Nd, int(Nd * T_chirp_duration), endpoint=True)  # ? correct ? int() ? needed ?

doppler_frequencies = np.fft.fftshift(np.fft.fftfreq(Nd, 1 / F_radar_sampling_freq))
range_values = np.linspace(0, max_range, Nr)

rdm = np.zeros((Nr, Nd), dtype=complex)
doppler_profile = np.zeros((Nr, Nd), dtype=complex)

for i, r in enumerate(range_values):
    # Additive white Gaussian noise (AWGN)
    # noise_power = np.mean(np.abs(Tx) ** 2) / SNR_lin  #! SNR value computed or arbitrary ?
    AGWN = np.random.normal(0, 1, Number_of_samples) + 1j * np.random.normal(
        0, 1, Number_of_samples
    )  # complex noise, both real and imaginary parts are independant and are white noise
    #! noise takes SNR in input ? -> through noise_power ?
    Rx_noise = Rx + AGWN  # received signal with noise
    range_profile = sft.fft(Rx_noise, Nr)
    doppler_profile = sft.fftshift(sft.fft(range_profile, Nd))  # ,axes=0 or nothing ?
    rdm[i, :] = doppler_profile

# plot the RDM
plt.figure(figsize=(10, 6))
# plt.imshow(np.abs(doppler_profile.reshape(1, -1)), extent=[-Nd / 2, Nd / 2, 0, Nr], cmap="jet", aspect="auto")  # , vmin=0, vmax=1) #? vmin and vmax ?
plt.imshow(
    np.abs(rdm),
    extent=[doppler_frequencies[0], doppler_frequencies[-1], 0, max_range],
    cmap="jet",
    aspect="auto",
)  # , vmin=0, vmax=1) #? vmin and vmax ?

plt.title("Range Doppler Map")
plt.xlabel("Doppler (Hz)")
plt.ylabel("Range (m)")
plt.colorbar(label="Amplitude")  # ? what Amplitude ? should we put it in dB ?
# plt.tight_layout()
plt.show()


# --- 4. --- Compute the range and Doppler resolutions and discuss the relevance of the radar parameters for the considered scenario.

range_estimation_resolution = c / (2 * B_freq_range)
doppler_freq_estimation_resolution = 1 / (K_slow_time_fft_size * T_chirp_duration)

print("Range resolution: ", range_estimation_resolution)
print("Doppler resolution: ", doppler_freq_estimation_resolution)

print("--- Relevance of radar parameters for the considered scenario: ---")
print("The frequency range B is the bandwidth of the transmitted signal. It directly determines the range resolution, inversely-proportional. For our case, it is set to 200 MHz.")
print("The chirp duration T_chirp_duration is the time duration of a chirp. It directly determines the Doppler frequency resolution, inversely-proportional. For our case, it is set between 0.1 and 0.4 ms.")
print("The number of chirps K_slow_time_fft_size is the number of chirps in a sequence. It directly determines the Doppler frequency resolution, inversely-proportional. For our case, it is set to 256.")


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

    #! done ? TODO: Simulate the impact of the single-target channel on the FMCW signal (the extension to a multi-target channel is obviously the sum of the target contributions)

    #! S/P does,'t work ? TODO: Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions

    #! done somewhat ? TODO: RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correc targets positions on the RDM.

    #* DONE Compute the range and Doppler resolutions and discuss the relevance of the radar parameters for the considered scenario.


if __name__ == "__main__":
    plot()
