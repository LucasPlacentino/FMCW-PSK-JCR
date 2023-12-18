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

# Parameters
T_chirp_duration = 2e-4  # chirp duration
Number_of_samples = 2**18
B_freq_range = 200e6
f_c = 24e9  # carrier frequency
F_simulation_sampling_freq = 512e6
F_radar_sampling_freq = 2e6
Beta_slope = B_freq_range / T_chirp_duration
# t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)
N_fast_time_fft_size = 512
K_slow_time_fft_size = 256

max_range = 20  # given
max_speed = 2  # given

## Signal in baseband
# def s_baseband(t):
#    return np.exp(1j * np.pi * Beta_slope * (t**2))

# --- 1. --- Generate the FMCW signal composed of K chirps
single_chirp_signal = step1.signal_baseband  # used with step1.t
# FMCW_over_K_chirps = K_slow_time_fft_size * single_chirp
FMCW_over_K_chirps = np.tile(single_chirp_signal, K_slow_time_fft_size)
t_over_k_chirps = np.linspace(
    0,
    T_chirp_duration * K_slow_time_fft_size,
    Number_of_samples * K_slow_time_fft_size,
    endpoint=True,
)
# FMCW_over_K_chirps = FMCW_over_K_chirps.reshape((K_slow_time_fft_size, -1)) # will do the S/P conversion later
print("FMCW_over_K_chirps: ", FMCW_over_K_chirps)

c = 3e8  # speed of light
tau_max = (2 * max_range) / c  # maximum possible delay
T_r = T_chirp_duration - tau_max  # duration
# N_samples_per_chirp = int(T_r * F_radar_sampling_freq) # ?
N_samples_per_chirp = Number_of_samples // K_slow_time_fft_size  # int division
# K chirps are observed:
# samples can be organised in a N x K matrix

# maximum_estimated_range = (F_radar_sampling_freq * T_chirp_duration * c / (2 * B_freq_range))
maximum_estimated_range = f_c * T_chirp_duration * c / (2 * B_freq_range)
print("maximum_estimated_range: ", maximum_estimated_range)
beat_freq_estimation = F_radar_sampling_freq / N_fast_time_fft_size  # = 1/T_r
print("beat_freq_estimation: ", beat_freq_estimation)
speed_estimation_resolution = (1 / (K_slow_time_fft_size * T_chirp_duration)) * (
    c / (2 * f_c)
)
print("speed_estimation_resolution: ", speed_estimation_resolution)

# The DFT over the chirp time index k (slow time) results in a Dirac pulse at the Dopppler frequency.

# N-FFT and K-FFT can be combined into a single 2D FFT of the N x K matrix of samples => Range Doppler Map (RDM)


# TODO: input FMWC_over_K_chirps signal
def target_contribution(target_range, target_velocity):  # , signal):
    delay = (2 * target_range) / c
    freq_shift = 2 * target_velocity * f_c / c

    # target_signal = np.exp(1j * 2 * np.pi * freq_shift * t) * np.exp(1j * 2 * np.pi * Beta_slope * (t - delay) ** 2)

    # target_signal = (
    #    signal
    #    * np.exp(1j * 2 * np.pi * freq_shift * t)
    #    * np.exp(
    #        1j * 2 * np.pi * B_freq_range * (t - delay) ** 2 / (2 * T_chirp_duration)
    #    )
    # )

    ## apply to each chirp:
    # target_signal = np.zeros_like(signal, dtype=complex)
    # for k in range(signal.shape[0]):
    #    target_signal[k, :] = (
    #        signal[k, :]
    #        * np.exp(1j * 2 * np.pi * freq_shift * t)
    #        * np.exp(
    #            1j
    #            * 2
    #            * np.pi
    #            * B_freq_range
    #            * (t - delay) ** 2
    #            / (2 * T_chirp_duration)
    #        )
    #    )

    target_signal = np.exp(1j * 2 * np.pi * freq_shift * t_over_k_chirps) * np.exp(
        1j
        * 2
        * np.pi
        * Beta_slope
        * (t_over_k_chirps - delay) ** 2
        / (2 * T_chirp_duration)
    )

    return target_signal


# max range is 20m, max speed is 2m/s
# single target:
target_range = 10  #! arbitrary, in m
target_velocity = 1  #! arbitrary, in m/s
signal_st = target_contribution(target_range, target_velocity)  # , FMCW_over_K_chirps)
print("signal_st: ", signal_st)

# multiple targets:
targets = [(11, 1.2), (20, 1), (3, 0.4), (7, -2), (15, 0)]
# signal_mt = sum(
#    target_contribution(range, speed, FMCW_over_K_chirps) for range, speed in targets
# )

# --- 2. --- Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions


# TODO: put in this function to reuse in step 3 ?
def radar_processing(signal):
    pass


# mixing with the transmitted signal
# mixed_signal_mt = (
#    signal_mt * FMCW_over_K_chirps
# )  # ? np.conj() needed?
mixed_signal_st = signal_st * FMCW_over_K_chirps  # ? np.conj() needed?

# sampling at F_s:
sampling_rate = F_radar_sampling_freq * T_chirp_duration / N_samples_per_chirp
sampling_interval = int(round(1 / sampling_rate))
# sampled_signal_mt = mixed_signal_mt[
#    :: int(
#        F_radar_sampling_freq * T_chirp_duration / N_samples_per_chirp
#    )  # this is step size, between each sample
# ]  #! TODO: is this sampling correct ?
# Ensure that the indices are within the signal length
end_index_st = min(len(mixed_signal_st), len(mixed_signal_st) - 1)
# sampled_signal_st = mixed_signal_st[
#    :: (F_radar_sampling_freq * T_chirp_duration / N_samples_per_chirp)
# ]  #! TODO: is this sampling correct ? # single target
sampled_signal_st = mixed_signal_st[::sampling_interval][: end_index_st + 1]
print("sampled_signal_st: ", sampled_signal_st)

# S/P conversion
# sp_conversion_mt = sampled_signal_mt.reshape((N_samples_per_chirp, -1)) # multiple targets
# sp_conversion_st = sampled_signal_st.reshape((N_samples_per_chirp, -1))  # single target
sp_conversion_st = sampled_signal_st.reshape(
    (len(sampled_signal_st), -1)
)  # single target

# Fast time FFT
# fast_time_fft_mt = sft.fft(sp_conversion_mt, axis=0) # multiple targets
fast_time_fft_st = sft.fft(sp_conversion_st, axis=0)  # single target

# Slow time FFT
# slow_time_fft_mt = sft.fft(fast_time_fft_mt, axis=1) # multiple targets
slow_time_fft_st = sft.fft(fast_time_fft_st, axis=1)  # single target

# --- 3. --- RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correct targets positions on the RDM.

#! TODO: should use the FFTs above (---2---)

range_estimation_resolution = c / (2 * B_freq_range)
doppler_freq_estimation_resolution = 1 / (K_slow_time_fft_size * T_chirp_duration)
# ? not above ?


def to_physical_units(start, index, resolution):
    return start + index * resolution


# range_bins = to_physical_units(
#    0, np.arange(N_fast_time_fft_size), range_estimation_resolution
# )
# doppler_bins = to_physical_units(
#    -K_slow_time_fft_size / 2,
#    np.arange(K_slow_time_fft_size),
#    doppler_freq_estimation_resolution,
# )

# in dB:
slow_time_fft_st_db = np.log10(
    np.abs(slow_time_fft_st) + 1e-12
)  # + 1e-12 to avoid log(0) #? 20 * np.log(...) ?

plt.figure(figsize=(10, 6))
plt.imshow(
    np.abs(slow_time_fft_st),
    # slow_time_fft_st_db,
    aspect="auto",
    extent=[0, slow_time_fft_st.shape[1], 0, slow_time_fft_st.shape[0]],
    # extent=[doppler_bins[0], doppler_bins[-1], range_bins[-1], range_bins[0]],
)
plt.title("Range-Doppler Map (RDM)")
plt.xlabel("Doppler Bins")
plt.ylabel("Range Bins")
plt.colorbar(label="Amplitude")
plt.show()
#! TODO: put range and doppler axes in meters and Hz ?


"""
# generate multiple random scenarios
# Rx = mixed_signal_multiple_targets  #! TODO: use sampled_signal_mt instead ?
Rx = sampled_signal_st

Nr = N_fast_time_fft_size  # 512  #! number of range cells / OR number of samples on each chirp ?
Nd = K_slow_time_fft_size  # 256  # number of doppler cells / number of chirps in one sequence
# t_rdm = np.linspace(0, Nr * Nd, int(Nd * T_chirp_duration), endpoint=True)  # ? correct ? int() ? needed ?

doppler_frequencies = np.fft.fftshift(np.fft.fftfreq(Nd, 1 / F_radar_sampling_freq))
range_values = np.linspace(0, max_range, Nr)

rdm = np.zeros((Nr, Nd), dtype=complex)
doppler_profile = np.zeros((Nr, Nd), dtype=complex)

for i, r in enumerate(range_values):
    ## Additive white Gaussian noise (AWGN)
    ## noise_power = np.mean(np.abs(Tx) ** 2) / SNR_lin  # SNR value computed or arbitrary ?
    # AGWN = np.random.normal(0, 1, Number_of_samples) + 1j * np.random.normal(
    #    0, 1, Number_of_samples
    # )  # complex noise, both real and imaginary parts are independant and are white noise
    ## noise takes SNR in input ? -> through noise_power ?
    # Rx_noise = Rx + AGWN  # received signal with noise
    # range_profile = sft.fft(Rx_noise, Nr)

    range_profile = sft.fft(Rx, Nr)
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
"""


# --- 4. --- Compute the range and Doppler resolutions and discuss the relevance of the radar parameters for the considered scenario.

# see --- 3. --- for computing resolutions

print("Range resolution: ", range_estimation_resolution, " m?")
print("Doppler resolution: ", doppler_freq_estimation_resolution, " Hz?")

print("--- Relevance of radar parameters for the considered scenario: ---")
print(
    "The frequency range B is the bandwidth of the transmitted signal. It directly determines the range resolution, inversely-proportional. For our case, it is set to 200 MHz."
)
print(
    "The chirp duration T_chirp_duration is the time duration of a chirp. It directly determines the Doppler frequency resolution, inversely-proportional. For our case, it is set between 0.1 and 0.4 ms."
)
print(
    "The number of chirps K_slow_time_fft_size is the number of chirps in a sequence. It directly determines the Doppler frequency resolution, inversely-proportional. For our case, it is set to 256."
)


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

    #! fix with above TODO: RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correc targets positions on the RDM.

    # * DONE Compute the range and Doppler resolutions and discuss the relevance of the radar parameters for the considered scenario.


if __name__ == "__main__":
    plot()
