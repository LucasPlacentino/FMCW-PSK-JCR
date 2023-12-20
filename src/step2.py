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

"why?"
# test:
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
b = np.reshape(a, (2, 6))
c = np.transpose(b)
print(a)
print(b)
print(c)


# Parameters
T_chirp_duration = 2e-4  # seconds, chirp duration
Number_of_samples = 2**18
B_freq_range = 200e6  # Hz
f_c = 24e9  # Hz, carrier frequency
F_simulation_sampling_freq = 512e6  # Hz
F_radar_sampling_freq = 2e6  # Hz
Beta_slope = B_freq_range / T_chirp_duration
# t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)
N_fast_time_fft_size = 512
K_slow_time_fft_size = 256
c = 3e8  # m/s, speed of light
max_range = 20  # m, given
max_speed = 2  # m/s, given
tau_max = (2 * max_range) / c  # maximum possible delay
#number_of_targets = 5  # arbitrary # could be asked for input
number_of_targets = int(input("Enter number of targets (int), e.i 5 : "))
guard_samples = 5 # given


radar_max_range = (c * F_radar_sampling_freq) / (
    2 * Beta_slope
)  # ? according to https://wirelesspi.com/fmcw-radar-part-3-design-guidelines/
print("radar theoretical maximum range ? : ", radar_max_range, "m")

# --- 1. --- Generate the FMCW signal composed of K chirps

## Signal in baseband
# def s_baseband(t):
#    return np.exp(1j * np.pi * Beta_slope * (t**2))

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
# print("FMCW_over_K_chirps: ", FMCW_over_K_chirps)


T_r = (
    T_chirp_duration - tau_max
)  # duration #! not correct? TODO: changes over chirps and targets?
N_samples_per_chirp = (
    F_radar_sampling_freq * T_r
)  # should be this one, see document #? useful?
print("N_samples_per_chirp: ", N_samples_per_chirp, "sould be 512?")

# K chirps are observed:
# samples can be organised in a N x K matrix

##maximum_estimated_range = (F_radar_sampling_freq * T_chirp_duration * c / (2 * B_freq_range))
# maximum_estimated_range = f_c * T_chirp_duration * c / (2 * B_freq_range)
# print("maximum_estimated_range: ", maximum_estimated_range)
# beat_freq_estimation = F_radar_sampling_freq / N_fast_time_fft_size  # = 1/T_r
# print("beat_freq_estimation: ", beat_freq_estimation)
# speed_estimation_resolution = (1 / (K_slow_time_fft_size * T_chirp_duration)) * (c / (2 * f_c))
# print("speed_estimation_resolution: ", speed_estimation_resolution)

# The DFT over the chirp time index k (slow time) results in a Dirac pulse at the Dopppler frequency.

# N-FFT and K-FFT can be combined into a single 2D FFT of the N x K matrix of samples => Range Doppler Map (RDM)

target_delays = (
    np.random.rand(number_of_targets) * tau_max
)  # random delay for each target
target_velocities = (
    np.random.rand(number_of_targets) * max_speed
)  # random speed for each target
for i in range(number_of_targets):
    print(
        "Target",
        i + 1,
        "- delay:",
        "{:.2f}".format(target_delays[i] * 1e9),
        "ns",
        end=", ",
    )
    print("velocity:", "{:.2f}".format(target_velocities[i]), "m/s")
#R_0 = c * target_delays / 2  # initial range

# target_beat_frequency = 1
# F_doppler_shift = 1
radar_wavelength = c / f_c
# velocity = (F_doppler_shift/2) * (radar_wavelength)
# distance = (c*target_beat_frequency) / (2*Beta_slope)


# ! FIXME: TODO: ?
def target_contribution(target_delay, target_velocity):#(target_range, target_velocity):  # , signal):
    #delay = (2 * target_range) / c
    #freq_shift = 2 * target_velocity * f_c / c

    R_0_initial_range = c * target_delay / 2
    doppler_freq = 2 * target_velocity * f_c / c
    beat_frequency = 2 * R_0_initial_range * Beta_slope / c
    kappa = np.exp(1j*4*np.pi*R_0_initial_range*f_c/c)*np.exp(1j*(-2)*np.pi* (Beta_slope**2) * (R_0_initial_range**2) / (c**2)) # ? complex factor
    t_prime = np.arange(0, T_chirp_duration, T_chirp_duration / (N_fast_time_fft_size + guard_samples)) #! "sampled time index t′ (the fast time index)" # or t over 1 chirp ?
    # ! use guard samples ?
    print("t_prime shape:",t_prime.shape)
    print(t_prime)

    complex_conjugated_video_signal = np.array([], dtype=complex)
    #complex_conjugated_video_signal = np.concatenate([kappa * np.exp(1j * 2 * np.pi * beat_frequency * t_prime[k])*np.exp(1j * 2 * np.pi * doppler_freq * k * T_chirp_duration)] for k in range(K_slow_time_fft_size))
    for k in range(K_slow_time_fft_size):
        complex_conjugated_video_signal_one_sample = np.array([], dtype=complex)
        np.concatenate(complex_conjugated_video_signal_one_sample,[kappa * np.exp(1j * 2 * np.pi * beat_frequency * t_prime[n])*np.exp(1j * 2 * np.pi * doppler_freq * k * T_chirp_duration) for n in N_fast_time_fft_size])
        print("vid signal one sample",k,complex_conjugated_video_signal_one_sample.shape)
        #complex_conjugated_video_signal_one_sample = np.array([], dtype=complex)
        #for n in range(N_fast_time_fft_size):
        #    np.concatenate(complex_conjugated_video_signal_one_sample,kappa * np.exp(1j * 2 * np.pi * beat_frequency * t_prime[n])*np.exp(1j * 2 * np.pi * doppler_freq * k * T_chirp_duration))
        np.concatenate(complex_conjugated_video_signal, complex_conjugated_video_signal_one_sample)
        print("vid signal",k,complex_conjugated_video_signal.shape)
    #target_signal = np.exp(1j * np.pi * Beta_slope * (t_over_k_chirps**2))
    target_signal = complex_conjugated_video_signal

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

    # target_signal = np.exp(1j * 2 * np.pi * freq_shift * t_over_k_chirps) * np.exp(
    #     1j
    #     * 2
    #     * np.pi
    #     * Beta_slope
    #     * (t_over_k_chirps - delay) ** 2
    #     / (2 * T_chirp_duration)
    # )

    return target_signal


## max range is 20m, max speed is 2m/s
## single target:
#target_range = 10  #! arbitrary, in m
#target_velocity = 1  #! arbitrary, in m/s
#signal_st = target_contribution(target_range, target_velocity)  # , FMCW_over_K_chirps)
#print("signal_st: ", signal_st)
#
## multiple targets:
#targets = [(11, 1.2), (20, 1), (3, 0.4), (7, -2), (15, 0)]
## signal_mt = sum(
##    target_contribution(range, speed, FMCW_over_K_chirps) for range, speed in targets
## )

signal_target = sum(target_contribution(target_delays[i], target_velocities[i]) for i in range(number_of_targets))
print("signal_target shape",signal_target.shape)

# --- 2. --- Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions


# TODO: put in this function to reuse in step 3 ?
def radar_processing(signal):
    pass


"""
# target_reception_times = np.zeros((number_of_targets, K_slow_time_fft_size)) #?
targets_reception_times_matrices = [
    np.zeros((K_slow_time_fft_size, N_fast_time_fft_size))
    for i in range(number_of_targets)
]  # each target's reception times matrix

emission_time_matrix = np.zeros((K_slow_time_fft_size, N_fast_time_fft_size))  # ?
phaseshift_at_t_matrix = np.zeros((K_slow_time_fft_size, N_fast_time_fft_size))  # ?
total_phaseshift_at_t = []
total_emission_time = []
total_reception_time = []
total_reception_time_vector = []

# TODO: optimize this? :
for i in range(number_of_targets):
    target_reception_time_matrix = np.zeros(
        (K_slow_time_fft_size, N_fast_time_fft_size)
    )  # ?
    # reception times for each chirp's samples:
    for j in range(K_slow_time_fft_size):
        target_reception_time = np.arange(
            target_delays[i],
            T_chirp_duration + target_delays[i],
            T_chirp_duration / N_fast_time_fft_size,
        )  # ?
        target_reception_time_matrix[j, :] = target_reception_time[
            :N_fast_time_fft_size
        ]  # ?
    targets_reception_times_matrices[i] = target_reception_time_matrix  # ?

# emission times and instant frequency for each chirp's samples:
# TODO: ?
for i in range(K_slow_time_fft_size):
    emission_time = np.arange(
        0, T_chirp_duration, T_chirp_duration / N_fast_time_fft_size
    )  # ?
    instant_freq_at_t = Beta_slope * emission_time  # ?
    emission_time_matrix[i, :] = emission_time[:N_fast_time_fft_size]  # ?
    # phaseshift_at_t = np.pi * instant_freq_at_t[:N_fast_time_fft_size] * emission_time[:N_fast_time_fft_size] # ?
    phaseshift_at_t_matrix = instant_freq_at_t[:N_fast_time_fft_size]  # ?

# put each chirp's emission times in a single vector:
for i in range(K_slow_time_fft_size):
    # instant_freq_at_t = Beta_slope * emission_time_matrix[i, :] # ?
    instant_freq_at_t = phaseshift_at_t_matrix[i, :]  # ?
    emission_time = emission_time_matrix[i, :]  # ?
    total_phaseshift_at_t = np.concatenate(
        (total_phaseshift_at_t, instant_freq_at_t)
    )  # ?
    _time_at_chirp_start = (i - 1) * T_chirp_duration
    total_emission_time = np.concatenate(
        (total_emission_time, _time_at_chirp_start + emission_time)
    )  # ?

# put each target's (and chirp) reception times in a single matrix:
for i in range(number_of_targets):
    _reception_time_matrix_current = targets_reception_times_matrices[i]  # ?
    for j in range(K_slow_time_fft_size):
        target_reception_time = _reception_time_matrix_current[j, :]  # ?
        _time_at_chirp_start = (j - 1) * T_chirp_duration
        total_reception_time_vector = np.concatenate(
            (total_reception_time, _time_at_chirp_start + target_reception_time)
        )  # ?
    total_reception_time[i, :] = total_reception_time_vector  # ?
    total_reception_time_vector = []  # ?

rdm_single = np.zeros((N_fast_time_fft_size, K_slow_time_fft_size))#, dtype=complex)  # ?
rdm_multiple = np.zeros((N_fast_time_fft_size, K_slow_time_fft_size))#, dtype=complex) # ?

for target in range(number_of_targets):
    pass

plt.figure(figsize=(10, 6))
plt.imshow(
    np.abs(rdm_multiple),
    # slow_time_fft_st_db,
    aspect="auto",
    cmap="jet",
    # extent=[0, slow_time_fft_st.shape[1], 0, slow_time_fft_st.shape[0]],
    extent=[0, K_slow_time_fft_size, 0, N_fast_time_fft_size],
    # extent=[doppler_bins[0], doppler_bins[-1], range_bins[-1], range_bins[0]],
)
plt.title("Range-Doppler Map (RDM)")
plt.xlabel("Doppler Bins")
plt.ylabel("Range Bins")
plt.colorbar(label="Amplitude")
# plt.colorbar(label="Amplitude (dB)")
plt.show()
"""

# mixing with the transmitted signal
# mixed_signal_mt = (
#    signal_mt * FMCW_over_K_chirps
# )  # ? np.conj() needed?
mixed_signal_st = signal_target * FMCW_over_K_chirps  # ? np.conj() needed?

# sampling :

# sampling_rate = F_radar_sampling_freq * T_chirp_duration / N_samples_per_chirp
# sampling_interval = int(round(1 / sampling_rate))

# sampling_interval = len(single_chirp_signal) / N_samples_per_chirp# should ne this one
# sampling_interval = T_chirp_duration / N_samples_per_chirp

# sampled_signal_mt = mixed_signal_mt[
#    :: int(
#        F_radar_sampling_freq * T_chirp_duration / N_samples_per_chirp
#    )  # this is step size, between each sample
# ]  #! TODO: is this sampling correct ?
# Ensure that the indices are within the signal length
# end_index_st = min(len(mixed_signal_st), len(mixed_signal_st) - 1)
# sampled_signal_st = mixed_signal_st[
#    :: (F_radar_sampling_freq * T_chirp_duration / N_samples_per_chirp)
# ]  #! TODO: is this sampling correct ? # single target

# samples mixed_signal_st with N_samples_per_chirp samples per chirp:
total_samples_over_K_chirps = N_fast_time_fft_size * K_slow_time_fft_size
sampling_interval = len(mixed_signal_st) / total_samples_over_K_chirps
sampled_signal_st = mixed_signal_st[:: int(sampling_interval)]  # Sample the signal

# sampled_signal_st = mixed_signal_st[::sampling_interval]#[: end_index_st + 1]
print("sampled_signal_st: ", sampled_signal_st, "length: ", len(sampled_signal_st))

# S/P conversion
# sp_conversion_mt = sampled_signal_mt.reshape((N_samples_per_chirp, -1)) # multiple targets
# sp_conversion_st = sampled_signal_st.reshape((N_samples_per_chirp, -1)) # single target
# sp_conversion_st = sampled_signal_st.reshape(
#    (len(sampled_signal_st), -1)
# )  # single target
# Serial to Parallel conversion, each column is a chirp, and each row of a chirp is a sample:
# sp_conversion_st = sampled_signal_st.reshape((N_samples_per_chirp, -1))  # single target
sp_conversion_st = np.reshape(
    sampled_signal_st, (K_slow_time_fft_size, N_fast_time_fft_size)
)  # single target, width:512 samples per chirp, height:256 chirps, needs to be transposed
sp_conversion_st = np.transpose(sp_conversion_st)
# print("height"+str(len(sp_conversion_st)), "width"+str(len(sp_conversion_st[0])))
print(
    "sp_conversion_st shape:",
    sp_conversion_st.shape,
    "(height, width), or (rows, columns)",
)

# Fast time FFT
# fast_time_fft_mt = sft.fft(sp_conversion_mt, axis=0) # multiple targets
fast_time_fft_st = np.fft.fft(
    sp_conversion_st, axis=0
)  # single target, axis=0 means columns
# fast_time_fft_st = sft.fftn(sp_conversion_st, axis=0)  # single target, axis=0 means columns

# Slow time FFT
# slow_time_fft_mt = sft.fft(fast_time_fft_mt, axis=1) # multiple targets
slow_time_fft_st = np.fft.fft(
    fast_time_fft_st, axis=1
)  # single target, axis=1 means rows
# slow_time_fft_st = sft.fftn(fast_time_fft_st, axis=1) # single target, axis=1 means rows

# debug:
print(slow_time_fft_st.shape)

# --- 3. --- RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correct targets positions on the RDM.

#! TODO: should use the FFTs above (---2---)

range_estimation_resolution = c / (2 * B_freq_range)
doppler_freq_estimation_resolution = 1 / (K_slow_time_fft_size * T_chirp_duration)
# ? not above ?


def to_physical_units(start, index, resolution):
    return start + index * resolution


range_bins = to_physical_units(
    0, np.arange(N_fast_time_fft_size), range_estimation_resolution
)
doppler_bins = to_physical_units(
    -K_slow_time_fft_size / 2,
    np.arange(K_slow_time_fft_size),
    doppler_freq_estimation_resolution,
)
## in dB:
# slow_time_fft_st_db = 20 * np.log10(np.abs(slow_time_fft_st) + 1e-12) / np.max(np.abs(slow_time_fft_st)) # + 1e-12 to avoid log(0) #? 20 * np.log(...) ?

plt.figure(figsize=(10, 6))
plt.imshow(
    np.abs(slow_time_fft_st),
    # slow_time_fft_st_db,
    aspect="auto",
    cmap="jet",
    # extent=[0, slow_time_fft_st.shape[1], 0, slow_time_fft_st.shape[0]],
    extent=[0, K_slow_time_fft_size, 0, N_fast_time_fft_size],
    # extent=[doppler_bins[0], doppler_bins[-1], range_bins[-1], range_bins[0]],
)
plt.title("Range-Doppler Map (RDM)")
plt.xlabel("Doppler Bins")
plt.ylabel("Range Bins")
plt.colorbar(label="Amplitude")
# plt.colorbar(label="Amplitude (dB)")
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

    #! naaaaaah ? TODO: Simulate the impact of the single-target channel on the FMCW signal (the extension to a multi-target channel is obviously the sum of the target contributions)

    #! should work ? TODO: Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions

    #! should work TODO: RDM obtained at the output of the 2 dimensional FFT for multiple randomly generated scenarios. Identify the correct targets positions on the RDM.

    # * DONE Compute the range and Doppler resolutions and discuss the relevance of the radar parameters for the considered scenario.


if __name__ == "__main__":
    plot()
