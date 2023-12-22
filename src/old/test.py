import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft
import step1

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
b = np.reshape(a, (2, 6))
c = np.transpose(b)
print(a)
print(b)
print(c)

T_chirp_duration = 2e-4  # seconds, chirp duration
B_freq_range = 200e6  # Hz
f_c = 24e9  # Hz, carrier frequency
F_simulation_sampling_freq = 512e6  # Hz
F_radar_sampling_freq = 2e6  # Hz
Beta_slope = B_freq_range / T_chirp_duration
N_fast_time_fft_size = 512
K_slow_time_fft_size = 256
c = 3e8  # m/s, speed of light
max_range = 20  # m, given
max_speed = 2  # m/s, given
tau_max = (2 * max_range) / c  # maximum possible delay
number_of_targets = int(input("Enter number of targets (int), e.i 5 : "))
guard_samples = 5 # given

number_of_simulation_samples_one_chirp = int(F_simulation_sampling_freq * T_chirp_duration)
print("number_of_simulation_samples_one_chirp: ", number_of_simulation_samples_one_chirp)
number_of_simulation_samples_over_k_chirps = number_of_simulation_samples_one_chirp * K_slow_time_fft_size

radar_max_range = (c * F_radar_sampling_freq) / (2 * Beta_slope)
print("radar theoretical maximum range ? : ", radar_max_range, "m")

single_chirp_signal = step1.signal_baseband
print("single_chirp_signal shape:", single_chirp_signal.shape)
FMCW_over_K_chirps = np.tile(single_chirp_signal, K_slow_time_fft_size)
print("FMCW shape:", FMCW_over_K_chirps.shape)

T_r = (T_chirp_duration - tau_max)
N_samples_per_chirp = (F_radar_sampling_freq * T_r)
print("N_samples_per_chirp: ", N_samples_per_chirp, "sould be 512?")

target_delays = (np.random.rand(number_of_targets) * tau_max)
target_velocities = (np.random.rand(number_of_targets) * max_speed)
for i in range(number_of_targets):
    print("Target", i + 1, "- delay:", "{:.2f}".format(target_delays[i] * 1e9), "ns", end=", ")
    print("velocity:", "{:.2f}".format(target_velocities[i]), "m/s")

radar_wavelength = c / f_c

def target_contribution(target_delay, target_velocity):
    R_0_initial_range = c * target_delay / 2
    doppler_freq = 2 * target_velocity * f_c / c
    beat_frequency = 2 * R_0_initial_range * Beta_slope / c
    kappa = np.exp(1j*4*np.pi*R_0_initial_range*f_c/c)*np.exp(1j*(-2)*np.pi* (Beta_slope**2) * (R_0_initial_range**2) / (c**2))
    t_prime = np.arange(0, T_chirp_duration, T_chirp_duration / (N_fast_time_fft_size + guard_samples))

    complex_conjugated_video_signal = np.array([], dtype=complex)
    for k in range(K_slow_time_fft_size):
        complex_conjugated_video_signal_one_sample = np.array([], dtype=complex)
        for n in range(N_fast_time_fft_size):
            sample = kappa * np.exp(1j * 2 * np.pi * beat_frequency * t_prime[n]) * np.exp(1j * 2 * np.pi * doppler_freq * k * T_chirp_duration)
            complex_conjugated_video_signal_one_sample = np.concatenate((complex_conjugated_video_signal_one_sample, [sample]), axis=0)
        complex_conjugated_video_signal = np.concatenate((complex_conjugated_video_signal, complex_conjugated_video_signal_one_sample))
    target_signal = complex_conjugated_video_signal

    return target_signal

signal_target = sum(target_contribution(target_delays[i], target_velocities[i]) for i in range(number_of_targets))
print("signal_target shape", signal_target.shape)

SNR = int(input("SNR (negative dB) : ")) # dB
noise_power = 10**(-SNR/20)
AWGN = np.random.normal(0, noise_power, len(signal_target)) + 1j * np.random.normal(0, 1, len(signal_target))
print("noise shape", AWGN.shape)

signal_target_noise = signal_target + AWGN

sampled_signal = signal_target_noise

sp_converted_signal = np.reshape(sampled_signal, (K_slow_time_fft_size, N_fast_time_fft_size))
sp_converted_signal = np.transpose(sp_converted_signal)
print("sp_converted_signal shape:", sp_converted_signal.shape, "(height, width), or (rows, columns)")

fast_time_fft = np.fft.fft(sp_converted_signal, axis=0)
slow_time_fft = np.fft.fft(fast_time_fft, axis=1)
print(slow_time_fft.shape)

range_estimation_resolution = c / (2 * B_freq_range)
doppler_freq_estimation_resolution = 1 / (K_slow_time_fft_size * T_chirp_duration)

def to_physical_units(start, index, resolution):
    return start + index * resolution

range_bins = to_physical_units(0, np.arange(N_fast_time_fft_size), range_estimation_resolution)
doppler_bins = to_physical_units(-K_slow_time_fft_size / 2, np.arange(K_slow_time_fft_size), doppler_freq_estimation_resolution)

plt.figure(figsize=(10, 6))
plt.imshow(np.abs(slow_time_fft), aspect="auto", cmap="jet", extent=[0, K_slow_time_fft_size, N_fast_time_fft_size, 0])
plt.title("Range-Doppler Map (RDM)")
plt.xlabel("Doppler Bins")
plt.ylabel("Range Bins")
plt.colorbar(label="Amplitude")
plt.show()

if __name__ == "__main__":
    plot()
