import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

# Constants
T_chirp_duration = 4e-4
B_freq_range = 200e6
f_c = 24e9
F_simulation_sampling_freq = 512e6
F_radar_sampling_freq = 2e6
Beta_slope = B_freq_range / T_chirp_duration
N_fast_time_fft_size = 512  # Adjusted from 22 for FFT
K_slow_time_fft_size = 24
c = 3e8
max_range = 20
max_speed = 2
tau_max = (2 * max_range) / c
kappa = 1

number_of_simulation_samples_one_chirp = int(F_simulation_sampling_freq * T_chirp_duration)
number_of_simulation_samples_over_k_chirps = number_of_simulation_samples_one_chirp * K_slow_time_fft_size
radar_max_range = (c * F_radar_sampling_freq) / (2 * Beta_slope)
single_chirp_signal = chirp(np.linspace(0, T_chirp_duration, number_of_simulation_samples_one_chirp), f0=0, f1=B_freq_range, t1=T_chirp_duration, method='linear')
FMCW_over_K_chirps = np.tile(single_chirp_signal, K_slow_time_fft_size)
range_estimation_resolution = c / (2 * B_freq_range)
doppler_freq_estimation_resolution = 1 / (K_slow_time_fft_size * T_chirp_duration)

def rdm_noise(number_of_targets, SNR, plot=False, in_db=True):
    target_delays = np.random.rand(number_of_targets) * tau_max
    target_velocities = np.random.uniform(-max_speed, max_speed, number_of_targets)
    signal_target = np.zeros(number_of_simulation_samples_over_k_chirps, dtype=complex)

    for i in range(number_of_targets):
        R_0_initial_range = c * target_delays[i] / 2
        doppler_freq = 2 * target_velocities[i] * f_c / c
        beat_frequency = 2 * R_0_initial_range * Beta_slope / c
        t_prime = np.linspace(0, T_chirp_duration, N_fast_time_fft_size)
        complex_conjugated_video_signal = kappa * np.exp(1j * 2 * np.pi * (beat_frequency * t_prime + doppler_freq * np.arange(K_slow_time_fft_size)[:, None] * T_chirp_duration))
        signal_target += complex_conjugated_video_signal.flatten()

    noise_power = 10**(-SNR/20)
    AWGN = np.random.normal(0, noise_power, signal_target.shape) + 1j * np.random.normal(0, noise_power, signal_target.shape)
    signal_target_noise = signal_target + AWGN
    sampled_signal = signal_target_noise.reshape(N_fast_time_fft_size, K_slow_time_fft_size)
    fast_time_fft = np.fft.fft(sampled_signal, axis=0)
    slow_time_fft = np.fft.fft(fast_time_fft, axis=1)
    slow_time_fft = np.flipud(np.fliplr(slow_time_fft))

    if not in_db:
        return slow_time_fft
    slow_time_fft_db = 20 * np.log10(np.abs(slow_time_fft)/np.max(np.abs(slow_time_fft)) + 1e-12)
    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(slow_time_fft_db, extent=[-max_speed, max_speed, 0, max_range])
        plt.colorbar()
        plt.show()

    return slow_time_fft_db

def plot_roc(SNR_values, threshold_values, number_of_targets):
    for SNR in SNR_values:
        P_fa = []
        P_md = []
        for threshold in threshold_values:
            rdm_db = rdm_noise(number_of_targets, SNR, plot=False, in_db=True)
            detection_matrix = rdm_db > threshold
            false_alarms = np.sum(detection_matrix) / detection_matrix.size
            P_fa.append(false_alarms)
            P_md.append(1 - false_alarms)  # Assuming all detections are true positives
        plt.plot(P_fa, P_md, label=f'SNR = {SNR} dB')
    plt.xlabel('Probability of False Alarm')
    plt.ylabel('Probability of Detection')
    plt.legend()
    plt.grid()
    plt.title('ROC Curves for Different SNR Values')
    plt.show()

if __name__ == "__main__":
    SNR_values = np.linspace(-20, 20, 9)
    threshold_values = np.linspace(-20, 0, 128)
    user_number_of_targets = 5
    plot_roc(SNR_values, threshold_values, user_number_of_targets)
