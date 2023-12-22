import numpy as np
import matplotlib.pyplot as plt

# Assume step1 is a module containing necessary functions and constants
import step1

# Configuration parameters
DEBUG = False
T_chirp_duration = 4e-4  # Chirp duration
B_freq_range = 200e6    # Frequency range
f_c = 24e9              # Carrier frequency
F_simulation_sampling_freq = 512e6  # Simulation sampling frequency
F_radar_sampling_freq = 2e6        # Radar sampling frequency
Beta_slope = B_freq_range / T_chirp_duration  # Frequency slope
N_fast_time_fft_size = 22  # FFT size for fast time
K_slow_time_fft_size = 24  # FFT size for slow time
c = 3e8  # Speed of light
max_range = 20  # Max range
max_speed = 2   # Max speed

tau_max = (2 * max_range) / c  
# User inputs
user_number_of_targets = int(input("Enter number of targets (int), i.e 5 : "))
user_SNR = int(input("SNR in dB (negative for more noise) : ")) 

# Signal processing parameters
guard_samples = 5  # Number of guard samples
kappa = 1  # Scaling factor

# Calculations for number of samples
number_of_simulation_samples_one_chirp = int(F_simulation_sampling_freq * T_chirp_duration)
number_of_simulation_samples_over_k_chirps = number_of_simulation_samples_one_chirp * K_slow_time_fft_size 

radar_max_range = (c * F_radar_sampling_freq) / (2 * Beta_slope)

# Assuming step1.signal_baseband is a predefined signal for a single chirp
single_chirp_signal = step1.signal_baseband
FMCW_over_K_chirps = np.tile(single_chirp_signal, K_slow_time_fft_size)

def rdm_noise(number_of_targets, SNR, plot=False, in_db=True):
    # Generating random targets with random delays and velocities
    target_delays = np.random.rand(number_of_targets) * tau_max
    target_velocities = np.random.uniform(-max_speed, max_speed, number_of_targets)

    # Create a time vector for signal generation
    t = np.arange(0, T_chirp_duration, 1 / F_simulation_sampling_freq)
    total_samples = len(t) * K_slow_time_fft_size

    # Initialize combined signal
    combined_signal = np.zeros(total_samples, dtype=complex)

    # Target contribution calculation
    for i in range(number_of_targets):
        R_0 = c * target_delays[i] / 2
        doppler_freq = 2 * target_velocities[i] * f_c / c
        beat_frequency = 2 * R_0 * Beta_slope / c
        signal = np.exp(1j * (2 * np.pi * beat_frequency * t + 2 * np.pi * doppler_freq * f_c * t))
        combined_signal += np.tile(signal, K_slow_time_fft_size)

    # Adding noise
    noise_power = 10 ** (-SNR / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.normal(size=combined_signal.shape) + 1j * np.random.normal(size=combined_signal.shape))
    noisy_signal = combined_signal + noise

    # Reshaping the signal into a 2D matrix for FFT
    signal_matrix = np.reshape(noisy_signal, (K_slow_time_fft_size, -1))
    range_fft = np.fft.fft(signal_matrix, axis=1)
    doppler_fft = np.fft.fft(range_fft, axis=0)
    RDM = np.abs(doppler_fft)

    # Normalization and conversion to dB
    RDM_normalized = RDM / np.max(RDM)
    if in_db:
        RDM_db = 20 * np.log10(RDM_normalized + 1e-12)

    # Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        if in_db:
            plt.imshow(RDM_db, aspect='auto', cmap='jet')
            plt.title(f"Range-Doppler Map in dB (SNR = {SNR})")
            plt.colorbar(label='Amplitude (dB)')
        else:
            plt.imshow(RDM_normalized, aspect='auto', cmap='jet')
            plt.title(f"Range-Doppler Map (SNR = {SNR})")
            plt.colorbar(label='Amplitude')
        plt.xlabel('Doppler')
        plt.ylabel('Range')
        plt.show()

    return RDM_db if in_db else RDM_normalized
def generate_roc_curves(SNR_range, num_points, num_targets):
    # Placeholder for false alarm and detection rates for each SNR value
    fa_rates = {snr: [] for snr in SNR_range}
    det_rates = {snr: [] for snr in SNR_range}

    for SNR in SNR_range:
        # Simulate the RDM and noise
        RDM = rdm_noise(num_targets, SNR, plot=False)  # This function needs to be defined as shown before

        # Generate threshold levels based on the RDM values
        thresholds = np.linspace(np.min(RDM), np.max(RDM), num_points)
        
        for thresh in thresholds:
            # Calculate the false alarm rate and detection rate for this threshold
            fa_rate = np.sum(RDM > thresh) / RDM.size  # False alarm rate
            det_rate = np.sum(RDM <= thresh) / num_targets  # Detection rate (assuming one target per bin)
            
            fa_rates[SNR].append(fa_rate)
            det_rates[SNR].append(1 - det_rate)  # 1 - misdetection rate

    # Plot the ROC curves
    plt.figure(figsize=(10, 8))
    for SNR in SNR_range:
        plt.plot(fa_rates[SNR], det_rates[SNR], label=f'SNR = {SNR} dB')

    plt.xlabel('False Alarm Rate')
    plt.ylabel('Detection Rate')
    plt.title('ROC Curves for Different SNR Values')
    plt.legend()
    plt.grid(True)
    plt.show()

SNR_values = np.linspace(-20, 20, 9)  # SNR values from -20 dB to 20 dB
generate_roc_curves(SNR_values, num_points=50, num_targets=1)

def plot():
    SNR_values = np.linspace(-20, 20, 5)
    threshold_values = np.linspace(-40, 0, 100)  # dB scale
    threshold_values_linear = 10 ** (threshold_values / 20)

    for SNR in SNR_values:
        P_fa = []  # Probability of false alarm
        P_md = []  # Probability of mis-detection

        RDM = rdm_noise(user_number_of_targets, SNR, plot=False, in_db=False)
        for threshold in threshold_values_linear:
            false_alarms = np.sum(RDM > threshold) / RDM.size
            P_fa.append(false_alarms)

            detections = np.sum(RDM <= threshold) / user_number_of_targets
            P_md.append(1 - detections)

        plt.plot(P_fa, P_md, label=f'SNR = {SNR} dB')

    plt.title('ROC Curves for Different SNR Values')
    plt.xlabel('False Alarm Probability')
    plt.ylabel('Mis-detection Probability')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot()  
