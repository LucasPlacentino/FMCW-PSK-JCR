import numpy as np
import matplotlib.pyplot as plt

# Constants
T_chirp_duration = 4e-4
B_freq_range = 200e6
f_c = 24e9
F_simulation_sampling_freq = 512e6
F_radar_sampling_freq = 2e6
Beta_slope = B_freq_range / T_chirp_duration
N_fast_time_fft_size = 22
K_slow_time_fft_size = 24
c = 3e8
max_range = 20
max_speed = 2
tau_max = (2 * max_range) / c
guard_samples = 5
kappa = 1

# Assuming step1 provides a function to generate the baseband signal for a single chirp
import step1
def signal_baseband_placeholder():
    # This function should return the baseband signal for a single chirp
    # The actual implementation should be provided by the user
    return np.ones(int(F_simulation_sampling_freq * T_chirp_duration))

def generate_rdm(number_of_targets, SNR):
  # Generate baseband signal for single chirp
    single_chirp_signal = signal_baseband_placeholder()

    # Tile the chirp signal to simulate multiple chirps
    FMCW_over_K_chirps = np.tile(single_chirp_signal, (K_slow_time_fft_size, 1))

    # Calculate the noise power and generate noise
    noise_power = 10 ** (-SNR / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*FMCW_over_K_chirps.shape) + 1j * np.random.randn(*FMCW_over_K_chirps.shape))

    # Add noise to the chirp signal to simulate the RDM
    rdm = FMCW_over_K_chirps + noise

    # Convert the RDM to the frequency domain using FFT
    fast_time_fft = np.fft.fft(rdm, axis=0)
    doppler_fft = np.fft.fft(fast_time_fft, axis=1)

    # Convert the FFT results to dB scale
    rdm_dB = 20 * np.log10(np.abs(doppler_fft))

    return rdm_dB
def plot_roc_curves(SNR_values, thresholds_dB, number_of_targets):
    plt.figure(figsize=(10, 8))
    
    for SNR in SNR_values:
        rdm_dB = generate_rdm(number_of_targets, SNR)
        
        P_d = []  # Probability of detection
        P_fa = []  # Probability of false alarm
        
        for threshold_dB in thresholds_dB:
            # Convert threshold to linear scale based on RDM dB scale
            threshold_linear = 10 ** ((threshold_dB - np.max(rdm_dB)) / 20)
            
            # Calculate detections above threshold
            detections = rdm_dB > threshold_linear
            
            # Calculate false alarms and detections
            P_fa.append(np.sum(detections) / np.size(rdm_dB))
            P_d.append(np.sum(detections) / (number_of_targets * K_slow_time_fft_size))
        
        # Plot ROC curve for this SNR
        plt.plot(P_fa, P_d, label=f'ROC curve: SNR = {SNR}dB')
    
    # Plot diagonal line from (0,0) to (1,1) for reference
    plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
    plt.title('Receiver Operating Characteristic Curves for different SNR values')
    plt.xlabel('Probability of False Alarm')
    plt.ylabel('Probability of Detection')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Example of usage
SNR_values = np.linspace(-20, 20, 9)  # SNR values from -20 dB to 20 dB
thresholds_dB = np.linspace(-20, 0, 128)  # Thresholds from -20 dB to 0 dB
number_of_targets = 5  # Example number of targets

plot_roc_curves(SNR_values, thresholds_dB, number_of_targets)
