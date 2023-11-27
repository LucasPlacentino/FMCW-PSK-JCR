import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft

# Constants and Parameters
T_chirp_duration = 2e-4
Number_of_samples = 2**18
B_freq_range = 200e6
f_c = 24e9
F_sampling_freq = 512e6
Beta_slope = B_freq_range / T_chirp_duration
t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)

# SNR
SNR_dB = 10
SNR_lin = 10**(SNR_dB/10)

# Transmitted Signal
Tx = np.exp(1j * np.pi * Beta_slope * (t**2))

# Received Signal (No Noise)
Rx = np.copy(Tx)

# Additive White Gaussian Noise (AWGN)
noise_power = np.mean(np.abs(Tx)**2) / SNR_lin
AWGN = np.sqrt(noise_power/2) * (np.random.normal(0, 1, Number_of_samples) + 1j * np.random.normal(0, 1, Number_of_samples))
Rx_noise = Rx + AWGN

# Plotting Transmitted, Received (No Noise), and Received (With Noise) Signals
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, Tx.real, label="Tx Real")
plt.title("Transmitted Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, Rx.real, label="Rx Real (No Noise)")
plt.title("Received Signal Without Noise")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, Rx_noise.real, label="Rx Real (With Noise)")
plt.title("Received Signal With Noise")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

# Generate the Range Doppler Map (RDM)
Ns = 512
Nd = 256

range_profile = sft.fft(Rx_noise, Ns)
doppler_profile = sft.fftshift(sft.fft(range_profile, Nd))

plt.figure(figsize=(10, 6))
plt.imshow(np.abs(doppler_profile), extent=[-Nd/2, Nd/2, 0, Ns], cmap="jet", aspect="auto")
plt.title("Range Doppler Map")
plt.xlabel("Doppler (Hz)")
plt.ylabel("Range (m)")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()

# False Alarm and Mis-detection Probability Calculations
threshold_values = np.linspace(0, 2, 100)
P_false_alarm = np.zeros(len(threshold_values))

for i, threshold in enumerate(threshold_values):
    P_false_alarm[i] = np.mean(np.abs(doppler_profile) > threshold)

plt.figure(figsize=(8, 6))
plt.plot(threshold_values, P_false_alarm)
plt.title("False Alarm Probability")
plt.xlabel("Threshold")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# Mis-detection Probability
P_mis_detection = np.zeros(len(threshold_values))

for i, threshold in enumerate(threshold_values):
    P_mis_detection[i] = 1 - np.mean(np.abs(doppler_profile) > threshold)

plt.figure(figsize=(8, 6))
plt.plot(threshold_values, P_mis_detection)
plt.title("Mis-detection Probability")
plt.xlabel("Threshold")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# Receiver Operating Characteristic Curve (ROC)
plt.figure(figsize=(8, 8))
plt.plot(P_false_alarm, P_mis_detection, label="ROC Curve")
plt.title("Receiver Operating Characteristic Curve")
plt.xlabel("Probability of False Alarm")
plt.ylabel("Probability of Mis-detection")
plt.legend()
plt.grid(True)
plt.show()
