import numpy as np
import matplotlib.pyplot as plt

# Define constants and parameters
c = 3e8  # Speed of light (m/s)
fc = 24e9  # Carrier frequency (Hz)
B = 200e6  # Bandwidth (Hz)
T_chirp = 5.5e-3  # Chirp duration (s)
Fs = 2e6  # Sampling frequency (Hz)
K = 256  # Number of chirps

# Derived parameters
samples_per_chirp = int(T_chirp * Fs)
t = np.linspace(0, T_chirp, samples_per_chirp, endpoint=False)

# Generate one FMCW chirp
slope = B / T_chirp
chirp = np.exp(1j * 2 * np.pi * slope * t**2 / 2)
# Re-executing the code with a target echo to simulate the presence of a target

# Create the received signal for K chirps with the target echo
# Assuming a specific target range and velocity
target_range = 150  # Target range in meters
target_velocity = -20  # Target velocity in m/s (negative for approaching target)

# Calculate the time delay and Doppler shift for the target
tau = 2 * target_range / c
fd = 2 * target_velocity * fc / c

# Reset the received signal
received_signal = np.zeros((K, samples_per_chirp), dtype=complex)

# Add the target echo to the received signal
for k in range(K):
    delay_samples = int(tau * Fs)
    doppler_phase_shift = np.exp(1j * 2 * np.pi * fd * k * T_chirp)
    received_signal[k, delay_samples:] = chirp[:-delay_samples] * doppler_phase_shift

# Radar signal processing (same as before)
mixed_signal = received_signal * np.conj(chirp)
range_fft = np.fft.fft(mixed_signal, axis=1)
range_fft = np.fft.fftshift(range_fft, axes=1)
doppler_fft = np.fft.fft(range_fft, axis=0)
doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
rdm = np.abs(doppler_fft)
# Calculate range and Doppler bins for RDM
N=512
range_bins = np.linspace(0, c * T_chirp * Fs / 2, N // 2)  # Range bins
doppler_bins = np.linspace(-Fs / 2, Fs / 2, K)  # Doppler bins

# Adjust the RDM plot
plt.figure(figsize=(12, 6))
plt.imshow(20 * np.log10(rdm / np.amax(rdm)), cmap='jet', aspect='auto',
           extent=[doppler_bins[0], doppler_bins[-1], range_bins[0], range_bins[-1]])
plt.colorbar(label='Normalized Amplitude (dB)')
plt.xlabel('Doppler Frequency (Hz)')
plt.ylabel('Range (m)')
plt.title('Range-Doppler Map (RDM)')
plt.show()
