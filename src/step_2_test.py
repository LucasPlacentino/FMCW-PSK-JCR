import numpy as np
import matplotlib.pyplot as plt
# Let's adjust the code to ensure the RDM is generated correctly.

# Constants
c = 3e8  # Speed of light (m/s)
fc = 24e9  # Carrier frequency (Hz)
B = 200e6  # Bandwidth (Hz)
T_chirp = 5.5e-3  # Chirp duration (s)
Fs = 2e6  # Sampling frequency (Hz)
K = 256  # Number of chirps
N = 512  # Number of samples per chirp

# Generate time vector for one chirp
t = np.linspace(0, T_chirp, N, endpoint=False)

# Generate one FMCW chirp
slope = B / T_chirp
chirp = np.exp(1j * 2 * np.pi * (fc * t + (slope / 2) * t**2))

# Simulate target echo
target_range = 150  # Target range in meters (example value)
target_velocity = -20  # Target velocity in m/s (negative for approaching target)

# Calculate the time delay and Doppler shift for the target
tau = 2 * target_range / c
fd = 2 * target_velocity * fc / c

# Create the received signal for K chirps with the target echo
received_signal = np.zeros((K, N), dtype=complex)
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

# Adjust the RDM plot according to the range resolution
range_axis = np.linspace(0, c / (2 * B) * N, N) / 2
doppler_axis = np.linspace(-Fs / 2, Fs / 2, K)

# Plot the RDM
plt.figure(figsize=(12, 8))
plt.imshow(20 * np.log10(rdm / np.max(rdm)), cmap='jet', aspect='auto', 
           extent=[doppler_axis[0], doppler_axis[-1], range_axis[0], range_axis[-1]])
plt.colorbar(label='Normalized Amplitude (dB)')
plt.xlabel('Doppler Frequency (Hz)')
plt.ylabel('Range (m)')
plt.title('Range-Doppler Map (RDM)')
plt.show()
