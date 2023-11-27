"""
ELEC-H311 - Joint Communications and Radar simulation project
Lucas Placentino
Salman Houdaibi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft

"""
    Step 3: Radar performance analysis

    Signal received at antenna corrupted by additive white Gaussian noise (AWGN), then mixed with transmitted signal and its quadrature, low-pass filtered to limit noise to the ADC bandwith and samples. Add noise directly on the complex samples at the rate F_s. Signal to noise ratio (SNR) defined as the power of the complex signal divided by the power of the complex noise.
    Targets detected by applying threshold on the RDM, might get false alarms and mis-detections due to noise. Estimate their probabilities as a function of the threshold and the SNR. Receiver operating characteristic curve (ROC) obtained by displaying mis-detection prob as a function of the false alarm prob.

    To do:
    - Add white Gaussian noise on the complex samples at the receiver
    - Asses the probability of false alarm and the probability of mis-detection as a function of the threshold for different values of the SNR
    Expected outcomes:
    - Compare the RDM obtained with and without noise
    - Draw the ROC curves for different values of the SNR
    - Discuss the choice of the SNR values
"""

T_chirp_duration = 2e-4  # Durée du chirp
Number_of_samples = 2**18
B_freq_range = 200e6
f_c = 24e9  # carrier freq
F_sim_sampling_freq = 512e6  # simulation
F_radar_sampling_freq = 2e6  # receiver sampling freq
Beta_slope = B_freq_range / T_chirp_duration
t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)

# Transmitted signal
Tx = np.exp(1j * np.pi * Beta_slope * (t**2))

# Received signal
Rx = np.copy(Tx)  # no noise

# Additive white Gaussian noise (AWGN)
AGWN = np.random.normal(0, 1, Number_of_samples) + 1j * np.random.normal(
    0, 1, Number_of_samples
)  # complex noise, both real and imaginary parts are independant and are white noise
#! noise takes SNR in input ??
Rx_noise = Rx + AGWN  # received signal with noise

#! TODO: compute the SNR: power of the complex signal divided by the power of the complex noise ?
SNR_dB = 10  #! arbitrary (dB) TODO: compute it instead
SNR_lib = 10 ** (SNR_dB / 10)  # (linear)

# compute signal power: modulus squared (square of the amplitude), then mean? Module au carré = puissance
# because signal is ergodic, we can compute the mean over time instead of the ensemble mean

# noise white gaussian noise: mean = 0, variance = 1, power = variance = 1 or 2 ?


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, Tx.real, label="Tx Real")
plt.title("Transmitted signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# ------------ cannot see the no-noise signal on this plot, maybe seperate graphs?
plt.subplot(2, 1, 2)
plt.plot(t, Rx.real, label="Rx Real")
plt.plot(t, Rx_noise.real, label="Rx Real with noise")
plt.title("Received signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()


# Generate the Range Doppler Map (RDM)
Ns = 512  #! number of samples ?
Nd = 256  # number of ?

range_profile = sft.fft(Rx_noise, Ns)

doppler_profile = sft.fftshift(sft.fft(range_profile, Nd))  # ,axes=0 or nothing ?

plt.figure(figsize=(10, 6))
#! NOT WORKING, TODO: FIX ?
# plt.imshow(np.abs(doppler_profile), extent=[-Nd/2, Nd/2, 0, Ns], cmap="jet", aspect="auto")

doppler_bins = np.fft.fftshift(np.fft.fftfreq(Nd, T_s))
range_bins = np.arange(Ns) * (F_radar_sampling_freq / (2 * Ns))
D, R = np.meshgrid(doppler_bins, range_bins)
plt.pcolormesh(D, R, np.abs(doppler_profile), cmap='jet')

plt.title("Range Doppler Map")
plt.xlabel("Doppler (Hz)")
plt.ylabel("Range (m)")
# plt.colorbar(label="Amplitude") #TODO: remove comment when img plot fixed
# plt.tight_layout()
plt.show()

# False alarm probability

# we vary the threshold values
# threshold = 0.5 #! arbitrary value
threshold_values = np.linspace(0, 2, 100)  #! arbitrary values

# P_false_alarm = 0 # false alarm probability
P_false_alarm = np.zeros(len(threshold_values))  # false alarms probability

# for TODO: loop

plt.figure(figsize=(8, 6))
plt.plot(threshold_values, P_false_alarm)
plt.title("False alarm probability")
plt.xlabel("Threshold")
plt.ylabel("Probability")
plt.grid(True)
# plt.tight_layout()
plt.show()

# Mis-detection probability

SNR_values = np.linspace(0, 20, 100)  #! arbitrary values

# P_mis_detection = 0 # mis-detection probability
# P_mis_detection = np.zeros(len(threshold_values)) # mis-detections probability
P_mis_detection = np.zeros(len(SNR_values))  # mis-detections probability

# for TODO: loop


plt.figure(figsize=(8, 6))
plt.plot(SNR_values, P_mis_detection)
plt.title("Mis-detection Probability")
plt.xlabel("SNR (db)")
plt.ylabel("Probability")
plt.grid(True)
# plt.tight_layout()
plt.show()


# Receiver operating characteristic curve (ROC) for different values of the SNR (or threshold?)
#! for different values of the SNR

plt.figure(figsize=(8, 8))
plt.plot(P_false_alarm, P_mis_detection, label="ROC curve")
plt.title("Receiver Operating Characteristic Curve")
plt.xlabel("Probability of false alarm")
plt.ylabel("Probability of mis-detection")
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.show()

# Discuss the choice of the SNR values


# -------------------
# we applied the noise directly to the samples, because it is easier for us to do so in this project (we dont have to compute through a low pass filter etc)
