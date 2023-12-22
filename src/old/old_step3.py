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
import step1

T_chirp_duration = 2e-4  # 0.1 to 0.4 ms
#Number_of_samples = 2**18
B_freq_range = 200e6  # 200 MHz
f_c = 24e9  # carrier freq, 24 GHz
F_sim_sampling_freq = 512e6  # simulation, 512 MHz
F_radar_sampling_freq = 2e6  # receiver sampling freq, 2 MHz
T_sampling_period = 1 / F_radar_sampling_freq
K_number_of_chirps = 256 # number of chirps
N_number_of_samples_per_chirp = 512 # number of samples per chirp
Beta_slope = B_freq_range / T_chirp_duration
t = np.linspace(0, T_chirp_duration*K_number_of_chirps, N_number_of_samples_per_chirp*K_number_of_chirps, endpoint=True)
t_one_chirp = np.linspace(0, T_chirp_duration, N_number_of_samples_per_chirp, endpoint=True)

max_range = 100  #! arbitrary, in m

# Transmitted signal
#Tx = np.tile(np.exp(1j * np.pi * Beta_slope * (t**2)), K_number_of_chirps)
#print("signal_baseband shape:",step1.signal_baseband.shape)
#Tx = np.tile(step1.signal_baseband, K_number_of_chirps)
#print("Tx shape:",Tx.shape)
#plt.figure(figsize=(10, 4))
#plt.plot(t, Tx.real, label="Tx Real")
#plt.xlim(0, t[-1])
#plt.title("Tx (Real)")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.legend()
#plt.show()

# Received signal
#Rx = np.copy(Tx)  # no noise
Rx = np.tile(np.exp(1j * np.pi * Beta_slope * (t_one_chirp**2)), K_number_of_chirps)  # no noise
print("Rx shape:",Rx.shape)

# Additive white Gaussian noise (AWGN)
# noise_power = np.mean(np.abs(Tx) ** 2) / SNR_lin  #! SNR value computed or arbitrary ?
SNR = int(input("SNR in dB (negative for more noise) : ")) # dB
noise_power = 10**(-SNR/20)
#AGWN = np.random.normal(0, 1, len(t)) + 1j * np.random.normal(0, 1, len(t)) # complex noise, both real and imaginary parts are independant and are white noise
AWGN = np.random.normal(0, noise_power, len(t)) + 1j * np.random.normal(0, 1, len(t))
print("AGWN shape:",AWGN.shape)
#! noise takes SNR in input ? -> through noise_power ?
Rx_noise = Rx + AWGN  # received signal with noise
print("Rx_noise shape:",Rx_noise.shape)

#! TODO: compute the SNR: power of the complex signal divided by the power of the complex noise ?
#SNR = np.abs(Rx) / np.abs(AWGN)  #! Rx or Tx ?

#SNR_dB = 10  #! arbitrary (dB) TODO: compute it instead
#SNR_lib = 10 ** (SNR_dB / 10)  # (linear)

# compute signal power: modulus squared (square of the amplitude), then mean? Module au carré = puissance
# because signal is ergodic, we can compute the mean over time instead of the ensemble mean
rx_power = np.mean(np.abs(Rx) ** 2)  #! Rx or Tx ?
rx_power_dB = 10 * np.log10(rx_power)
print("Signal power: "+str(rx_power)+" ("+str(rx_power_dB)+" dB)")

rx_noise_power = np.mean(np.abs(Rx_noise) ** 2)
rx_noise_power_dB = 10 * np.log10(rx_noise_power)
print("Noisy signal power: "+str(rx_noise_power)+" ("+str(rx_noise_power_dB)+" dB)")

#power_ratio = rx_power / rx_noise_power

# white gaussian noise: mean = 0, variance = 1, power = variance = 1 or 2 ??


plt.figure(figsize=(10, 4))
plt.plot(t, Rx_noise.real, label="Rx Real")
plt.plot(t, Rx.real, label="Rx Real")
plt.xlim(0, t[-1])
plt.title("Rx Real with and without noise (no target)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# maybe seperate graphs? :
plt.figure(figsize=(10, 4))
plt.plot(t, Rx_noise.real, label="Rx Real with noise (no target)")
plt.plot(t, Rx.real, label="Rx Real (no target)")
plt.xlim(0, t[-1]/20)
plt.title("Received signal with and without noise, SNR="+str(SNR)+"dB")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
#plt.tight_layout()
plt.show()


# Generate the Range Doppler Map (RDM)

Nr = 512  #! number of range cells / OR number of samples on each chirp ?
Nd = 256  # number of doppler cells / number of chirps in one sequence
#t_rdm = np.linspace(0, Nr * Nd, int(Nd * T_chirp_duration), endpoint=True)  # ? correct ? int() ? needed ?

doppler_frequencies = np.fft.fftshift(np.fft.fftfreq(Nd, 1 / F_radar_sampling_freq))
range_values = np.linspace(0, max_range, Nr)

# ------------------ do not do this, use above --------------------------------------
#rdm = np.zeros((Nr, Nd), dtype=complex)
#doppler_profile = np.zeros((Nr, Nd), dtype=complex)
#
#for i, r in enumerate(range_values):
#    # Additive white Gaussian noise (AWGN)
#    # noise_power = np.mean(np.abs(Tx) ** 2) / SNR_lin  #! SNR value computed or arbitrary ?
#    AGWN = np.random.normal(0, 1, Number_of_samples) + 1j * np.random.normal(
#        0, 1, Number_of_samples
#    )  # complex noise, both real and imaginary parts are independant and are white noise
#    #! noise takes SNR in input ? -> through noise_power ?
#    Rx_noise = Rx + AGWN  # received signal with noise
#    range_profile = sft.fft(Rx_noise, Nr)
#    doppler_profile = sft.fftshift(sft.fft(range_profile, Nd))  # ,axes=0 or nothing ?
#    rdm[i, :] = doppler_profile
#
# ----------------------------------------------------------------------------------

# range_bins = np.arange(Nr) * (F_radar_sampling_freq / (2 * Nr))
# doppler_bins = np.fft.fftshift(np.fft.fftfreq(Nd, T_sampling_period))

# plot the RDM

# plt.figure(figsize=(10, 6))
# D, R = np.meshgrid(doppler_bins, range_bins)
# plt.pcolormesh(D, R, np.abs(doppler_profile), cmap="jet")
# plt.show()

plt.figure(figsize=(10, 6))
# plt.imshow(np.abs(doppler_profile.reshape(1, -1)), extent=[-Nd / 2, Nd / 2, 0, Nr], cmap="jet", aspect="auto")  # , vmin=0, vmax=1) #? vmin and vmax ?
plt.imshow(
    np.abs(rdm),
    extent=[doppler_frequencies[0], doppler_frequencies[-1], 0, max_range],
    cmap="jet",
    aspect="auto",
)  # , vmin=0, vmax=1) #? vmin and vmax ?

# doppler_bins = np.fft.fftshift(np.fft.fftfreq(Nd, T_sampling_period))
# range_bins = np.arange(Ns) * (F_radar_sampling_freq / (2 * Nr))
# D, R = np.meshgrid(doppler_bins, range_bins)
# plt.pcolormesh(D, R, np.abs(doppler_profile), cmap='jet')

plt.title("Range Doppler Map")
plt.xlabel("Doppler (Hz)")
plt.ylabel("Range (m)")
plt.colorbar(label="Amplitude")  #? what is Amplitude ? should we put it in dB ?
# plt.tight_layout()
plt.show()

#! TODO: compare the RDM obtained with and without noise

# False alarm probability

# we vary the threshold values
max_threshold = 2000  #! arbitrary value: yes see rfom RDM
threshold_values = np.linspace(0, max_threshold, 100)  #! arbitrary values: yes see from RDM

def false_alarm_probability(threshold_values):
    # P_false_alarm = 0 # false alarm probability
    P_false_alarm = np.zeros(len(threshold_values))  # false alarms probability

    for i, threshold in enumerate(threshold_values):
        false_alarm = np.sum(np.abs(doppler_profile) > threshold)
        P_false_alarm[i] = false_alarm / len(doppler_profile)

    return P_false_alarm

P_false_alarm = false_alarm_probability(threshold_values)

plt.figure(figsize=(8, 6))
plt.plot(threshold_values, P_false_alarm) #? plot with threshold in dB ?
plt.title("False alarm probability")
plt.xlabel("Threshold")
plt.ylabel("Probability")
plt.grid(True)
# plt.tight_layout()
plt.show()

# Mis-detection probability

SNR_values = np.linspace(0, 10, 100)  #! arbitrary values, yea, 
#SNR_values = np.arange(0, 20, 2) # ?

def mis_detection_probability(SNR_values):
    # P_mis_detection = 0 # mis-detection probability
    # P_mis_detection = np.zeros(len(threshold_values)) # mis-detections probability
    P_mis_detection = np.zeros(len(SNR_values))  # mis-detections probability

    mis_detection_threshold = 1200  #? = threshold_values[-1]

    for i, SNR in enumerate(SNR_values):
        # Additive white Gaussian noise (AWGN)
        # noise_power = np.mean(np.abs(Tx) ** 2) / SNR_lin  #! SNR value computed or arbitrary ?
        AGWN = np.random.normal(
            0, np.sqrt(1 / 10 ** (SNR / 10)), Number_of_samples
        ) + 1j * np.random.normal(
            0, np.sqrt(1 / 10 ** (SNR / 10)), Number_of_samples
        )  # complex noise, both real and imaginary parts are independant and are white noise
        #! noise takes SNR in input ? -> through noise_power ?
        Rx_noise = Rx + AGWN  # received signal with noise
        range_profile = sft.fft(Rx_noise, Nr)
        doppler_profile = sft.fftshift(sft.fft(range_profile, Nd))  # ,axes=0 or nothing ?

        mis_detection = np.sum(np.abs(doppler_profile) < mis_detection_threshold)

        P_mis_detection[i] = mis_detection / len(doppler_profile)
    return P_mis_detection

P_mis_detection = mis_detection_probability(SNR_values)

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
#for SNR in SNR_values:
#    plt.plot(false_alarm_probability(SNR), mis_detection_probability(SNR), label="ROC curve - SNR: "+str(SNR))
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


#! would be interesting to compute the maximum detection range for a given SNR
