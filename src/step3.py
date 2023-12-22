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

# test:
#a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#b = np.reshape(a, (2, 6))
#c = np.transpose(b)
#print(a)
#print(b)
#print(c)

DEBUG = False

# Parameters
T_chirp_duration = 4e-4  # seconds, chirp duration: 0.1 ms to 0.4 ms
B_freq_range = 200e6  # Hz
f_c = 24e9  # Hz, carrier frequency
F_simulation_sampling_freq = 512e6  # Hz
F_radar_sampling_freq = 2e6  # Hz
Beta_slope = B_freq_range / T_chirp_duration
# t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)
N_fast_time_fft_size = 22 #512 #24 #22
K_slow_time_fft_size = 24# 256 #16 #24
c = 3e8  # m/s, speed of light
max_range = 20  # m, given
max_speed = 2  # m/s, given
tau_max = (2 * max_range) / c  # maximum possible delay
#number_of_targets = 5  # arbitrary # could be asked for input
user_number_of_targets = int(input("Enter number of targets (int), i.e 5 : "))
user_SNR = int(input("SNR in dB (negative for more noise) : ")) # dB

guard_samples = 5 # given
kappa = 1 # simplification, we assume no amplitude change from the radar transmission to the received target signal (no loss)

number_of_simulation_samples_one_chirp = int(F_simulation_sampling_freq * T_chirp_duration) # number of SIMULATION samples over 1 chirp
print("number_of_simulation_samples_one_chirp: ", number_of_simulation_samples_one_chirp)
number_of_simulation_samples_over_k_chirps = number_of_simulation_samples_one_chirp * K_slow_time_fft_size # number of SIMULATION samples over the K chirps

radar_max_range = (c * F_radar_sampling_freq) / (
    2 * Beta_slope
)  # ? according to https://wirelesspi.com/fmcw-radar-part-3-design-guidelines/
print("radar theoretical maximum range ? : ", radar_max_range, "m")

# --- 1. --- Generate the FMCW signal composed of K chirps

## Signal in baseband
# def s_baseband(t):
#    return np.exp(1j * np.pi * Beta_slope * (t**2))
#t_one_chirp = np.linspace(0, T_chirp_duration, number_of_simulation_samples_one_chirp, endpoint=True)
#print("t_one_chirp shape:",t_one_chirp.shape)
#t_over_k_chirps = np.linspace(0,T_chirp_duration*K_slow_time_fft_size,number_of_simulation_samples_over_k_chirps,endpoint=True) 
#print("t_over_k_chirps shape:",t_over_k_chirps.shape)
#signal_baseband_one_chirp = np.exp(1j * np.pi * Beta_slope * (t_one_chirp**2))
single_chirp_signal = step1.signal_baseband  # used with step1.t
#single_chirp_signal = signal_baseband_one_chirp  # used with t_over_k_chirps
print("single_chirp_signal shape:",single_chirp_signal.shape)
# FMCW_over_K_chirps = K_slow_time_fft_size * single_chirp
FMCW_over_K_chirps = np.tile(single_chirp_signal, K_slow_time_fft_size)
print("FMCW shape:",FMCW_over_K_chirps.shape)
#plt.figure(figsize=(10, 4))
#plt.plot(t_over_k_chirps, FMCW_over_K_chirps, label="FMCW signal")
#plt.xlim(0, t_over_k_chirps[-1])
#plt.title("FMCW over K chirps")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.legend()
#plt.show()


# FMCW_over_K_chirps = FMCW_over_K_chirps.reshape((K_slow_time_fft_size, -1)) # will do the S/P conversion later
# print("FMCW_over_K_chirps: ", FMCW_over_K_chirps)


#T_r = (
#    T_chirp_duration - tau_max
#)  # duration # not correct, changes over chirps and targets?
#N_samples_per_chirp = (
#    F_radar_sampling_freq * T_r
#)  # should be this one, see document #? useful?
#print("N_samples_per_chirp: ", N_samples_per_chirp, "sould be 512?")

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

def rdm_noise(number_of_targets, SNR, plot=False, in_db=True):
    print("Generating random targets...")
    target_scale = 95 #100
    target_delays = (
        np.random.rand(number_of_targets) * tau_max# *target_scale#
    )  # random delay for each target
    target_velocities = (
        np.random.uniform(-max_speed, max_speed, number_of_targets)#*target_scale#
    )  # random speed for each target

    #debug target:
    if DEBUG:
        target_delays = np.concatenate((target_delays,[2*10/c, 2*19/c])) # 10m, 19m
        target_velocities = np.concatenate((target_velocities,[1e-9, -1.9])) # 0m/s, -1.9m/s
        number_of_targets += 2
        print("############### /!\\ DEBUG: debug target(s) added ##############")

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

    # target_beat_frequency = 1
    # F_doppler_shift = 1
    radar_wavelength = c / f_c
    # velocity = (F_doppler_shift/2) * (radar_wavelength)
    # distance = (c*target_beat_frequency) / (2*Beta_slope)


    # ! FIXME: speeds ?
    def target_contribution(target_delay, target_velocity):#(target_range, target_velocity):  # , signal):
        #delay = (2 * target_range) / c
        #freq_shift = 2 * target_velocity * f_c / c

        R_0_initial_range = c * target_delay / 2 # from the linear movement simplification
        print("R_0_initial_range: ", R_0_initial_range, "m")
        doppler_freq = 2 * target_velocity * f_c / c * target_scale#
        beat_frequency = 2 * R_0_initial_range * Beta_slope / c
        #kappa = np.exp(1j*4*np.pi*R_0_initial_range*f_c/c)*np.exp(1j*(-2)*np.pi* (Beta_slope**2) * (R_0_initial_range**2) / (c**2)) # ? complex factor
        #kappa = 1 # see constants beginning of file
        t_prime = np.arange(0, T_chirp_duration, T_chirp_duration / (N_fast_time_fft_size + guard_samples)) #! "sampled time index t′ (the fast time index)" #? or t over 1 chirp $t' \in [0,T]$ ?
        #print("t_prime shape:",t_prime.shape)
        #print(t_prime)

        complex_conjugated_video_signal = np.array([], dtype=complex)
        #complex_conjugated_video_signal = np.concatenate([kappa * np.exp(1j * 2 * np.pi * beat_frequency * t_prime[k])*np.exp(1j * 2 * np.pi * doppler_freq * k * T_chirp_duration)] for k in range(K_slow_time_fft_size))
        for k in range(K_slow_time_fft_size):
            #complex_conjugated_video_signal_one_sample = np.array([], dtype=complex)
            #np.concatenate(complex_conjugated_video_signal_one_sample,[kappa * np.exp(1j * 2 * np.pi * beat_frequency * t_prime[n])*np.exp(1j * 2 * np.pi * doppler_freq * k * T_chirp_duration) for n in N_fast_time_fft_size])
            
            complex_conjugated_video_signal_one_sample = np.array([], dtype=complex)
            for n in range(N_fast_time_fft_size):
                sample = kappa * np.exp(1j * 2 * np.pi * beat_frequency * t_prime[n]) * np.exp(1j * 2 * np.pi * doppler_freq * k * T_chirp_duration)
                complex_conjugated_video_signal_one_sample = np.concatenate((complex_conjugated_video_signal_one_sample, [sample]), axis=0)
                #np.concatenate((complex_conjugated_video_signal_one_sample,[sample]), dtype=complex)
                #print(complex_conjugated_video_signal_one_sample)
                #print("vid signal one sample shape",k,complex_conjugated_video_signal_one_sample.shape)
            complex_conjugated_video_signal = np.concatenate((complex_conjugated_video_signal, complex_conjugated_video_signal_one_sample))
            #print("vid signal shape",k,complex_conjugated_video_signal.shape)
        #target_signal = np.exp(1j * np.pi * Beta_slope * (t_over_k_chirps**2))
        target_signal = complex_conjugated_video_signal

        return target_signal

    signal_target = sum(target_contribution(target_delays[i], target_velocities[i]) for i in range(number_of_targets))
    print("signal_target shape",signal_target.shape)


    # ---- add AWGN ----
    noise_power = 10**(-SNR/20) #! /10 ? /20 ?
    #noise = np.random.normal(0, noise_power, signal_target.shape)
    AWGN = np.random.normal(0, noise_power, len(signal_target)) + 1j * np.random.normal(0, 1, len(signal_target)) # complex noise, both real and imaginary parts are independant and are white noise
    print("noise shape",AWGN.shape)

    # Mix the target_signal with the noise:
    signal_target_noise = signal_target + AWGN


    # samples mixed_signal_st with N_samples_per_chirp samples per chirp:
    #total_samples_over_K_chirps = N_fast_time_fft_size * K_slow_time_fft_size
    #total_number_of_samples = #(T_chirp_duration*K_slow_time_fft_size) * 1/#F_radar_sampling_freq
    #sampling_interval = len(mixed_signal_st) / total_samples_over_K_chirps
    #sampled_signal_st = mixed_signal_st[:: int(sampling_interval)]  # Sample the signal
    #sampled_signal = mixed_signal[:: int(N_samples_per_chirp)]  # Sample the signal

    # Calculate the sampling interval
    #sampling_interval = int(len(signal_target) / total_number_of_samples)
    #sampled_signal = signal_target[::sampling_interval]
    sampled_signal = signal_target_noise #* signal is already sampled from the target_contribution function

    # sampled_signal_st = mixed_signal_st[::sampling_interval]#[: end_index_st + 1]
    ##print("sampled_signal_st: ", sampled_signal_st, "length: ", len(sampled_signal_st))
    #print("sampled_signal: ", sampled_signal, "length: ", len(sampled_signal))

    # S/P conversion
    sp_converted_signal = np.reshape(sampled_signal, (K_slow_time_fft_size, N_fast_time_fft_size)) # width, height, needs to be transposed :
    sp_converted_signal = np.transpose(sp_converted_signal)
    print("sp_converted_signal shape:",sp_converted_signal.shape,"(height, width), or (rows, columns)")

    # Fast time FFT
    fast_time_fft = np.fft.fft(sp_converted_signal, axis=0)  # axis=0 means columns

    # Slow time FFT
    slow_time_fft = np.fft.fft(fast_time_fft, axis=1)  # axis=1 means rows

    # debug:
    #print(slow_time_fft_mt.shape)
    #print(slow_time_fft_st.shape)
    print(slow_time_fft.shape)

    range_estimation_resolution = c / (2 * B_freq_range)
    print("range_estimation_resolution: ", range_estimation_resolution)
    doppler_freq_estimation_resolution = 1 / (K_slow_time_fft_size * T_chirp_duration)
    print("doppler_freq_estimation_resolution: ", doppler_freq_estimation_resolution)
    # ? not above ?

    #def to_physical_units(start, index, resolution):
    #    return start + index * resolution
    #
    #range_bins = to_physical_units(
    #    0, np.arange(N_fast_time_fft_size), range_estimation_resolution
    #)
    #doppler_bins = to_physical_units(
    #    -K_slow_time_fft_size / 2,
    #    np.arange(K_slow_time_fft_size),
    #    doppler_freq_estimation_resolution,
    #)

    #! flip up-down RDM for ease of visually reading range: ?
    #slow_time_fft = np.transpose(slow_time_fft)
    slow_time_fft = np.flipud(slow_time_fft)
    #! flip left-right RDM to have negative speeds on the left ?
    slow_time_fft = np.fliplr(slow_time_fft)

    ## in dB:
    # slow_time_fft_st_db = 20 * np.log10(np.abs(slow_time_fft_st) + 1e-12) / np.max(np.abs(slow_time_fft_st)) # + 1e-12 to avoid log(0) #? 20 * np.log(...) ?
    # FIXME: needs fixing ?
    slow_time_fft_db = 20 * np.log10(np.abs(slow_time_fft)/np.max(np.abs(slow_time_fft)) + 1e-12) #/ np.max(np.abs(slow_time_fft)) # + 1e-12 to avoid log(0)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(
            #np.abs(slow_time_fft_st),
            #np.abs(slow_time_fft),
            slow_time_fft_db, # TODO: ? in dB
            vmax=np.max(slow_time_fft_db),
            #vmin=0.8*np.mean(slow_time_fft_db),
            vmin=np.max(slow_time_fft_db)-20,

            aspect="auto",
            cmap="jet",
            # normal extent:
            ## extent=[0, slow_time_fft_st.shape[1], 0, slow_time_fft_st.shape[0]],
            ## extent=[-2, 2, 0, max_range],
            #extent=[0, K_slow_time_fft_size, N_fast_time_fft_size, 0],
            #extent=[doppler_bins[0], doppler_bins[-1], range_bins[-1], range_bins[0]],

            # flipped extent:
            # extent=[doppler_bins[-1], doppler_bins[0], range_bins[0], range_bins[-1]], #flipud
            #extent=[doppler_bins[0], doppler_bins[-1], range_bins[0], range_bins[-1]], #flipud and fliplr
            extent=[-max_speed, max_speed, 0, max_range], #fliplr
        )
        plt.title("Range-Doppler Map (RDM), SNR = " + str(SNR) + "dB")
        #plt.xlabel("Doppler Bins")
        plt.xlabel("Target Speed (m/s)")
        #plt.ylabel("Range Bins")
        plt.ylabel("Target Range (m)")
        #plt.colorbar(label="Amplitude")
        plt.colorbar(label="Amplitude (dB)")
        plt.show()

    if not in_db:
        return slow_time_fft
    return slow_time_fft_db



def plot():

    rdm_noise(
        number_of_targets=user_number_of_targets,
        SNR=user_SNR,
        plot=True
    )

    errors_test_number_of_targets = 1 # arbitrary
    SNR_values = np.linspace(-20, 5, 6)  # arbitrary values, yea,
    steps = 128 # arbitrary
    # we vary the threshold values
    threshold_values = np.linspace(-20, 0, steps)  # arbitrary values: yes see from RDM

    #! TODO: normalize RDM and use thresholds \in [0, 1] ?

    #! TODO: False alarm probability

    def false_alarm_probability(fa_threshold_values, SNR_value):
        # P_false_alarm = 0 # false alarm probability
        P_false_alarm = np.zeros(len(fa_threshold_values))  # false alarms probability

        # FIXME: ?
        rdm = np.abs(rdm_noise(errors_test_number_of_targets, SNR_value, plot=False, in_db=False))
        #print("DEBUG: max rdm fa",np.max(rdm),"min rdm fa",np.min(rdm))
        normalized_rdm = rdm / np.max(rdm) # normalize RDM
        #print("DEBUG: max normalized_rdm fa",np.max(normalized_rdm),"min normalized_rdm fa",np.min(normalized_rdm))
        #print("DEBUG: normalized_rdm shape",normalized_rdm.shape,"len:",len(normalized_rdm))

        number_of_values = normalized_rdm.shape[0] * normalized_rdm.shape[1]

        fa_threshold_values_lin = 10**(fa_threshold_values/20) # threshold in linear scale
        #print("DEBUG: fa_threshold_values_lin",fa_threshold_values_lin)

        for i, threshold in enumerate(fa_threshold_values_lin):
            #! FIXME: ?
            false_alarm = np.sum(normalized_rdm > threshold) # each sample of the signal is a statistics random variable
            #print("DEBUG: false_alarm",false_alarm)
            P_false_alarm[i] = false_alarm / number_of_values
            #print("DEBUG: P_false_alarm",P_false_alarm[i])

        return P_false_alarm

    P_false_alarm = false_alarm_probability(threshold_values, SNR_value=user_SNR)

    plt.figure(figsize=(8, 6))
    plt.plot(threshold_values, P_false_alarm) #? plot with threshold in dB ?
    plt.title("False alarm probability")
    plt.xlabel("Threshold (dB)")
    plt.ylabel("Probability")
    plt.grid(True)
    # plt.tight_layout()
    plt.show()


    #! TODO: Mis-detection probability

    #SNR_values = np.arange(0, 20, 2) # ?

    def mis_detection_probability(md_threshold_values, SNR_value):
        # P_mis_detection = 0 # mis-detection probability
        # P_mis_detection = np.zeros(len(threshold_values)) # mis-detections probability
        P_mis_detection = np.zeros(len(md_threshold_values))  # mis-detections probability

        # FIXME: ?
        rdm = np.abs(rdm_noise(errors_test_number_of_targets, SNR_value, plot=False, in_db=False))
        #print("DEBUG: max rdm md",np.max(rdm),"min rdm md",np.min(rdm))
        normalized_rdm = rdm / np.max(rdm) # normalize RDM
        #print("DEBUG: max normalized_rdm md",np.max(normalized_rdm),"min normalized_rdm md",np.min(normalized_rdm))
        #print("DEBUG: normalized_rdm shape",normalized_rdm.shape,"len:",len(normalized_rdm))

        number_of_values = normalized_rdm.shape[0] * normalized_rdm.shape[1]

        md_threshold_values_lin = 10**(md_threshold_values/20) # threshold in linear scale
        #print("DEBUG: md_threshold_values_lin",md_threshold_values_lin)

        for i, threshold in enumerate(md_threshold_values_lin):
            #! FIXME: ?
            mis_detection = np.sum(normalized_rdm < threshold)
            #print("DEBUG: mis_detection",mis_detection)
            P_mis_detection[i] = mis_detection / number_of_values
            #print("DEBUG: P_mis_detection",P_mis_detection[i])
        
        return P_mis_detection

    P_mis_detection = mis_detection_probability(threshold_values, SNR_value=user_SNR)

    plt.figure(figsize=(8, 6))
    plt.plot(threshold_values, P_mis_detection)
    plt.title("Mis-detection Probability")
    plt.xlabel("Threshold (dB)")
    plt.ylabel("Probability")
    plt.grid(True)
    # plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8, 8))
    #plt.plot(P_false_alarm, P_mis_detection, label="ROC curve")
    for i, SNR in enumerate(SNR_values):
        snr_string = "{:.1f}dB".format(SNR) # format with only 1 decimal
        plt.plot(false_alarm_probability(threshold_values, SNR), mis_detection_probability(threshold_values, SNR), label="ROC curve: SNR = "+snr_string)
    # TODO: add linear dotted line for reference ?
    #plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle="--")

    plt.title("Receiver Operating Characteristic Curves for different SNR values")
    plt.xlabel("Probability of false alarm")
    plt.ylabel("Probability of mis-detection")
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.show()


    # Discuss the choice of the SNR values
    print("The \"choice\" of the SNR value would be to optimize the ROC curve, i.e. to have the lowest possible mis-detection probability for the lowest possible false alarm probability, depending on the application of the radar, this means having the highest SNR possible. In reality, we would need to eliminate noise as much as possible, and increase the signal power, to have a high SNR.")


    # -------------------
    # we applied the noise directly to the the received signal (or its samples?), because it is easier for us to do so in this project (we dont have to compute through a low pass filter, etc)


    #! FIXME: TODO: - False alarm probability and Mis-detection probability (as a function of the SNR? or just the threshold?)

    #! almost done ? TODO: - ROC curves for different SNR values


if __name__ == "__main__":
    plot()