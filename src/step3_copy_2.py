import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft

import step1

DEBUG = False

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
user_number_of_targets = int(input("Enter number of targets (int), i.e 5 : "))
user_SNR = int(input("SNR in dB (negative for more noise) : ")) 

guard_samples = 5 
#kappa = 1 

number_of_simulation_samples_one_chirp = int(F_simulation_sampling_freq * T_chirp_duration) 
print("number_of_simulation_samples_one_chirp: ", number_of_simulation_samples_one_chirp)
number_of_simulation_samples_over_k_chirps = number_of_simulation_samples_one_chirp * K_slow_time_fft_size 

radar_max_range = (c * F_radar_sampling_freq) / (2 * Beta_slope)
print("radar theoretical maximum range ? : ", radar_max_range, "m")

single_chirp_signal = step1.signal_baseband  
print("single_chirp_signal shape:",single_chirp_signal.shape)
FMCW_over_K_chirps = np.tile(single_chirp_signal, K_slow_time_fft_size)
print("FMCW shape:",FMCW_over_K_chirps.shape)

def rdm_noise(number_of_targets, SNR, plot=False, in_db=True):
    print("Generating random targets...")
    target_scale = 95 
    target_delays = np.random.rand(number_of_targets) * tau_max
    target_velocities = np.random.uniform(-max_speed, max_speed, number_of_targets)

    if DEBUG:
        target_delays = np.concatenate((target_delays,[2*10/c, 2*19/c])) 
        target_velocities = np.concatenate((target_velocities,[1e-9, -1.9])) 
        number_of_targets += 2
        print("############### /!\\ DEBUG: debug target(s) added ##############")

    for i in range(number_of_targets):
        print("Target", i + 1, "- delay:", "{:.2f}".format(target_delays[i] * 1e9), "ns", end=", ",)
        print("velocity:", "{:.2f}".format(target_velocities[i]), "m/s")

    radar_wavelength = c / f_c

    def target_contribution(target_delay, target_velocity):
        R_0_initial_range = c * target_delay / 2 
        print("R_0_initial_range: ", R_0_initial_range, "m")
        doppler_freq = 2 * target_velocity * f_c / c * target_scale
        beat_frequency = 2 * R_0_initial_range * Beta_slope / c
        t_prime = np.arange(0, T_chirp_duration, T_chirp_duration / (N_fast_time_fft_size + guard_samples))

        kappa = np.exp(1j*4*np.pi*R_0_initial_range*f_c/c)*np.exp(1j*(-2)*np.pi* (Beta_slope**2) * (R_0_initial_range**2) / (c**2)) # ? complex factor

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
    print("signal_target shape",signal_target.shape)

    noise_power = 10**(-SNR/20) 
    AWGN = np.random.normal(0, noise_power, len(signal_target)) + 1j * np.random.normal(0, 1, len(signal_target)) 
    print("noise shape",AWGN.shape)

    signal_target_noise = signal_target + AWGN

    sampled_signal = signal_target_noise 

    sp_converted_signal = np.reshape(sampled_signal, (K_slow_time_fft_size, N_fast_time_fft_size)) 
    sp_converted_signal = np.transpose(sp_converted_signal)
    print("sp_converted_signal shape:",sp_converted_signal.shape,"(height, width), or (rows, columns)")

    fast_time_fft = np.fft.fft(sp_converted_signal, axis=0)  

    slow_time_fft = np.fft.fft(fast_time_fft, axis=1)  

    print(slow_time_fft.shape)

    range_estimation_resolution = c / (2 * B_freq_range)
    print("range_estimation_resolution: ", range_estimation_resolution)
    doppler_freq_estimation_resolution = 1 / (K_slow_time_fft_size * T_chirp_duration)
    print("doppler_freq_estimation_resolution: ", doppler_freq_estimation_resolution)

    slow_time_fft = np.flipud(slow_time_fft)
    slow_time_fft = np.fliplr(slow_time_fft)

    slow_time_fft_db = 20 * np.log10(np.abs(slow_time_fft)/np.max(np.abs(slow_time_fft)) + 1e-12)

    if plot:
        if in_db:
            plt.figure(figsize=(10, 6))
            plt.imshow(slow_time_fft_db, vmax=np.max(slow_time_fft_db), vmin=np.max(slow_time_fft_db)-20, aspect="auto", cmap="jet", extent=[-max_speed, max_speed, 0, max_range],)
            plt.title("Range-Doppler Map (RDM), SNR = " + str(SNR) + "dB")
            plt.xlabel("Target Speed (m/s)")
            plt.ylabel("Target Range (m)")
            plt.colorbar(label="Amplitude (dB)")
            plt.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.imshow(np.abs(slow_time_fft), aspect="auto", cmap="jet", extent=[-max_speed, max_speed, 0, max_range],)
            plt.title("Range-Doppler Map (RDM), SNR = " + str(SNR) + "dB")
            plt.xlabel("Target Speed (m/s)")
            plt.ylabel("Target Range (m)")
            plt.colorbar(label="Amplitude")
            plt.show()

    if not in_db:
        return slow_time_fft
    return slow_time_fft_db

def plot():
    rdm_noise(number_of_targets=user_number_of_targets, SNR=user_SNR, plot=True)

    rdm_noise(number_of_targets=user_number_of_targets, SNR=user_SNR, plot=True, in_db=False)

    errors_test_number_of_targets = 50
    SNR_values = np.linspace(-80, 0, 8) # -20, 20, 6
    steps = 800
    threshold_values = np.linspace(-20, 0, steps)
    threshold_values_lin = np.linspace(0, 470, steps)
    noise_test = 100

    def false_alarm_probability(fa_threshold_values, SNR_value):
        prob = np.empty((noise_test, len(fa_threshold_values)))
        for n in range(noise_test):
            P_false_alarm = np.zeros(len(fa_threshold_values))  

            rdm = np.abs(rdm_noise(errors_test_number_of_targets, SNR_value, plot=False, in_db=False))
            normalized_rdm = rdm / np.max(rdm)
            #rdm_lin = np.abs(rdm_noise(errors_test_number_of_targets, SNR_value, plot=False, in_db=False))
            #normalized_rdm = rdm_lin / np.max(rdm_lin)
            #fa_threshold_values_lin = fa_threshold_values

            number_of_values = normalized_rdm.shape[0] * normalized_rdm.shape[1]

            fa_threshold_values_lin = 10**(fa_threshold_values/20)

            for i, threshold in enumerate(fa_threshold_values_lin):
                #false_alarm = np.sum(normalized_rdm > threshold) 

                targets_found = np.sum(normalized_rdm > (threshold/max(fa_threshold_values_lin)))
                #print("targets_found",targets_found)

                number_false_alarm = max(targets_found - errors_test_number_of_targets, 0)

                #P_false_alarm[i] = false_alarm / number_of_values
                P_false_alarm[i] = number_false_alarm / (number_of_values-errors_test_number_of_targets)
            prob[n] = P_false_alarm

        mean_prob = np.mean(prob, axis=0)
        print("mean_prob fa:",mean_prob)

        return mean_prob

    #P_false_alarm = false_alarm_probability(threshold_values, SNR_value=user_SNR)
    P_false_alarm = false_alarm_probability(threshold_values_lin, SNR_value=user_SNR)

    plt.figure(figsize=(8, 6))
    plt.plot(threshold_values, P_false_alarm) 
    plt.title("False alarm probability")
    plt.xlabel("Threshold (dB)")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()

    def mis_detection_probability(md_threshold_values, SNR_value):
        prob = np.empty((noise_test, len(md_threshold_values)))
        for n in range(noise_test):
            P_mis_detection = np.zeros(len(md_threshold_values))  

            rdm = np.abs(rdm_noise(errors_test_number_of_targets, SNR_value, plot=False, in_db=False))
            normalized_rdm = rdm / np.max(rdm)
            #rdm_lin = np.abs(rdm_noise(errors_test_number_of_targets, SNR_value, plot=False, in_db=False))
            #normalized_rdm = rdm_lin / np.max(rdm_lin)
            #md_threshold_values_lin = md_threshold_values

            number_of_values = normalized_rdm.shape[0] * normalized_rdm.shape[1]

            md_threshold_values_lin = 10**(md_threshold_values/20) 

            for i, threshold in enumerate(md_threshold_values_lin):
                #mis_detection = np.sum(normalized_rdm < threshold)
                targets_found = np.sum(normalized_rdm > (threshold/max(md_threshold_values_lin)))
                #print("targets_found md",targets_found)

                #targets_found_prob = 1 - targets_found / errors_test_number_of_targets

                number_mis_detection = max(errors_test_number_of_targets - targets_found, 0)
                
                #P_mis_detection[i] = mis_detection / number_of_values
                P_mis_detection[i] = number_mis_detection / errors_test_number_of_targets
            prob[n] = P_mis_detection

        mean_prob = np.mean(prob, axis=0)
        
        return mean_prob

    #P_mis_detection = mis_detection_probability(threshold_values, SNR_value=user_SNR)
    P_mis_detection = mis_detection_probability(threshold_values_lin, SNR_value=user_SNR)

    plt.figure(figsize=(8, 6))
    plt.plot(threshold_values, P_mis_detection)
    plt.title("Mis-detection Probability")
    plt.xlabel("Threshold (dB)")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 8))
    for i, SNR in enumerate(SNR_values):
        snr_string = "{:.1f}dB".format(SNR) 
        plt.plot(false_alarm_probability(threshold_values, SNR), mis_detection_probability(threshold_values, SNR), label="ROC curve: SNR = "+snr_string)

    plt.title("Receiver Operating Characteristic Curves for different SNR values")
    plt.xlabel("Probability of false alarm")
    plt.ylabel("Probability of mis-detection")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("The \"choice\" of the SNR value would be to optimize the ROC curve, i.e. to have the lowest possible mis-detection probability for the lowest possible false alarm probability, depending on the application of the radar, this means having the highest SNR possible. In reality, we would need to eliminate noise as much as possible, and increase the signal power, to have a high SNR.")
if __name__ == "__main__":
    plot()