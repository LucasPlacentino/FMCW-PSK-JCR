# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft

# Importing a module named "step1" from the same directory
import step1

# Creating and manipulating numpy arrays
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
b = np.reshape(a, (2, 6))
c = np.transpose(b)

# Printing the arrays
print(a)
print(b)
print(c)

# Defining various parameters for the radar system
T_chirp_duration = 2e-4
Number_of_samples = 2**18
B_freq_range = 200e6
f_c = 24e9
F_simulation_sampling_freq = 512e6
F_radar_sampling_freq = 2e6
Beta_slope = B_freq_range / T_chirp_duration
N_fast_time_fft_size = 512
K_slow_time_fft_size = 256
c = 3e8
max_range = 1000
max_speed = 100
tau_max = (2 * max_range) / c

# Taking user input for the number of targets
number_of_targets = int(input("Enter number of targets (int), e.i 5 : "))

# Defining the guard samples
guard_samples = 5

# Calculating the radar theoretical maximum range
radar_max_range = (c * F_radar_sampling_freq) / (2 * Beta_slope)
print("radar theoretical maximum range ? : ", radar_max_range, "m")

# Generating the FMCW signal for each chirp
single_chirp_signal = step1.signal_baseband
FMCW_over_K_chirps = np.tile(single_chirp_signal, K_slow_time_fft_size)

# Generating the time vector for each chirp
t_over_k_chirps = np.linspace(
    0,
    T_chirp_duration * K_slow_time_fft_size,
    Number_of_samples * K_slow_time_fft_size,
    endpoint=True,
)

# Calculating the number of samples per chirp
T_r = T_chirp_duration - tau_max
N_samples_per_chirp = F_radar_sampling_freq * T_r
print("N_samples_per_chirp: ", N_samples_per_chirp, "sould be 512?")

# Defining the target delays and velocities
target_delays = [tau_max*0.2, tau_max]
target_velocities = [-max_speed*0.1, max_speed*0.1]

# Printing the target delays and velocities
for i in range(number_of_targets):
    print("Target", i + 1, "- delay:", "{:.2f}".format(target_delays[i] * 1e9), "ns", end=", ")
    print("velocity:", "{:.2f}".format(target_velocities[i]), "m/s")

# Function to calculate the target contribution to the received signal
def target_contribution(target_delay, target_velocity):
    R_0_initial_range = c * target_delay / 2
    doppler_freq = 2 * target_velocity * f_c / c
    beat_frequency = 2 * R_0_initial_range * Beta_slope / c
    kappa = np.exp(1j*4*np.pi*R_0_initial_range*f_c/c)*np.exp(1j*(-2)*np.pi* (Beta_slope**2) * (R_0_initial_range**2) / (c**2))
    t_prime = np.arange(0, T_chirp_duration, T_chirp_duration / (N_fast_time_fft_size + guard_samples))
    print("t_prime shape:",t_prime.shape)
    print(t_prime)

    complex_conjugated_video_signal = np.array([], dtype=complex)
    for k in range(K_slow_time_fft_size):
        complex_conjugated_video_signal_one_sample = np.array([], dtype=complex)
        for n in range(N_fast_time_fft_size):
            sample = kappa * np.exp(1j * 2 * np.pi * beat_frequency * t_prime[n]) * np.exp(1j * 2 * np.pi * doppler_freq * k * T_chirp_duration)
            complex_conjugated_video_signal_one_sample = np.concatenate((complex_conjugated_video_signal_one_sample, [sample]), axis=0)
        complex_conjugated_video_signal = np.concatenate((complex_conjugated_video_signal, complex_conjugated_video_signal_one_sample))
        print("vid signal shape",k,complex_conjugated_video_signal.shape)
    target_signal = complex_conjugated_video_signal

    return target_signal

# Calculating the total target contribution to the received signal
signal_target = sum(target_contribution(target_delays[i], target_velocities[i]) for i in range(number_of_targets))
print("signal_target shape",signal_target.shape)

# Calculating the total number of samples and the sampling interval
total_number_of_samples = (T_chirp_duration*K_slow_time_fft_size) * 1/F_radar_sampling_freq
sampling_interval = int(len(signal_target) / total_number_of_samples)
sampled_signal = signal_target
print("sampled_signal: ", sampled_signal, "length: ", len(sampled_signal))

# Converting the sampled signal into a 2D array
sp_converted_signal = np.reshape(sampled_signal, (K_slow_time_fft_size, N_fast_time_fft_size))
sp_converted_signal = np.transpose(sp_converted_signal)
print("sp_converted_signal shape:",sp_converted_signal.shape,"(height, width), or (rows, columns)")

# Performing fast time and slow time FFT
fast_time_fft = np.fft.fft(sp_converted_signal, axis=0)
slow_time_fft = np.fft.fft(fast_time_fft, axis=1)
print(slow_time_fft.shape)

# Plotting the Range-Doppler Map (RDM)
plt.figure(figsize=(10, 6))
plt.imshow(
    np.abs(slow_time_fft),
    aspect="auto",
    cmap="jet",
    extent=[0, K_slow_time_fft_size, N_fast_time_fft_size, 0],
)
plt.title("Range-Doppler Map (RDM)")
plt.xlabel("Doppler Bins")
plt.ylabel("Range Bins")
plt.colorbar(label="Amplitude")
plt.show()

# Printing the range and Doppler resolutions
print("Range resolution: ", c / (2 * B_freq_range), " m?")
print("Doppler resolution: ", 1 / (K_slow_time_fft_size * T_chirp_duration), " Hz?")

# Printing the relevance of radar parameters for the considered scenario
print("--- Relevance of radar parameters for the considered scenario: ---")
print("The frequency range B is the bandwidth of the transmitted signal. It directly determines the range resolution, inversely-proportional. For our case, it is set to 200 MHz.")
print("The chirp duration T_chirp_duration is the time duration of a chirp. It directly determines the Doppler frequency resolution, inversely-proportional. For our case, it is set between 0.1 and 0.4 ms.")
print("The number of chirps K_slow_time_fft_size is the number of chirps in a sequence. It directly determines the Doppler frequency resolution, inversely-proportional. For our case, it is set to 256.")

# Function to plot the Range-Doppler Map (RDM)
def plot():
    plt.show()

# Calling the plot function if the script is run directly
if __name__ == "__main__":
    plot()
