"""
ELEC-H311 - Joint Communications and Radar simulation project
Lucas Placentino
Salman Houdaibi
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go

# import math
# import pandas as pd
import numpy as np
import scipy as sp
import step1

"""
    Step 2: Radar processing

    Generate a Range Doppler Map (RDM)
    Target range is max 20m, and speed is max 2m/s with respect to the radar
"""

T_chirp_duration = 2e-4  # chirp duration: 0.1 ms to 0.4 ms

fc_carrier_freq = 24e9  # f_c

B_freq_range = 200e6  # 200 MHz

Fs_radar_sampling_freq = 2e6  # 2 MHz

F_sim_sampling_freq = 512e6  # 512 MHz

N_fast_time_fft_size = 512

Number_guard_samples = 5

K_slow_time_fft_size = 256

def main():

    # --- 1. --- Generate the FMCW signal composed of K chirps

    single_chirp = step1.signal_baseband  # used with step1.t

    FMCW_over_K_chirps = K_slow_time_fft_size * single_chirp

    """
    plt.plot(step1.t*K_slow_time_fft_size, FMCW_over_K_chirps.real, label="Real")
    plt.plot(step1.t*K_slow_time_fft_size, FMCW_over_K_chirps.imag, label="Imaginary", linestyle="--")
    plt.grid()
    plt.title("FMCW signal over K chirps")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.show()
    """


    # --- 2. --- Simulate the impact of the single-target channel on the FMCW signal (the extension to a multi-target channel is obviously the sum of the target contributions)

    # single target: shifted signal (shifted in time by a constant delay in freq by a constant Doppler). Neglect amplitude changes due to losses over the distance.

    # multiple targets: multiple persons moving at different speeds and distances from the radar. The signal is the sum of the shifted signals of each target.


    # --- 3. --- Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions

    # mixing signals ? mixed signal sampled at F_s. S/P conversion: sampled signal from serial format to parallel format forming blocks of samples. Then 2D FFT (over the fast and slow time dimensions) of the parallel converted signal => generates the RDM.
    # amplitude peaks in the RDM are the targets.







if __name__ == "__main__":
    main()