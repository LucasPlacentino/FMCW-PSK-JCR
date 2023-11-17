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

"""
    Step 2: Radar processing

    Generate a Range Doppler Map (RDM)
    Target range is max 20m, and speed is max 2m/s with respect to the radar
"""

fc_carrier_freq = 24e9  # f_c

B_freq_range = 200e6  # 200 MHz

Fs_radar_sampling_freq = 2e6  # 2 MHz

F_sim_sampling_freq = 512e6  # 512 MHz

N_fast_time_fft_size = 512

Number_guard_samples = 5

K_slow_time_fft_size = 256


# 1. Generate the FMCW signal composed of K chirps

chirp = 

K_chirps = K_slow_time_fft_size*






# 2. Simulate the impact of the single-target channel on the FMCW signal (the extension to a multi-target channel is obviously the sum of the target contributions)








# 3. Implement the radar processing: mixing with the transmitted signal, sampling at F_s, S/P conversion,FFT over the fast and slow time dimensions













