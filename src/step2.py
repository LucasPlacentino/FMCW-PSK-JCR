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
"""

fc_carrier_freq = 24e9  # f_c

B_freq_range = 200e6  # 200 MHz

Fs_radar_sampling_freq = 2e6  # 2 MHz

F_sim_sampling_freq = 512e6  # 512 MHz

N_fast_time_fft_size = 512

Number_guard_samples = 5

K_slow_time_fft_size = 256

# Radar processign is implemented and applied on a multi-target channel.
