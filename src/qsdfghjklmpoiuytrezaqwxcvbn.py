# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
import step1  # Je suppose que step1 est un module avec les fonctions nécessaires

# Constantes
c = 3e8  # Vitesse de la lumière en m/s

# Paramètres du radar
T_chirp_duration = 2e-4  # Durée du chirp en secondes
B_freq_range = 200e6  # Plage de fréquences en Hz
f_c = 24e9  # Fréquence porteuse en Hz
K_slow_time_fft_size = 256  # Nombre de chirps
N_fast_time_fft_size = 512  # Taille de la FFT pour le temps rapide

# Définition d'autres paramètres du radar
F_simulation_sampling_freq = 512e6
F_radar_sampling_freq = 2e6
Beta_slope = B_freq_range / T_chirp_duration
max_range = 1000
max_speed = 100
tau_max = (2 * max_range) / c

# Calcul de la résolution de portée et de vitesse
range_resolution = c / (2 * B_freq_range)  # Résolution de la portée en mètres
T_observation = K_slow_time_fft_size * T_chirp_duration  # Durée de la période d'observation
velocity_resolution = c / (2 * f_c * T_observation)  # Résolution de la vitesse en m/s

# Préparation du signal FMCW pour chaque chirp (À partir de step1)
single_chirp_signal = step1.signal_baseband
FMCW_over_K_chirps = np.tile(single_chirp_signal, K_slow_time_fft_size)

# Génération du vecteur temporel pour chaque chirp
t_over_k_chirps = np.linspace(
    0,
    T_chirp_duration * K_slow_time_fft_size,
    Number_of_samples * K_slow_time_fft_size,
    endpoint=True,
)

# Traitement du signal radar pour les cibles (Cela dépend de votre implémentation dans step1)
# ... [Votre code pour le traitement du signal radar] ...

# Transformation de Fourier pour la cartographie Range-Doppler
sp_converted_signal = np.reshape(signal_target, (K_slow_time_fft_size, N_fast_time_fft_size))
sp_converted_signal = np.transpose(sp_converted_signal)

fast_time_fft = np.fft.fft(sp_converted_signal, axis=0)
slow_time_fft = np.fft.fft(fast_time_fft, axis=1)

# Ajustement des axes de la RDM pour l'affichage
range_axis = np.linspace(0, range_resolution * N_fast_time_fft_size, N_fast_time_fft_size)
velocity_axis = np.linspace(-max_speed, max_speed, K_slow_time_fft_size)

# Affichage de la RDM avec les axes ajustés
plt.figure(figsize=(10, 6))
plt.imshow(
    np.abs(slow_time_fft),
    aspect="auto",
    cmap="jet",
    extent=[velocity_axis[0], velocity_axis[-1], range_axis[-1], range_axis[0]]
)
plt.title("Range-Doppler Map (RDM)")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Range (m)")
plt.colorbar(label="Amplitude")
plt.show()
