"""
ELEC-H311 - Joint Communications and Radar simulation project
Lucas Placentino
Salman Houdaibi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft

"""
    Step 2: PSK Modulation of FMCW Signal
"""

# Paramètres existants
T_chirp_duration = 2e-4  # Durée du chirp
Number_of_samples = 2**18
B_freq_range = 200e6
f_c = 24e9
F_sampling_freq = 512e6
Beta_slope = B_freq_range / T_chirp_duration
t = np.linspace(0, T_chirp_duration, Number_of_samples, endpoint=True)


# Fonction du signal de baseband
def s_baseband(t):
    return np.exp(1j * np.pi * Beta_slope * (t**2))


# Génération de la séquence PSK (exemple avec BPSK)
N_symbols = 100  # Nombre de symboles PSK
symbols = np.random.choice([-1, 1], N_symbols)  # Séquence BPSK
symbol_rate = 1 / (T_chirp_duration / N_symbols)  # Taux de symboles
samples_per_symbol = int(F_sampling_freq / symbol_rate)  # Échantillons par symbole
psk_sequence = np.repeat(symbols, samples_per_symbol)

# Redimensionner la séquence PSK pour qu'elle corresponde à la durée du chirp
psk_sequence = np.pad(psk_sequence, (0, Number_of_samples - len(psk_sequence)), "constant")

# Modulation PSK du signal FMCW
signal_baseband = s_baseband(t)
signal_modulated = signal_baseband * psk_sequence

# Transformation de Fourier
fft_signal_modulated = sft.fft(signal_modulated)
fft_freq = sft.fftfreq(Number_of_samples, d=1 / F_sampling_freq)
fft_shifted_modulated = sft.fftshift(fft_signal_modulated)
fft_freq_shifted = sft.fftshift(fft_freq)


def main():
    # Affichage
    plt.figure(figsize=(12, 10))

    # Signal FMCW modulé en PSK
    plt.subplot(3, 1, 1)
    plt.plot(t, signal_modulated.real, label="Real")
    plt.plot(t, signal_modulated.imag, label="Imaginary", linestyle="--")
    plt.grid()
    plt.title("PSK Modulated FMCW Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Spectre du signal FMCW modulé
    plt.subplot(3, 1, 2)
    plt.plot(fft_freq_shifted, np.abs(fft_shifted_modulated))
    plt.grid()
    plt.title("Spectrum of PSK Modulated FMCW Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
