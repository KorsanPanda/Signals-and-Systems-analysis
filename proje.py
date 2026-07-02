import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, stft, istft
import scipy.io.wavfile as wavfile

# =========================
# KLASÖR YAPILANDIRMASI
# =========================
BASE_DIR = "SesDosyasi"
ORIGINAL_DIR = os.path.join(BASE_DIR, "original")
NOISY_DIR = os.path.join(BASE_DIR, "noisy")
FILTERED_DIR = os.path.join(BASE_DIR, "filtered")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FILTERED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def to_mono(audio):
    if len(audio.shape) > 1:
        return np.mean(audio, axis=1)
    return audio

def normalize_audio(audio):
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val

def calculate_mse(reference, test):
    min_len = min(len(reference), len(test))
    reference = reference[:min_len]
    test = test[:min_len]
    return np.mean((reference - test) ** 2)

def calculate_snr(reference, test):
    min_len = min(len(reference), len(test))
    reference = reference[:min_len]
    test = test[:min_len]

    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - test) ** 2)

    if noise_power == 0:
        return np.inf

    return 10 * np.log10(signal_power / noise_power)

# =========================
# AKILLI FİLTRELEME VE GÜRÜLTÜ AZALTMA
# =========================
def spectral_subtraction_dynamic(audio, fs, alpha=1.2, noise_ratio=0.1, beta=0.05):
    """
    STFT kullanarak gürültü profilini EN SESSİZ anlardan dinamik olarak çıkarır.
    Müzikal gürültüyü (robotik sesleri) önlemek için Spectral Floor (beta) kullanır.
    """
    # Zaman domeninden STFT ile frekans domenine geçiş
    f, t, Zxx = stft(audio, fs=fs, nperseg=1024, noverlap=512)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # 1. Her bir zaman çerçevesinin (frame) enerjisini hesapla
    frame_energy = np.sum(magnitude, axis=0)

    # 2. Enerjisi en düşük olan kısımları "sadece gürültü" kabul et
    num_noise_frames = max(1, int(noise_ratio * magnitude.shape[1]))
    lowest_energy_indices = np.argsort(frame_energy)[:num_noise_frames]

    # 3. Gürültü profilini sadece bu sessiz anların ortalamasıyla oluştur
    noise_profile = np.mean(magnitude[:, lowest_energy_indices], axis=1, keepdims=True)

    # Gürültü profilini tüm sinyalden çıkar (Daha hassas bir alpha=1.2 ile)
    subtracted_magnitude = magnitude - alpha * noise_profile
    
    # Sıfıra indirme! Gürültünün %5'ini (beta) taban olarak bırak
    clean_magnitude = np.maximum(subtracted_magnitude, beta * noise_profile)

    # Temizlenmiş frekansları tekrar zaman domenine çevir
    Zxx_clean = clean_magnitude * np.exp(1j * phase)
    _, clean_audio = istft(Zxx_clean, fs=fs)
    
    return clean_audio

def bandpass_filter(audio, fs, lowcut=80, highcut=6000, order=4):
    """
    İnsan sesini boğmamak için 80-6000Hz arası genişletilmiş Bant Geçiren Filtre.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio)

# =========================
# GÖRSELLEŞTİRME VE ANA PROGRAM
# =========================
def plot_fft(original, noisy, filtered, fs, filename):
    min_len = min(len(original), len(noisy), len(filtered))
    original = original[:min_len]
    noisy = noisy[:min_len]
    filtered = filtered[:min_len]

    freq = np.fft.rfftfreq(min_len, d=1 / fs)
    fft_original = np.abs(np.fft.rfft(original))
    fft_noisy = np.abs(np.fft.rfft(noisy))
    fft_filtered = np.abs(np.fft.rfft(filtered))

    plt.figure(figsize=(12, 6))
    plt.plot(freq, fft_noisy, label="Noisy", alpha=0.7)
    plt.plot(freq, fft_filtered, label="Filtered", alpha=0.8)
    plt.plot(freq, fft_original, label="Original", alpha=0.6)

    plt.title(f"FFT Frekans Spektrumu - {filename}")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Genlik")
    plt.xlim(0, fs / 2)
    plt.legend()
    plt.grid(True)

    figure_path = os.path.join(FIGURES_DIR, filename.replace(".wav", "_fft.png"))
    plt.savefig(figure_path, dpi=300)
    plt.close()

def plot_time_domain(original, noisy, filtered, fs, filename):
    min_len = min(len(original), len(noisy), len(filtered))
    original = original[:min_len]
    noisy = noisy[:min_len]
    filtered = filtered[:min_len]

    time = np.arange(min_len) / fs

    plt.figure(figsize=(12, 6))
    plt.plot(time, noisy, label="Noisy", alpha=0.5)
    plt.plot(time, filtered, label="Filtered", alpha=0.8)
    plt.plot(time, original, label="Original", alpha=0.5)

    plt.title(f"Zaman Domeni Karşılaştırması - {filename}")
    plt.xlabel("Zaman (saniye)")
    plt.ylabel("Genlik")
    plt.legend()
    plt.grid(True)

    figure_path = os.path.join(FIGURES_DIR, filename.replace(".wav", "_time.png"))
    plt.savefig(figure_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Çalışma klasörü:", os.getcwd())

    if not os.path.exists(ORIGINAL_DIR) or not os.path.exists(NOISY_DIR):
        print("HATA: original veya noisy klasörü bulunamadı.")
        exit()

    results = []
    files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith(".wav")])
    print("\nToplam işlenecek dosya sayısı:", len(files))

    for index, filename in enumerate(files):
        original_path = os.path.join(ORIGINAL_DIR, filename)
        noisy_path = os.path.join(NOISY_DIR, filename)

        if not os.path.exists(original_path):
            continue

        fs_original, original_audio = wavfile.read(original_path)
        fs_noisy, noisy_audio = wavfile.read(noisy_path)

        if fs_original != fs_noisy:
            continue

        fs = fs_original

        original_audio = normalize_audio(to_mono(original_audio))
        noisy_audio = normalize_audio(to_mono(noisy_audio))

        # --- YENİ FİLTRELEME AŞAMASI ---
        reduced_noise_audio = spectral_subtraction_dynamic(noisy_audio, fs, alpha=1.2, noise_ratio=0.1, beta=0.05)
        filtered_audio = bandpass_filter(reduced_noise_audio, fs, lowcut=80, highcut=6000, order=4)

        # --- METRİK HESAPLAMALARI ---
        # SNR ve MSE hesaplamaları sesi normalize ETMEDEN ÖNCE yapılmalıdır.
        mse_noisy = calculate_mse(original_audio, noisy_audio)
        mse_filtered = calculate_mse(original_audio, filtered_audio)

        snr_noisy = calculate_snr(original_audio, noisy_audio)
        snr_filtered = calculate_snr(original_audio, filtered_audio)

        improvement = snr_filtered - snr_noisy

        results.append({
            "Dosya": filename,
            "MSE_Noisy": mse_noisy,
            "MSE_Filtered": mse_filtered,
            "SNR_Noisy_dB": snr_noisy,
            "SNR_Filtered_dB": snr_filtered,
            "SNR_Improvement_dB": improvement
        })

        # Sesi diske kaydetmek ve grafik çizmek için normalize et
        filtered_audio_normalized = normalize_audio(filtered_audio)
        
        filtered_path = os.path.join(FILTERED_DIR, filename)
        wavfile.write(filtered_path, fs, filtered_audio_normalized)

        if index < 5:
            plot_fft(original_audio, noisy_audio, filtered_audio_normalized, fs, filename)
            plot_time_domain(original_audio, noisy_audio, filtered_audio_normalized, fs, filename)

        print(f"[{index+1}/{len(files)}] {filename} işlendi. SNR İyileşmesi: {improvement:.2f} dB")

    results_df = pd.DataFrame(results)
    results_path = os.path.join(BASE_DIR, "results.csv")
    results_df.to_csv(results_path, index=False)

    print("\nİşlem tamamlandı.")
    print(f"Ortalama SNR İyileşmesi: {results_df['SNR_Improvement_dB'].mean():.2f} dB")