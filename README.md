# 🔊 Signals and Systems: Audio Noise Reduction & Spectral Analysis

![Language](https://img.shields.io/badge/Language-Python_3-blue.svg)
![Domain](https://img.shields.io/badge/Domain-Signals_%26_Systems-orange.svg)
![Method](https://img.shields.io/badge/Method-STFT_%7C_Spectral_Subtraction-green.svg)
![Filter](https://img.shields.io/badge/Filter-Butterworth_Bandpass-purple.svg)

A Signal Processing application that performs dynamic noise reduction on WAV audio files using **Short-Time Fourier Transform (STFT)**, **Spectral Subtraction**, and **Butterworth Bandpass Filtering**. The framework quantitatively evaluates audio enhancement through **Signal-to-Noise Ratio (SNR)** and **Mean Squared Error (MSE)** metric benchmarks.

---

## 📌 Project Overview

Noise reduction in acoustic signals requires separating unwanted noise components from the target audio signal without distorting vocal frequencies or introducing musical noise (robotic artifacts).

This project processes audio datasets by combining frequency-domain spectral subtraction with time-domain digital filtering:
1. **Dynamic Spectral Subtraction (`STFT` / `ISTFT`):** Estimates the noise profile dynamically from low-energy quiet frames in the Short-Time Fourier Transform magnitude spectrum. It applies an over-subtraction factor ($\alpha = 1.2$) and a spectral floor ($\beta = 0.05$) to prevent robotic artifacts.
2. **Butterworth Bandpass Filter:** Applies a 4th-order Butterworth digital filter configured for the human vocal frequency band ($80 \text{ Hz} - 6000 \text{ Hz}$).
3. **Metric Evaluation:** Calculates pre- and post-filtering **MSE** and **SNR (dB)** improvements, generating time-domain and Fast Fourier Transform (FFT) spectrum comparison plots.

---

## 📊 Signal Processing Workflow

```text
  +------------------+      +--------------------+      +-----------------------+
  | Noisy WAV Audio  | ---> |   Mono & Normal    | ---> | STFT Spectrum Analysis|
  +------------------+      +--------------------+      +-----------+-----------+
                                                                    |
  +------------------+      +--------------------+                  v
  | Filtered Output  | <--- | 80-6000Hz Bandpass | <--- | Dynamic Noise Profile |
  |   WAV & Plots    |      | Butterworth Filter |      | Spectral Subtraction  |
  +------------------+      +--------------------+      +-----------------------+
```

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.8+**
* Required packages: `numpy`, `scipy`, `matplotlib`, `pandas`, `soundfile`

### Environment Verification & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/KorsanPanda/signals-and-systems-analysis.git](https://github.com/KorsanPanda/signals-and-systems-analysis.git)
   cd signals-and-systems-analysis
   ```

2. **Verify dependencies:**
   ```bash
   python test.py
   ```

3. **Execute Audio Processing Pipeline:**
   *Make sure your original and noisy audio files are located inside `SesDosyasi/original` and `SesDosyasi/noisy` directories.*
   ```bash
   python proje.py
   ```
   *Processes audio, generates filtered WAV outputs in `SesDosyasi/filtered`, logs metric outputs to `results.csv`, and exports FFT/Time comparison plots to `SesDosyasi/figures`.*

4. **Run Statistical Metrics & Histogram Analysis:**
   ```bash
   python analiz.py
   ```
   *Computes overall SNR improvement success rates, mean MSE values, and saves `snr_histogram.png`.*

---

## 📁 Directory Structure

```text
korsanpanda-signals-and-systems-analysis/
├── proje.py          # Main signal processing script (STFT, Spectral Subtraction, Bandpass Filter, MSE/SNR calculation)
├── analiz.py         # Results analysis & SNR improvement histogram plotter
├── test.py           # Dependency verification script
├── results.csv       # Benchmark results dataset (MSE and SNR metrics)
└── README.md         # Project documentation
```
