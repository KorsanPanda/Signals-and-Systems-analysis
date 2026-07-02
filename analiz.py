import pandas as pd
import matplotlib.pyplot as plt

# Sonuç dosyasını oku
df = pd.read_csv("SesDosyasi/results.csv")

# SNR iyileşme dağılımı grafiği
plt.figure(figsize=(10, 5))
plt.hist(df["SNR_Improvement_dB"], bins=30)

plt.title("SNR Improvement Distribution")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("Number of Files")
plt.grid(True)

# Grafiği kaydet
plt.savefig("snr_histogram.png", dpi=300)
plt.close()

print("SNR histogram grafiği snr_histogram.png olarak kaydedildi.")

# Özet bilgiler
total_files = len(df)
successful_files = len(df[df["SNR_Improvement_dB"] > 0])
failed_files = len(df[df["SNR_Improvement_dB"] <= 0])

success_rate = successful_files / total_files * 100

print("\n========== ANALİZ ÖZETİ ==========")
print("Toplam Dosya Sayısı:", total_files)
print("Başarılı Dosya Sayısı:", successful_files)
print("Başarısız Dosya Sayısı:", failed_files)
print("Başarı Oranı (%):", success_rate)

print("\nOrtalama MSE Noisy:", df["MSE_Noisy"].mean())
print("Ortalama MSE Filtered:", df["MSE_Filtered"].mean())
print("Ortalama SNR Noisy:", df["SNR_Noisy_dB"].mean())
print("Ortalama SNR Filtered:", df["SNR_Filtered_dB"].mean())
print("Ortalama SNR İyileşmesi:", df["SNR_Improvement_dB"].mean())

print("\nEn İyi Sonuç:")
print(df.loc[df["SNR_Improvement_dB"].idxmax()])

print("\nEn Kötü Sonuç:")
print(df.loc[df["SNR_Improvement_dB"].idxmin()])