# src/oresmen/__init__.py
"""
A module for generating Oresme numbers (harmonic series partial sums)
Oresme sayıları (harmonik seri kısmi toplamları) üretmek için bir modül.
Bu sürüm, hesaplamaları hızlandırmak için Numba kullanır.
"""

# Projenizin versiyon numarasını belirtmek iyi bir pratiktir.
__version__ = "0.1.1"

# main.py dosyasındaki ana sınıfları ve fonksiyonları buraya import et
from .main import (
    # Temel Hesaplama Fonksiyonları
    harmonic_number,          # Numba ile optimize edilmiş en hızlı tekil float hesaplama
    harmonic_numbers_numba,   # Numba ile optimize edilmiş float dizisi hesaplama
    harmonic_numbers,         # Kesin sonuçlar için yavaş ama hassas Fraction tabanlı hesaplama
    oresme_sequence,          # Orijinal Oresme dizisi fonksiyonu
    harmonic_generator_numba, # Numba destekli üreteç
    is_in_hilbert,

    # Yaklaşım (Approximation) Fonksiyonları
    harmonic_number_approx,      # Yaklaşık değer hesaplayan ana fonksiyon
    harmonic_sum_approx_numba,   # Numba ile optimize edilmiş yaklaşık değer hesaplama

    # Kullanıcıların ihtiyaç duyacağı yardımcılar
    ApproximationMethod,      # Yaklaşım metodunu seçmek için gereken Enum sınıfı
    EULER_MASCHERONI,         # Önemli bir matematiksel sabit
)

# __all__ listesi, "from oresmen import *" komutu kullanıldığında nelerin import edileceğini tanımlar.
# Bu, kütüphanenizin genel arayüzünü (public API) belirlemek için iyi bir pratiktir.
__all__ = [
    # Temel Hesaplama Fonksiyonları
    "harmonic_number",
    "harmonic_numbers_numba",
    "harmonic_numbers",
    "oresme_sequence",
    "harmonic_generator_numba",
    "is_in_hilbert",

    # Yaklaşım Fonksiyonları
    "harmonic_number_approx",
    "harmonic_sum_approx_numba",

    # Yardımcı Sınıflar ve Sabitler
    "ApproximationMethod",
    "EULER_MASCHERONI",
]
