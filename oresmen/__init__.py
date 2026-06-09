# oresmen/__init__.py
"""
Oresme, Harmonic Series and Hilbert Space Module
Oresme, Harmonik Seri ve Hilbert Uzayı Modülü

This module provides:
- Harmonic number calculations (exact fractions and floating point)
- Oresme sequence (n / 2^n) generation
- Hilbert space (ℓ²) membership tests (mathematically sound)
- Numba-accelerated computations for large‑scale work
- Sequence analysis and comparison utilities

Bu modül şunları sağlar:
- Harmonik sayı hesaplamaları (kesirli tam sonuçlar ve kayan noktalı)
- Oresme dizisi (n / 2^n) üretimi
- ℓ² (Hilbert uzayı) aidiyet testleri (matematiksel olarak doğru)
- Büyük ölçekli işlemler için Numba ile hızlandırılmış hesaplamalar
- Dizi analizi ve karşılaştırma yardımcıları
"""

# Projenizin versiyon numarasını belirtmek iyi bir pratiktir.
__version__ = "0.1.9"
__author__ = "Mehmet Keçeci <mkececi@yaani.com>"
__license__ = "AGPL 3.0-or-later"

# oresmen.py dosyasındaki ana sınıfları ve fonksiyonları buraya import et
from .oresmen import (
    # Temel Hesaplama Fonksiyonları
    harmonic_number,          # Numba ile optimize edilmiş en hızlı tekil float hesaplama
    harmonic_numbers_numba,   # Numba ile optimize edilmiş float dizisi hesaplama
    harmonic_numbers,         # Kesin sonuçlar için yavaş ama hassas Fraction tabanlı hesaplama
    oresme_sequence,          # Orijinal Oresme dizisi fonksiyonu
    harmonic_generator_numba, # Numba destekli üreteç
    is_in_hilbert,
    harmonic_sequence,
    p_series,
    geometric_sequence,
    plot_comparative_performance,
    _run_tests,
    main,

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
    "harmonic_sequence",
    "p_series",
    "geometric_sequence",

    # Yaklaşım Fonksiyonları
    "harmonic_number_approx",
    "harmonic_sum_approx_numba",

    # Yardımcı Sınıflar ve Sabitler
    "ApproximationMethod",
    "EULER_MASCHERONI",
    #test
    "plot_comparative_performance",
    "_run_tests",
    "main",
]
