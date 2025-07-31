# oresmen.py
"""
A module for generating Oresme numbers (harmonic series partial sums)
Oresme sayıları (harmonik seri kısmi toplamları) üretmek için bir modül.
Bu sürüm, hesaplamaları hızlandırmak için Numba kullanır.
"""
import os
import numba
import numpy as np
from functools import lru_cache
from fractions import Fraction
import math
from typing import List, Union, Generator, Tuple, Optional
import time
import logging
from enum import Enum, auto

# -----------------------------
# Logging Yapılandırması
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('harmonic_numba')
logger.propagate = False  # Yinelenen logları engelle

# Handler'ı yalnızca bir kez ekle
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -----------------------------
# Sabitler ve Enum'lar
# -----------------------------

class ApproximationMethod(Enum):
    """Harmonik sayı yaklaştırma yöntemleri"""
    EULER_MASCHERONI = auto()
    EULER_MACLAURIN = auto()
    ASYMPTOTIC = auto()

EULER_MASCHERONI = 0.57721566490153286060
EULER_MASCHERONI_FRACTION = Fraction(303847, 562250)

# -----------------------------
# Temel Fonksiyonlar
# -----------------------------

def oresme_sequence(n_terms: int, start: int = 1) -> List[float]:
    """Oresme dizisi: a_i = i / 2^i"""
    if n_terms <= 0:
        raise ValueError("Terim sayısı pozitif olmalıdır")
    return [i / (2 ** i) for i in range(start, start + n_terms)]

@lru_cache(maxsize=128)
def harmonic_numbers(n_terms: int, start_index: int = 1) -> Tuple[Fraction]:
    """Kesirli harmonik sayılar (önbellekli)"""
    if n_terms <= 0:
        raise ValueError("n_terms pozitif olmalıdır")
    if start_index <= 0:
        raise ValueError("start_index pozitif olmalıdır")

    sequence = []
    current_sum = Fraction(0)
    for i in range(start_index, start_index + n_terms):
        current_sum += Fraction(1, i)
        sequence.append(current_sum)
    return tuple(sequence)

# Bu fonksiyon Numba ile de hızlandırılabilir.
@numba.njit
def harmonic_number(n: int) -> float:
    """n-inci harmonik sayı (float, Numba ile hızlandırılmış)"""
    if n <= 0:
        # njit modunda istisna yükseltme sınırlıdır, ancak basit durumlar çalışır.
        raise ValueError("n pozitif olmalıdır")
    total = 0.0
    for k in range(1, n + 1):
        total += 1.0 / k
    return total

# -----------------------------
# Numba ile Optimize Edilmiş Fonksiyonlar
# -----------------------------

@numba.njit
def harmonic_number_numba(n: int) -> float:
    """JIT derlenmiş harmonik sayı fonksiyonu"""
    return np.sum(1.0 / np.arange(1, n + 1))

@numba.njit
def harmonic_numbers_numba(n: int) -> np.ndarray:
    """Numba ile hızlandırılmış harmonik sayılar"""
    return np.cumsum(1.0 / np.arange(1, n + 1))

def harmonic_generator_numba(n: int) -> Generator[float, None, None]:
    """Numba destekli harmonik sayı üreteci"""
    sums = harmonic_numbers_numba(n)
    for i in range(n):
        yield float(sums[i])

# -----------------------------
# Yaklaştırma Fonksiyonları
# -----------------------------

def harmonic_number_approx(
    n: int,
    method: ApproximationMethod = ApproximationMethod.EULER_MASCHERONI,
    k: int = 2
) -> float:
    """Yaklaşık harmonik sayı hesaplaması"""
    if n <= 0:
        raise ValueError("n pozitif olmalıdır")

    if method == ApproximationMethod.EULER_MASCHERONI:
        return math.log(n) + EULER_MASCHERONI + 1/(2*n) - 1/(12*n**2)
    elif method == ApproximationMethod.EULER_MACLAURIN:
        result = math.log(n) + EULER_MASCHERONI + 1/(2*n)
        for i in range(1, k+1):
            B = bernoulli_number(2*i)
            term = B / (2*i) * (1/n)**(2*i)
            result -= term
        return result
    elif method == ApproximationMethod.ASYMPTOTIC:
        return math.log(n) + EULER_MASCHERONI + 1/(2*n)
    else:
        raise ValueError("Bilinmeyen yaklaştırma yöntemi")

@lru_cache(maxsize=32)
def bernoulli_number(n: int) -> float:
    """Bernoulli sayıları (önbellekli)"""
    if n == 0: return 1.0
    elif n == 1: return -0.5
    elif n % 2 != 0: return 0.0
    else:
        from scipy.special import bernoulli
        return bernoulli(n)[n]

# -----------------------------
# Performans Analizi
# -----------------------------

def benchmark_harmonic(n: int, runs: int = 10) -> dict:
    """Farklı hesaplama yöntemlerini karşılaştırır"""
    results = {}

    # Isınma çağrısı
    _ = harmonic_number_numba(10)

    # Saf Python (Döngü Numba tarafından hızlandırıldı)
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number(n)
    results['pure_python_numba_loop'] = (time.perf_counter() - start)/runs

    # Numba (NumPy ile)
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number_numba(n)
    results['numba'] = (time.perf_counter() - start)/runs

    # Yaklaşık
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number_approx(n)
    results['approximate'] = (time.perf_counter() - start)/runs

    return results

def compare_with_approximation(n: int) -> dict:
    """Tam ve yaklaşık değerleri karşılaştırır"""
    exact = harmonic_number(n)
    approx = harmonic_number_approx(n)
    error = abs(exact - approx)
    relative_error = error / exact if exact != 0 else 0

    return {
        'exact': exact,
        'approximate': approx,
        'absolute_error': error,
        'relative_error': relative_error,
        'percentage_error': relative_error * 100
    }

# -----------------------------
# Görselleştirme Fonksiyonları
# -----------------------------

def plot_comparative_performance(max_n=50000, step=5000, runs=10):
    """Karşılaştırmalı performans analizi"""
    import matplotlib.pyplot as plt

    # Veri hazırlığı
    n_values = list(range(5000, max_n+1, step))
    results = {
        'python_loop': [],
        'numba': [],
        'approx': []
    }

    # Isınma çağrısı
    _ = harmonic_number_numba(100)

    for n in n_values:
        # Python döngü performansı
        py_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = harmonic_number(n)
            py_times.append(time.perf_counter() - start)

        # Numba performansı
        numba_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = harmonic_number_numba(n)
            numba_times.append(time.perf_counter() - start)

        # Yaklaşık metot
        approx_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = harmonic_number_approx(n)
            approx_times.append(time.perf_counter() - start)

        # Sonuçları milisaniye cinsinden sakla
        results['python_loop'].append(np.mean(py_times)*1000)
        results['numba'].append(np.mean(numba_times)*1000)
        results['approx'].append(np.mean(approx_times)*1000)

    # Çizim
    plt.figure(figsize=(12, 8))
    plt.plot(n_values, results['python_loop'], 'b-o', label='Saf Python Döngüsü (@njit)')
    plt.plot(n_values, results['numba'], 'r-s', label='Numba (NumPy ile)')
    plt.plot(n_values, results['approx'], 'g-^', label='Yaklaşık')

    plt.title('Hesaplama Yöntemlerinin Performans Karşılaştırması')
    plt.xlabel('n değeri')
    plt.ylabel('Süre (ms)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Detaylı veri çıktısı
    print("\nDetaylı Performans Verileri (milisaniye cinsinden):")
    print(f"{'n':>8} | {'Python (@njit)':>15} | {'Numba (NumPy)':>15} | {'Yaklaşık':>10} | {'Hızlanma':>10}")
    print("-" * 75)
    for i, n in enumerate(n_values):
        speedup = results['python_loop'][i] / results['numba'][i]
        print(f"{n:8} | {results['python_loop'][i]:15.3f} | "
              f"{results['numba'][i]:15.3f} | {results['approx'][i]:10.3f} | {speedup:9.2f}x")

# -----------------------------
# Gelişmiş Harmonik Yaklaştırmalar
# -----------------------------

@numba.njit
def harmonic_sum_approx_numba(n: np.ndarray,
                            method: int = 1,  # 0:EULER_MASCHERONI, 1:EULER_MACLAURIN
                            order: int = 4) -> np.ndarray:
    """
    Numba uyumlu optimize edilmiş harmonik yaklaştırma versiyonu.
    Not: JIT uyumluluğu için Enum yerine tamsayı bayrakları kullanır.
    """
    gamma = EULER_MASCHERONI
    log_n = np.log(n)
    inv_n = 1.0 / n

    # Temel terimler
    result = gamma + log_n

    if method >= 1:  # EULER_MASCHERONI 1/(2n) içerir
        result += 0.5 * inv_n

        if order >= 2:
            inv_n2 = inv_n * inv_n
            result -= inv_n2 / 12

            if order >= 4:
                inv_n4 = inv_n2 * inv_n2
                result += inv_n4 / 120

                if order >= 6:
                    inv_n6 = inv_n4 * inv_n2
                    result -= inv_n6 / 252

    return result

# -----------------------------
# Yakınsama Analizi Yardımcıları
# -----------------------------

def harmonic_convergence_analysis(n_values: np.ndarray) -> dict:
    """
    Verilen değerler için harmonik seri yakınsamasını analiz eder.
    
    Args:
        n_values: Analiz edilecek n değerleri dizisi.
    
    Returns:
        Sözlük:
        - exact_sums: Tam harmonik toplamlar
        - approx_sums: Yaklaşık toplamlar
        - errors: Mutlak hatalar
        - log_fit: Logaritmik uyum katsayıları
    """
    # -1, 0 tabanlı indeksleme için
    exact = harmonic_numbers_numba(n_values[-1])[n_values-1]
    approx = harmonic_sum_approx_numba(n_values.astype(float))

    return {
        'exact_sums': exact,
        'approx_sums': approx,
        'errors': np.abs(exact - approx),
        'log_fit': np.polyfit(np.log(n_values), exact, 1)  # a*ln(n) + b
    }


# -----------------------------
# Ana Program
# -----------------------------

def main():
    """Ana fonksiyon"""
    # Hesaplamalar
    logger.info("Oresme Dizisi (ilk 5 terim): %s", oresme_sequence(5))
    logger.info("Kesirli Harmonik Sayılar (H1-H3): %s", harmonic_numbers(3))
    logger.info("5. Harmonik Sayı: %.4f", harmonic_number(5))

    # Yaklaşık değerler
    logger.info("1000. Harmonik Sayı Yaklaştırmaları:")
    logger.info("Euler-Mascheroni: %.8f",
               harmonic_number_approx(1000, ApproximationMethod.EULER_MASCHERONI))
    logger.info("Asimptotik: %.8f",
               harmonic_number_approx(1000, ApproximationMethod.ASYMPTOTIC))

    # Numba hesaplamaları
    _ = harmonic_number_numba(10)  # Isınma
    logger.info("Numba ile Hızlandırılmış (H1-H5): %s", harmonic_numbers_numba(5))
    logger.info("Numba Üreteci (H1-H3): %s", list(harmonic_generator_numba(3)))

    # Performans testi
    n_test = 100000
    logger.info("Performans Testi (n=%d):", n_test)
    bench_results = benchmark_harmonic(n_test)
    for method, time_taken in bench_results.items():
        logger.info("%25s: %.6f s/run", method, time_taken)

    # Karşılaştırma
    logger.info("Tam/Yaklaşık Değer Karşılaştırması (H_100):")
    comparison = compare_with_approximation(100)
    for key, value in comparison.items():
        logger.info("%20s: %.10f", key, value)

if __name__ == "__main__":
    main()
    plot_comparative_performance()
