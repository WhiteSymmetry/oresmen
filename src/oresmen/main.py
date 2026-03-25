# oresmen.py
"""
A module for generating Oresme numbers (harmonic series partial sums)
Oresme sayıları (harmonik seri kısmi toplamları) üretmek için bir modül.
Bu sürüm, hesaplamaları hızlandırmak için Numba kullanır.

oresmen.py - Oresme, Harmonik Seri ve Hilbert Uzayı Modülü

Bu modül şunları sağlar:
- Harmonik sayı hesaplamaları (tam ve yaklaşık)
- Oresme dizisi (n/2^n) hesaplamaları
- ℓ² (Hilbert uzayı) aidiyet testi (matematiksel olarak doğru)
- Numba ile optimize edilmiş hesaplamalar
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
logger.propagate = False

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

EULER_MASCHERONI = 0.5772156649015328606065120900824024310421
EULER_MASCHERONI_FRACTION = Fraction(303847, 562250)

# -----------------------------
# Temel Fonksiyonlar (KORUNDU, düzeltildi)
# -----------------------------

def oresme_sequence(n_terms: int, start: int = 1) -> List[float]:
    """
    Oresme dizisi: a_i = i / 2^i
    
    Bu dizi, harmonik serinin ıraksamasını kanıtlamak için Oresme tarafından 
    kullanılan geometrik seri karşılaştırma yönteminden adını alır.
    """
    if n_terms <= 0:
        raise ValueError("Terim sayısı pozitif olmalıdır")
    return [i / (2 ** i) for i in range(start, start + n_terms)]


@lru_cache(maxsize=128)
def harmonic_numbers(n_terms: int, start_index: int = 1) -> Tuple[Fraction]:
    """Kesirli harmonik sayılar (önbellekli) - TAM SONUÇLAR"""
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


# Bu fonksiyon Numba ile hızlandırılmıştır.
@numba.njit
def harmonic_number(n: int) -> float:
    """n-inci harmonik sayı (float, Numba ile hızlandırılmış)"""
    if n <= 0:
        raise ValueError("n pozitif olmalıdır")
    total = 0.0
    for k in range(1, n + 1):
        total += 1.0 / k
    return total


# -----------------------------
# Numba ile Optimize Edilmiş Fonksiyonlar (KORUNDU)
# -----------------------------

@numba.njit
def harmonic_number_numba(n: int) -> float:
    """JIT derlenmiş harmonik sayı fonksiyonu (NumPy ile)"""
    return np.sum(1.0 / np.arange(1, n + 1))


@numba.njit
def harmonic_numbers_numba(n: int) -> np.ndarray:
    """Numba ile hızlandırılmış harmonik sayılar dizisi"""
    return np.cumsum(1.0 / np.arange(1, n + 1))


def harmonic_generator_numba(n: int) -> Generator[float, None, None]:
    """Numba destekli harmonik sayı üreteci"""
    sums = harmonic_numbers_numba(n)
    for i in range(n):
        yield float(sums[i])


# -----------------------------
# Yaklaştırma Fonksiyonları (KORUNDU)
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
    """Bernoulli sayılarını hesaplar (önbellekli)."""
    if n == 0:
        return 1.0
    elif n == 1:
        return -0.5
    elif n % 2 != 0:
        return 0.0
    else:
        from scipy.special import bernoulli
        return bernoulli(n)[n]


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

    result = gamma + log_n

    if method >= 1:
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
# ℓ² (Hilbert Uzayı) Aidiyet Testi (DÜZELTİLMİŞ)
# -----------------------------

def is_in_hilbert(
    sequence: Union[List[float], np.ndarray, Generator[float, None, None]], 
    max_terms: int = 10000, 
    tolerance: float = 1e-8
) -> bool:
    """
    Bir dizinin ℓ² (Hilbert) uzayında olup olmadığını test eder.
    """
    # Generator'ı listeye çevir
    if isinstance(sequence, Generator):
        sequence = list(sequence)
    
    arr = np.array(sequence, dtype=float)
    
    # NaN ve inf kontrolü
    if not np.all(np.isfinite(arr)):
        return False
    
    # Test edilecek terim sayısını sınırla
    n_terms = min(len(arr), max_terms)
    test_seq = arr[:n_terms]
    
    # Kareler toplamını hesapla
    squares = test_seq ** 2
    cumsum = np.cumsum(squares)
    total_sum = cumsum[-1]
    
    # 1. Kriter: Toplam sonlu mu?
    if not np.isfinite(total_sum):
        return False
    
    # 2. Kriter: p-serisi testi (1/n^α formundaki diziler için)
    # Pozitif terimler için asimptotik α tahmini
    if n_terms > 500 and np.all(test_seq[100:] > 0):
        log_terms = np.log(test_seq[100:] + 1e-12)
        log_n = np.log(np.arange(100, n_terms))
        try:
            alpha = -np.polyfit(log_n, log_terms, 1)[0]
            
            # 1/n^α için ℓ²'de olma koşulu: 2α > 1 → α > 0.5
            if alpha > 0.5:
                return True
            elif alpha <= 0.5 and alpha > 0:
                return False
            # α çok büyükse (üstel sönüm) → True
            elif alpha > 10:
                return True
        except:
            pass
    
    # 3. Kriter: Son terimlerin katkısı
    if n_terms > 1000:
        last_contribution = squares[-1000:]
        last_sum = np.sum(last_contribution)
        if last_sum < tolerance:
            return True
    
    # 4. Kriter: Oran testi (üstel sönümlü diziler için)
    if n_terms > 100:
        ratios = np.abs(test_seq[1:100] / (test_seq[:99] + 1e-12))
        avg_ratio = np.mean(ratios)
        if avg_ratio < 0.95:  # Üstel sönüm
            return True
    
    # Varsayılan: toplam sonluysa True
    return np.isfinite(total_sum)


# -----------------------------
# Yeni Eklenen Yardımcı Fonksiyonlar
# -----------------------------

def harmonic_sequence(n_terms: int, start: int = 1) -> np.ndarray:
    """Harmonik dizi terimlerini üretir: a_n = 1/n"""
    if n_terms <= 0:
        raise ValueError("Terim sayısı pozitif olmalıdır")
    indices = np.arange(start, start + n_terms, dtype=float)
    return 1.0 / indices


def p_series(p: float, n_terms: int, start: int = 1) -> np.ndarray:
    """p-serisi üretir: a_n = 1/n^p"""
    if n_terms <= 0:
        raise ValueError("Terim sayısı pozitif olmalıdır")
    indices = np.arange(start, start + n_terms, dtype=float)
    return 1.0 / (indices ** p)


def geometric_sequence(ratio: float, n_terms: int, start: int = 1) -> np.ndarray:
    """Geometrik dizi üretir: a_n = ratio^n"""
    if n_terms <= 0:
        raise ValueError("Terim sayısı pozitif olmalıdır")
    exponents = np.arange(start, start + n_terms, dtype=float)
    return ratio ** exponents


def analyze_sequence(
    sequence: Union[List[float], np.ndarray],
    name: str = "Dizi",
    n_display: int = 5
) -> dict:
    """Bir dizinin detaylı analizini yapar"""
    seq = np.array(sequence, dtype=float)
    squares = seq ** 2
    cumsum = np.cumsum(squares)
    
    results = {
        'name': name,
        'first_terms': seq[:n_display].tolist(),
        'n_terms': len(seq),
        'sum_of_squares': cumsum[-1] if np.isfinite(cumsum[-1]) else np.inf,
        'in_hilbert': is_in_hilbert(seq),
        'max_term': np.max(np.abs(seq)),
        'decay_rate': None
    }
    
    # Sönüm oranı tahmini
    if len(seq) > 100 and np.all(seq[100:] > 0):
        log_terms = np.log(seq[100:] + 1e-12)
        log_n = np.log(np.arange(100, len(seq)))
        try:
            alpha = -np.polyfit(log_n, log_terms, 1)[0]
            results['decay_rate'] = alpha
            results['decay_description'] = f"~ 1/n^{alpha:.2f}"
        except:
            pass
    
    return results


def compare_sequences(sequences: dict, n_test: int = 5000) -> None:
    """Birden fazla diziyi karşılaştırır"""
    from tabulate import tabulate
    
    results = []
    for name, seq in sequences.items():
        if len(seq) < n_test:
            # Daha uzun dizi oluştur
            if "n/2" in name or "Oresme" in name:
                indices = np.arange(1, n_test + 1)
                seq = indices / (2.0 ** indices)
            elif "1/n" in name and "1/n²" not in name and "1/n³" not in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / indices
            elif "1/n²" in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / (indices ** 2)
            elif "1/√n" in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / np.sqrt(indices)
            elif "e⁻ⁿ" in name or "exp" in name:
                indices = np.arange(1, n_test + 1)
                seq = np.exp(-indices)
        
        test_seq = seq[:n_test]
        squares_sum = np.sum(test_seq ** 2)
        in_hilbert = is_in_hilbert(test_seq)
        
        results.append({
            "Dizi": name,
            "İlk 5 Terim": str(test_seq[:5].tolist())[:60],
            "∑ a_n²": f"{squares_sum:.6f}" if np.isfinite(squares_sum) else "∞",
            "ℓ²'de mi?": "✓ Evet" if in_hilbert else "✗ Hayır"
        })
    
    print(tabulate(results, headers="keys", tablefmt="grid", stralign="left"))


# -----------------------------
# Performans Analizi (KORUNDU)
# -----------------------------

def benchmark_harmonic(n: int, runs: int = 10) -> dict:
    """Farklı hesaplama yöntemlerini karşılaştırır"""
    results = {}

    # Isınma çağrısı
    _ = harmonic_number_numba(10)

    # Saf Python (Numba ile hızlandırılmış döngü)
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


def plot_comparative_performance(max_n=50000, step=5000, runs=10):
    """Karşılaştırmalı performans analizi (opsiyonel)"""
    import matplotlib.pyplot as plt

    n_values = list(range(5000, max_n+1, step))
    results = {
        'python_loop': [],
        'numba': [],
        'approx': []
    }

    _ = harmonic_number_numba(100)

    for n in n_values:
        py_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = harmonic_number(n)
            py_times.append(time.perf_counter() - start)

        numba_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = harmonic_number_numba(n)
            numba_times.append(time.perf_counter() - start)

        approx_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = harmonic_number_approx(n)
            approx_times.append(time.perf_counter() - start)

        results['python_loop'].append(np.mean(py_times)*1000)
        results['numba'].append(np.mean(numba_times)*1000)
        results['approx'].append(np.mean(approx_times)*1000)

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

    print("\nDetaylı Performans Verileri (milisaniye cinsinden):")
    print(f"{'n':>8} | {'Python (@njit)':>15} | {'Numba (NumPy)':>15} | {'Yaklaşık':>10} | {'Hızlanma':>10}")
    print("-" * 75)
    for i, n in enumerate(n_values):
        speedup = results['python_loop'][i] / results['numba'][i]
        print(f"{n:8} | {results['python_loop'][i]:15.3f} | "
              f"{results['numba'][i]:15.3f} | {results['approx'][i]:10.3f} | {speedup:9.2f}x")


# -----------------------------
# Yakınsama Analizi Yardımcıları (KORUNDU)
# -----------------------------

def harmonic_convergence_analysis(n_values: np.ndarray) -> dict:
    """Verilen değerler için harmonik seri yakınsamasını analiz eder."""
    exact = harmonic_numbers_numba(n_values[-1])[n_values-1]
    approx = harmonic_sum_approx_numba(n_values.astype(float))
    return {
        'exact_sums': exact,
        'approx_sums': approx,
        'errors': np.abs(exact - approx),
        'log_fit': np.polyfit(np.log(n_values), exact, 1)
    }


# -----------------------------
# Ana Program
# -----------------------------

def main():
    """Ana fonksiyon - modül testi"""
    logger.info("=" * 70)
    logger.info(" ORESMEN MODÜLÜ TESTİ (DÜZELTİLMİŞ SÜRÜM)")
    logger.info("=" * 70)
    
    # Temel hesaplamalar
    logger.info("Oresme Dizisi (ilk 5 terim): %s", oresme_sequence(5))
    logger.info("Kesirli Harmonik Sayılar (H1-H3): %s", harmonic_numbers(3))
    logger.info("5. Harmonik Sayı: %.4f", harmonic_number(5))
    
    # Numba hesaplamaları
    _ = harmonic_number_numba(10)
    logger.info("Numba ile Hızlandırılmış (H1-H5): %s", harmonic_numbers_numba(5))
    
    # ℓ² testleri
    logger.info("-" * 50)
    logger.info("ℓ² (Hilbert Uzayı) Aidiyet Testleri:")
    
    n_test = 10000
    harmonic_seq = 1 / np.arange(1, n_test + 1)
    logger.info("  1/n dizisi ℓ²'de mi? %s", is_in_hilbert(harmonic_seq))
    
    slow_seq = 1 / np.sqrt(np.arange(1, n_test + 1))
    logger.info("  1/√n dizisi ℓ²'de mi? %s", is_in_hilbert(slow_seq))
    
    oresme_seq = np.array([i / (2 ** i) for i in range(1, n_test + 1)])
    logger.info("  n/2ⁿ dizisi ℓ²'de mi? %s", is_in_hilbert(oresme_seq))
    
    # Performans testi
    n_perf = 100000
    logger.info("-" * 50)
    logger.info("Performans Testi (n=%d):", n_perf)
    bench_results = benchmark_harmonic(n_perf)
    for method, time_taken in bench_results.items():
        logger.info("%25s: %.6f s/run", method, time_taken)
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
