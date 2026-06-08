# oresmen.py
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


import numpy as np
# NumPy 2.5.0rc1 uyumluluğu için row_stack takma adı
if not hasattr(np, 'row_stack'):
    np.row_stack = np.vstack

import os
import math
import time
import logging
from enum import Enum, auto
from functools import lru_cache
from fractions import Fraction
from typing import List, Union, Generator, Tuple, Optional

import numba
import numpy as np

# -----------------------------
# Logging Configuration / Loglama Yapılandırması
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('harmonic_numba')
logger.propagate = False

# -----------------------------
# Constants and Enums / Sabitler ve Enum'lar
# -----------------------------
class ApproximationMethod(Enum):
    """Harmonic number approximation methods / Harmonik sayı yaklaştırma yöntemleri"""
    EULER_MASCHERONI = auto()
    EULER_MACLAURIN = auto()
    ASYMPTOTIC = auto()

EULER_MASCHERONI = 0.5772156649015328606065120900824024310421
EULER_MASCHERONI_FRACTION = Fraction(303847, 562250)

# -----------------------------
# Optional dependency handling / Opsiyonel bağımlılık yönetimi
# -----------------------------
try:
    from scipy.special import bernoulli as _scipy_bernoulli
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    logger.info("scipy bulunamadı, Bernoulli sayıları için yedek yöntem kullanılacak.")

# -----------------------------
# Basic Functions / Temel Fonksiyonlar
# -----------------------------

def oresme_sequence(n_terms: int, start: int = 1) -> List[float]:
    """
    Generate the Oresme sequence: a_i = i / 2^i
    Oresme dizisi: a_i = i / 2^i
    """
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    return [i / (2 ** i) for i in range(start, start + n_terms)]


@lru_cache(maxsize=128)
def harmonic_numbers(n_terms: int, start_index: int = 1) -> Tuple[Fraction]:
    """
    Exact fractional harmonic numbers (cached) / Kesirli tam harmonik sayılar (önbellekli)
    """
    if n_terms <= 0:
        raise ValueError("n_terms must be positive / n_terms pozitif olmalıdır")
    if start_index <= 0:
        raise ValueError("start_index must be positive / start_index pozitif olmalıdır")

    sequence = []
    current_sum = Fraction(0)
    for i in range(start_index, start_index + n_terms):
        current_sum += Fraction(1, i)
        sequence.append(current_sum)
    return tuple(sequence)


@numba.njit
def harmonic_number(n: int) -> float:
    """
    n-th harmonic number (float, Numba JIT) / n-inci harmonik sayı (float, Numba JIT)
    """
    if n <= 0:
        raise ValueError("n must be positive / n pozitif olmalıdır")
    total = 0.0
    for k in range(1, n + 1):
        total += 1.0 / k
    return total


# -----------------------------
# Numba Optimized Functions / Numba ile Optimize Edilmiş Fonksiyonlar
# -----------------------------

@numba.njit
def harmonic_number_numba(n: int) -> float:
    """JIT compiled harmonic number using NumPy / NumPy ile JIT derlenmiş harmonik sayı"""
    return np.sum(1.0 / np.arange(1, n + 1))


@numba.njit
def harmonic_numbers_numba(n: int) -> np.ndarray:
    """Array of harmonic numbers H_1 … H_n / H_1 … H_n harmonik sayılar dizisi"""
    return np.cumsum(1.0 / np.arange(1, n + 1))


def harmonic_generator_numba(n: int) -> Generator[float, None, None]:
    """Numba-backed generator of harmonic numbers / Numba destekli harmonik sayı üreteci"""
    sums = harmonic_numbers_numba(n)
    for i in range(n):
        yield float(sums[i])


# -----------------------------
# Approximation Functions / Yaklaştırma Fonksiyonları
# -----------------------------

@lru_cache(maxsize=32)
def bernoulli_number(n: int) -> float:
    """
    Compute Bernoulli numbers (cached). Uses scipy if available, else internal fallback.
    Bernoulli sayılarını hesaplar (önbellekli). Varsa scipy, yoksa yedek yöntem kullanır.
    """
    if n == 0:
        return 1.0
    if n == 1:
        return -0.5
    if n % 2 != 0:
        return 0.0

    if _HAS_SCIPY:
        # scipy returns all Bernoulli numbers up to n
        return _scipy_bernoulli(n)[n]

    # Fallback using precomputed even Bernoulli numbers up to B_12
    # Yedek yöntem: önceden hesaplanmış çift Bernoulli sayıları (B_2 … B_12)
    even_bernoulli = {
        2: 1/6,
        4: -1/30,
        6: 1/42,
        8: -1/30,
        10: 5/66,
        12: -691/2730,
    }
    if n in even_bernoulli:
        return even_bernoulli[n]
    # For larger n, compute via zeta relation using scipy if needed, else raise
    # Daha büyük n için scipy gerekir, yoksa hata
    raise NotImplementedError(
        f"Bernoulli number B_{n} requires scipy. Please install scipy. "
        f"Bernoulli sayısı B_{n} için scipy gereklidir."
    )


def harmonic_number_approx(
    n: int,
    method: ApproximationMethod = ApproximationMethod.EULER_MASCHERONI,
    k: int = 2
) -> float:
    """
    Approximate harmonic number / Yaklaşık harmonik sayı
    """
    if n <= 0:
        raise ValueError("n must be positive / n pozitif olmalıdır")

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
        raise ValueError("Unknown approximation method / Bilinmeyen yaklaştırma yöntemi")


@numba.njit
def harmonic_sum_approx_numba(
    n: np.ndarray,
    method: int = 1,          # 0:EULER_MASCHERONI, 1:EULER_MACLAURIN
    order: int = 4
) -> np.ndarray:
    """
    Numba-compatible vectorized harmonic approximation.
    Numba uyumlu vektörel harmonik yaklaştırma.
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
# ℓ² (Hilbert Space) Membership Test / ℓ² (Hilbert Uzayı) Aidiyet Testi
# -----------------------------
def is_in_hilbert(
    sequence: Union[List[float], np.ndarray, Generator[float, None, None]],
    max_terms: int = 10000,
    tolerance: float = 1e-8
) -> bool:
    """
    Test whether a sequence belongs to ℓ² (Hilbert space).
    Bir dizinin ℓ² (Hilbert) uzayında olup olmadığını test eder.
    Determines if a given sequence belongs to the Hilbert space ℓ².
    A sequence {a_n} is in ℓ² (Hilbert space) if the sum of the squares of its terms is finite:
        Σ |a_n|² < ∞
    This function computes the partial sum of squared terms up to `max_terms` and checks
    whether the sum converges within a given tolerance (i.e., the increments become negligible).
    Parameters
    ----------
    sequence : list, np.ndarray, or generator
        The input sequence to test (e.g., [1, 1/2, 1/3, ...]).
    max_terms : int, optional
        Maximum number of terms to consider for convergence check. Default is 10,000.
    tolerance : float, optional
        The threshold for determining convergence. If the increment in cumulative sum
        falls below this value for consecutive steps, the series is considered convergent.
        Default is 1e-6.
    Returns
    -------
    bool
        True if the sequence is likely in ℓ² (sum of squares converges), False otherwise.
    Examples
    --------
    >>> from oresmen import harmonic_numbers_numba, is_in_hilbert
    >>> import numpy as np
    # Harmonic terms: a_n = 1/n → sum(1/n²) converges → in Hilbert space
    >>> n = 1000
    >>> harmonic_terms = 1 / np.arange(1, n+1)
    >>> is_in_hilbert(harmonic_terms)
    True
    # Constant terms: a_n = 1 → sum(1²) = ∞ → not in Hilbert space
    >>> constant_terms = np.ones(1000)
    >>> is_in_hilbert(constant_terms)
    False
    Notes
    -----
    - This is a numerical approximation. True mathematical convergence may require
      analytical proof, but this function provides a practical check for common sequences.
    - Sequences like 1/n, 1/n^(0.6), log(n)/n are tested implicitly via their decay rate.
    """

    if isinstance(sequence, Generator):
        sequence = list(sequence)

    arr = np.array(sequence, dtype=float)

    if arr.size == 0:
        return True

    if not np.all(np.isfinite(arr)):
        return False

    n_terms = min(len(arr), max_terms)
    test_seq = arr[:n_terms]

    squares = test_seq ** 2
    cumsum = np.cumsum(squares)
    total_sum = cumsum[-1]

    if not np.isfinite(total_sum):
        return False

    # p‑series heuristic – eşik 100
    if n_terms > 100 and np.all(test_seq[100:] > 0):
        log_terms = np.log(test_seq[100:] + 1e-12)
        log_n = np.log(np.arange(100, n_terms))
        try:
            alpha = -np.polyfit(log_n, log_terms, 1)[0]
            if alpha > 0.5:
                return True
            elif 0 < alpha <= 0.5:
                return False
            elif alpha > 10:   # üstel sönüm
                return True
        except Exception:
            pass

    # kuyruk katkısı
    if n_terms > 1000:
        last_contribution = squares[-1000:]
        if np.sum(last_contribution) < tolerance:
            return True

    # oran testi (üstel sönüm)
    if n_terms > 100:
        ratios = np.abs(test_seq[1:100] / (test_seq[:99] + 1e-12))
        if np.mean(ratios) < 0.95:
            return True

    # Hiçbir yakınsama belirtisi yoksa ℓ²'de değildir
    return False


# -----------------------------
# Utility Functions / Yardımcı Fonksiyonlar
# -----------------------------

def harmonic_sequence(n_terms: int, start: int = 1) -> np.ndarray:
    """Generate harmonic sequence terms: a_n = 1/n / Harmonik dizi terimlerini üretir: a_n = 1/n"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    indices = np.arange(start, start + n_terms, dtype=float)
    return 1.0 / indices


def p_series(p: float, n_terms: int, start: int = 1) -> np.ndarray:
    """Generate p-series: a_n = 1/n^p / p-serisi üretir: a_n = 1/n^p"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    indices = np.arange(start, start + n_terms, dtype=float)
    return 1.0 / (indices ** p)


def geometric_sequence(ratio: float, n_terms: int, start: int = 1) -> np.ndarray:
    """Generate geometric sequence: a_n = ratio^n / Geometrik dizi üretir: a_n = ratio^n"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    exponents = np.arange(start, start + n_terms, dtype=float)
    return ratio ** exponents


def analyze_sequence(
    sequence: Union[List[float], np.ndarray],
    name: str = "Sequence / Dizi",
    n_display: int = 5
) -> dict:
    """Detailed analysis of a sequence / Bir dizinin detaylı analizi"""
    seq = np.array(sequence, dtype=float)
    squares = seq ** 2
    cumsum = np.cumsum(squares)

    results = {
        'name': name,
        'first_terms': seq[:n_display].tolist(),
        'n_terms': len(seq),
        'sum_of_squares': cumsum[-1] if np.isfinite(cumsum[-1]) else np.inf,
        'in_hilbert': is_in_hilbert(seq),
        'max_term': float(np.max(np.abs(seq))),
        'decay_rate': None
    }

    if len(seq) > 100 and np.all(seq[100:] > 0):
        log_terms = np.log(seq[100:] + 1e-12)
        log_n = np.log(np.arange(100, len(seq)))
        try:
            alpha = -np.polyfit(log_n, log_terms, 1)[0]
            results['decay_rate'] = alpha
            results['decay_description'] = f"~ 1/n^{alpha:.2f}"
        except Exception:
            pass

    return results


def compare_sequences(sequences: dict, n_test: int = 5000) -> None:
    """Compare multiple sequences / Birden fazla diziyi karşılaştırır"""
    results = []
    for name, seq in sequences.items():
        # Auto-extend if sequence is too short / Dizi çok kısaysa otomatik uzat
        if len(seq) < n_test:
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
            else:
                continue  # Cannot extend automatically / Otomatik uzatılamaz

        test_seq = seq[:n_test]
        squares_sum = np.sum(test_seq ** 2)
        in_hilbert = is_in_hilbert(test_seq)

        results.append({
            "Sequence / Dizi": name,
            "First 5 terms / İlk 5 terim": str(test_seq[:5].tolist())[:60],
            "∑ a_n²": f"{squares_sum:.6f}" if np.isfinite(squares_sum) else "∞",
            "In ℓ²? / ℓ²'de mi?": "✓ Yes / Evet" if in_hilbert else "✗ No / Hayır"
        })

    # Pretty print if tabulate is available, otherwise simple print
    # Tabulate varsa güzel çıktı, yoksa basit çıktı
    try:
        from tabulate import tabulate
        print(tabulate(results, headers="keys", tablefmt="grid", stralign="left"))
    except ImportError:
        for row in results:
            print(row)


# -----------------------------
# Performance Analysis / Performans Analizi
# -----------------------------

def benchmark_harmonic(n: int, runs: int = 10) -> dict:
    """Compare different computation methods / Farklı hesaplama yöntemlerini karşılaştırır"""
    results = {}

    # Warm-up / Isınma
    _ = harmonic_number_numba(10)

    # Pure Python loop (Numba JIT) / Saf Python döngüsü (Numba JIT)
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number(n)
    results['pure_python_numba_loop'] = (time.perf_counter() - start) / runs

    # Numba with NumPy / NumPy ile Numba
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number_numba(n)
    results['numba'] = (time.perf_counter() - start) / runs

    # Approximation / Yaklaşık
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number_approx(n)
    results['approximate'] = (time.perf_counter() - start) / runs

    return results


def compare_with_approximation(n: int) -> dict:
    """Compare exact and approximate values / Tam ve yaklaşık değerleri karşılaştırır"""
    exact = harmonic_number(n)
    approx = harmonic_number_approx(n)
    error = abs(exact - approx)
    relative_error = error / exact if exact != 0 else 0.0

    return {
        'exact': exact,
        'approximate': approx,
        'absolute_error': error,
        'relative_error': relative_error,
        'percentage_error': relative_error * 100
    }


def harmonic_convergence_analysis(n_values: np.ndarray) -> dict:
    """Analyze harmonic series convergence for given n values / Verilen n değerleri için harmonik seri yakınsamasını analiz eder"""
    n_max = n_values[-1]
    all_sums = harmonic_numbers_numba(n_max)          # H_1 ... H_n_max
    exact = all_sums[n_values - 1]                    # select requested indices / istenen indisleri seç
    approx = harmonic_sum_approx_numba(n_values.astype(float))
    return {
        'exact_sums': exact,
        'approx_sums': approx,
        'errors': np.abs(exact - approx),
        'log_fit': np.polyfit(np.log(n_values), exact, 1)
    }


# -----------------------------
# Main Program / Ana Program
# -----------------------------

def main():
    """Main test routine / Ana test rutini"""
    logger.info("=" * 70)
    logger.info(" ORESMEN MODULE TEST (UPDATED VERSION) / ORESMEN MODÜL TESTİ (GÜNCEL SÜRÜM)")
    logger.info("=" * 70)

    # Basic calculations / Temel hesaplamalar
    logger.info("Oresme sequence (first 5) / Oresme dizisi (ilk 5): %s", oresme_sequence(5))
    logger.info("Fractional harmonic numbers H1-H3 / Kesirli harmonik sayılar H1-H3: %s", harmonic_numbers(3))
    logger.info("5th harmonic number / 5. harmonik sayı: %.4f", harmonic_number(5))

    # Numba computations / Numba hesaplamaları
    _ = harmonic_number_numba(10)   # warm-up
    logger.info("Numba accelerated H1-H5 / Numba ile hızlandırılmış H1-H5: %s", harmonic_numbers_numba(5))

    # ℓ² tests / ℓ² testleri
    logger.info("-" * 50)
    logger.info("ℓ² (Hilbert space) membership tests / ℓ² (Hilbert uzayı) aidiyet testleri:")
    n_test = 10000
    harmonic_seq = 1 / np.arange(1, n_test + 1)
    logger.info("  1/n in ℓ²? / 1/n ℓ²'de mi? %s", is_in_hilbert(harmonic_seq))

    slow_seq = 1 / np.sqrt(np.arange(1, n_test + 1))
    logger.info("  1/√n in ℓ²? / 1/√n ℓ²'de mi? %s", is_in_hilbert(slow_seq))

    oresme_seq = np.array([i / (2 ** i) for i in range(1, n_test + 1)])
    logger.info("  n/2ⁿ in ℓ²? / n/2ⁿ ℓ²'de mi? %s", is_in_hilbert(oresme_seq))

    # Performance test / Performans testi
    n_perf = 100000
    logger.info("-" * 50)
    logger.info("Performance test (n=%d) / Performans testi (n=%d):", n_perf, n_perf)
    bench = benchmark_harmonic(n_perf)
    for method, t in bench.items():
        logger.info("%25s: %.6f s/run", method, t)

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
