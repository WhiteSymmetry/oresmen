# Oresme Numba

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16634186.svg)](https://doi.org/10.5281/zenodo.16634186)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI--blue)](https://doi.org/)

[![figshare DOI](https://img.shields.io/badge/DOI--blue)](https://doi.org/)

[![ResearchGate DOI](https://img.shields.io/badge/DOI-10.13140/RG.2.2.19566.52804-blue)](https://doi.org/10.13140/RG.2.2.19566.52804)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/oresmen/badges/version.svg)](https://anaconda.org/bilgi/oresmen)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/oresmen/badges/latest_release_date.svg)](https://anaconda.org/bilgi/oresmen)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/oresmen/badges/platforms.svg)](https://anaconda.org/bilgi/oresmen)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/oresmen/badges/license.svg)](https://anaconda.org/bilgi/oresmen)
[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python CI](https://github.com/WhiteSymmetry/oresmen/actions/workflows/python_ci.yml/badge.svg?branch=main)](https://github.com/WhiteSymmetry/oresmen/actions/workflows/python_ci.yml)
[![codecov](https://codecov.io/gh/WhiteSymmetry/oresmen/graph/badge.svg?token=N8TAVZUJ1C)](https://codecov.io/gh/WhiteSymmetry/oresmen)
[![Documentation Status](https://readthedocs.org/projects/oresmen/badge/?version=latest)](https://oresmen.readthedocs.io/en/latest/)
[![Binder](https://terrarium.evidencepub.io/badge_logo.svg)](https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/oresmen/HEAD)
[![PyPI version](https://badge.fury.io/py/oresmen.svg)](https://badge.fury.io/py/oresmen)
[![PyPI Downloads](https://static.pepy.tech/badge/oresmen)](https://pepy.tech/projects/oresmen)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md) 
[![Linted with Ruff](https://img.shields.io/badge/Linted%20with-Ruff-green?logo=python&logoColor=white)](https://github.com/astral-sh/ruff)

---

<p align="left">
    <table>
        <tr>
            <td style="text-align: center;">PyPI</td>
            <td style="text-align: center;">
                <a href="https://pypi.org/project/oresmen/">
                    <img src="https://badge.fury.io/py/oresmen.svg" alt="PyPI version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">Conda</td>
            <td style="text-align: center;">
                <a href="https://anaconda.org/bilgi/oresmen">
                    <img src="https://anaconda.org/bilgi/oresmen/badges/version.svg" alt="conda-forge version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">DOI</td>
            <td style="text-align: center;">
                <a href="https://doi.org/10.5281/zenodo.16634186">
                    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.16634186.svg" alt="DOI" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">License: MIT</td>
            <td style="text-align: center;">
                <a href="https://opensource.org/licenses/MIT">
                    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" height="18"/>
                </a>
            </td>
        </tr>
    </table>
</p>

---


Oresme numbers refer to the sums related to the harmonic series.

---
### **Türkçe Tanım:**
**Oresme Sayıları**, 14. yüzyılda Nicole Oresme tarafından incelenen matematiksel serilerdir. Oresme sayıları harmonik seriye ait toplamları ifade eder. İki türü vardır:  
1. **\( \frac{n}{2^n} \) serisi** (Oresme'nin orijinal çalışması),  
2. **Harmonik sayılar** (\( H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n} \)).  
Bu sayılar, analiz ve sayı teorisinde önemli rol oynar.

---

### **English Definition:**
**Oresme Numbers** are mathematical series studied by Nicole Oresme in the 14th century. Oresme numbers refer to the sums related to the harmonic series. They include two types:  
1. The **\( \frac{n}{2^n} \) sequence** (Oresme's original work),  
2. **Harmonic numbers** (\( H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n} \)).  
These numbers play a key role in analysis and number theory.

---

### **Fark/Karşılaştırma (Difference):**
- **Oresme'nin \( \frac{n}{2^n} \) serisi** ıraksaklık kanıtları için önemlidir.  
- **Harmonik sayılar** (\( H_n \)) ise logaritmik büyüme gösterir ve \( n \to \infty \) iken ıraksar.  
- Modern literatürde "Oresme numbers" terimi daha çok tarihsel bağlamda kullanılır.

---

## Kurulum (Türkçe) / Installation (English)

### Python ile Kurulum / Install with pip, conda, mamba
```bash
pip install oresmen -U
python -m pip install -U oresmen
conda install bilgi::oresmen -y
mamba install bilgi::oresmen -y
```

```diff
- pip uninstall Oresme -y
+ pip install -U oresmen
+ python -m pip install -U oresmen
```

[PyPI](https://pypi.org/project/Oresme/)

### Test Kurulumu / Test Installation

```bash
pip install -i https://test.pypi.org/simple/ oresmen -U
```

### Github Master Kurulumu / GitHub Master Installation

**Terminal:**

```bash
pip install git+https://github.com/WhiteSymmetry/oresmen.git
```

**Jupyter Lab, Notebook, Visual Studio Code:**

```python
!pip install git+https://github.com/WhiteSymmetry/oresmen.git
# or
%pip install git+https://github.com/WhiteSymmetry/oresmen.git
```

---

## Kullanım (Türkçe) / Usage (English)

Note: "\Lib\site-packages\numba\__init__.py"

    if numpy_version > (2, 0):
        msg = (f"Numba needs NumPy 2.0 or less. Got NumPy "
               f"{numpy_version[0]}.{numpy_version[1]}.")
        raise ImportError(msg)

-->

    if numpy_version > (2, 5):
        msg = (f"Numba needs NumPy 2.5 or less. Got NumPy "
               f"{numpy_version[0]}.{numpy_version[1]}.")
        raise ImportError(msg)

```python
import oresmen as on

# Doğrudan erişim (on.main.harmonic_number yerine)
n = 100
hn = on.harmonic_number(n)
print(f"H_{n} = {hn}")

# Enum sınıfına doğrudan erişim
approx_hn = on.harmonic_number_approx(
    n,
    method=on.ApproximationMethod.EULER_MASCHERONI
)
print(f"H_{n} (Yaklaşık) = {approx_hn}")

# Numba ile hızlandırılmış diziye erişim
sums_array = on.harmonic_numbers_numba(10)
print(f"İlk 10 harmonik sayı: {sums_array}")
```

```python
import oresmen as on
import numpy as np
import numba
import time
import matplotlib.pyplot as plt

# Simple usage example
plt.figure(figsize=(10, 5))
plt.plot(on.harmonic_numbers_numba(500))
plt.title("First 5000000 Harmonic Numbers")
plt.xlabel("n")
plt.ylabel("H(n)")
plt.show()
```

```python
import oresmen
oresmen.__version__
```

```python
import importlib
import inspect
import oresmen as on  # Varsa import hatasını yakalamak için


def diagnose_module(module_name):
    try:
        # Modülü yükle
        module = importlib.import_module(module_name)
        
        print(f"\n{' Modül Tanılama Raporu ':=^80}")
        print(f"Modül adı: {module_name}")
        print(f"Modül dosya yolu: {inspect.getfile(module)}")
        
        # Modülün tüm özelliklerini listele
        print("\nModülde bulunan özellikler:")
        members = inspect.getmembers(module)
        public_members = [name for name, _ in members if not name.startswith('_')]
        print(public_members)
        
        # Özel olarak kontrol edilecek fonksiyonlar
        required_functions = [
            'oresme_sequence',
            'harmonic_numbers',
            'harmonic_number',
            'harmonic_number_numba',
            'harmonic_numbers_numba',
            'harmonic_generator_numba',
            'harmonic_number_approx'
        ]
        
        print("\nEksik olan fonksiyonlar:")
        missing = [fn for fn in required_functions if not hasattr(module, fn)]
        print(missing if missing else "Tüm gerekli fonksiyonlar mevcut")
        
        # __all__ değişkenini kontrol et
        print("\n__all__ değişkeni:")
        if hasattr(module, '__all__'):
            print(module.__all__)
        else:
            print("__all__ tanımlı değil (tüm public fonksiyonlar içe aktarılır)")
            
    except ImportError as e:
        print(f"\nHATA: Modül yüklenemedi - {e}")
    except Exception as e:
        print(f"\nBeklenmeyen hata: {e}")

# Tanılama çalıştır
diagnose_module('oresmen')

# Alternatif olarak doğrudan kontrol
print("\nDoğrudan fonksiyon varlığı kontrolü:")
try:
    print("harmonic_numbers_numba mevcut mu?", hasattr(on, 'harmonic_numbers_numba'))
    if hasattr(on, 'harmonic_numbers_numba'):
        print("Fonksiyon imzası:", inspect.signature(on.harmonic_numbers_numba))
    else:
        print("Eksik fonksiyon: harmonic_numbers_numba")
except Exception as e:
    print("Kontrol sırasında hata:", e)
```

```python
# 1. Alternatif içe aktarma yöntemi
from oresmen import harmonic_numbers_numba  # Doğrudan import deneyin
import oresmen as on

# 2. Modülü yeniden yükleme
import importlib
importlib.reload(on)

# 3. Fonksiyonun alternatif isimle var olup olmadığını kontrol
print("Alternatif fonksiyon isimleri:", [name for name in dir(on) if 'harmonic' in name.lower()])
```
---

### Development
```bash
# Clone the repository
git clone https://github.com/WhiteSymmetry/oresmen.git
cd oresmen

# Install in development mode
python -m pip install -ve . # Install package in development mode

# Run tests
pytest

Notebook, Jupyterlab, Colab, Visual Studio Code
!python -m pip install git+https://github.com/WhiteSymmetry/oresmen.git
```
---

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX


### APA

```

Keçeci, M. (2025). oresmen [Data set]. ResearchGate. https://doi.org/10.13140/RG.2.2.19566.52804

Keçeci, M. (2025). oresmen [Data set]. figshare. https://doi.org/

Keçeci, M. (2025). oresmen [Data set]. WorkflowHub. https://doi.org/

Keçeci, M. (2025). oresmen (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.16634186
```

### Chicago

```

Keçeci, Mehmet. oresmen [Data set]. ResearchGate, 2025. https://doi.org/10.13140/RG.2.2.19566.52804

Keçeci, Mehmet (2025). oresmen [Data set]. figshare, 2025. https://doi.org/

Keçeci, Mehmet. oresmen [Data set]. WorkflowHub, 2025. https://doi.org/

Keçeci, Mehmet. oresmen. Open Science Articles (OSAs), Zenodo, 2025. [https://doi.org/](https://doi.org/10.5281/zenodo.16634186)

```


### Lisans (Türkçe) / License (English)

```
This project is licensed under the MIT License.
```
