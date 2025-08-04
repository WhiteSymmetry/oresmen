from setuptools import setup, find_packages

setup(
    name="oresmen",
    use_scm_version=True,  # Sürüm bilgisini setuptools_scm ile alır
    setup_requires=["setuptools", "wheel", "setuptools_scm"],  # Gerekli kurulum bağımlılıkları
    version='0.1.0',
    packages=find_packages(where="src"),  # src dizinindeki modülleri bul
    package_dir={"": "src"},  # src dizinine yönlendirme
    include_package_data=True,  # Ek dosyaları dahil et
    install_requires=["numpy","numba"],
    author="Mehmet Keçeci",
    description="A module for generating Oresme numbers (harmonic series partial sums)",
    url="https://github.com/WhiteSymmetry/oresmen",
    license="MIT",
    python_requires='>=3.9',
)
