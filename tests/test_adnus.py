# tests/test_oresmen.py
import pytest
import math
from fractions import Fraction
from oresmen.main import (
    BicomplexNumber,
    NeutrosophicNumber,
    NeutrosophicComplexNumber,
    NeutrosophicBicomplexNumber,
    oresme_sequence,
    harmonic_numbers,
    binet_formula,
)

class TestBicomplexNumber:
    def test_addition(self):
        bc1 = BicomplexNumber(1 + 2j, 3 + 4j)
        bc2 = BicomplexNumber(5 + 6j, 7 + 8j)
        assert bc1 + bc2 == BicomplexNumber(6 + 8j, 10 + 12j)

    def test_subtraction(self):
        bc1 = BicomplexNumber(1 + 2j, 3 + 4j)
        bc2 = BicomplexNumber(5 + 6j, 7 + 8j)
        assert bc1 - bc2 == BicomplexNumber(-4 - 4j, -4 - 4j)

    def test_multiplication(self):
        bc1 = BicomplexNumber(1 + 1j, 1 + 1j)
        bc2 = BicomplexNumber(1 + 1j, 1 + 1j)
        # (1+i)(1+i) - (1+i)(1+i) = 2i - 2i = 0
        # (1+i)(1+i) + (1+i)(1+i) = 2i + 2i = 4i
        assert bc1 * bc2 == BicomplexNumber(0, 4j)

    def test_norm(self):
        bc = BicomplexNumber(3 + 4j, 5 + 12j) # |3+4j|=5, |5+12j|=13
        assert math.isclose(bc.norm(), math.sqrt(5**2 + 13**2))


class TestNeutrosophicNumber:
    def test_addition(self):
        n1 = NeutrosophicNumber(1.5, 2.5)
        n2 = NeutrosophicNumber(3.0, 4.0)
        assert n1 + n2 == NeutrosophicNumber(4.5, 6.5)

    def test_multiplication(self):
        n1 = NeutrosophicNumber(2, 3)
        n2 = NeutrosophicNumber(4, 5)
        # a*c = 2*4 = 8
        # ad+bc+bd = 2*5 + 3*4 + 3*5 = 10 + 12 + 15 = 37
        assert n1 * n2 == NeutrosophicNumber(8, 37)


class TestNeutrosophicComplexNumber:
    def test_addition(self):
        nc1 = NeutrosophicComplexNumber(1 + 2j, 3 + 4j)
        nc2 = NeutrosophicComplexNumber(5 + 6j, 7 + 8j)
        assert nc1 + nc2 == NeutrosophicComplexNumber(6 + 8j, 10 + 12j)

    def test_multiplication(self):
        A, B = 1 + 1j, 2 + 2j
        C, D = 3 + 3j, 4 + 4j
        nc1 = NeutrosophicComplexNumber(A, B)
        nc2 = NeutrosophicComplexNumber(C, D)
        # AC
        det_part = A * C
        # AD + BC + BD
        ind_part = A * D + B * C + B * D
        assert nc1 * nc2 == NeutrosophicComplexNumber(det_part, ind_part)


class TestNeutrosophicBicomplexNumber:
    def test_addition(self):
        nbc1 = NeutrosophicBicomplexNumber(BicomplexNumber(1,1j), BicomplexNumber(2,2j))
        nbc2 = NeutrosophicBicomplexNumber(BicomplexNumber(3,3j), BicomplexNumber(4,4j))
        result = nbc1 + nbc2
        assert result.determinate == BicomplexNumber(4, 4j)
        assert result.indeterminate == BicomplexNumber(6, 6j)


class TestHelperFunctions:
    def test_oresme_sequence(self):
        assert oresme_sequence(3) == [1/2, 2/4, 3/8]
        assert oresme_sequence(0) == []

    def test_harmonic_numbers(self):
        harmonics = list(harmonic_numbers(3))
        assert harmonics == [Fraction(1, 1), Fraction(3, 2), Fraction(11, 6)]

    def test_binet_formula(self):
        # F(10) = 55
        assert math.isclose(binet_formula(10), 55.0)
        with pytest.raises(ValueError):
            binet_formula(-1)
