import pytest
from enum import Enum
import re

from DASC500.classes.AirfoilSeries import AirfoilSeries  # Replace your_module with the actual module name

def test_airfoil_series_from_string_naca():
    assert AirfoilSeries.from_string("some.NACA.something") == AirfoilSeries.NACA

def test_airfoil_series_from_string_wortmann_fx():
    assert AirfoilSeries.from_string("some.WORTMANN_FX.something") == AirfoilSeries.WORTMANN_FX

def test_airfoil_series_from_string_eppler():
    assert AirfoilSeries.from_string("some.EPL.something") == AirfoilSeries.EPPLER

def test_airfoil_series_from_string_selig():
    assert AirfoilSeries.from_string("some.SELIG.something") == AirfoilSeries.SELIG

def test_airfoil_series_from_string_goettingen():
    assert AirfoilSeries.from_string("some.GOETTINGEN.something") == AirfoilSeries.GOETTINGEN

def test_airfoil_series_from_string_risk():
    assert AirfoilSeries.from_string("some.RISK.something") == AirfoilSeries.RISK

def test_airfoil_series_from_string_laminar():
    assert AirfoilSeries.from_string("some.LAM.something") == AirfoilSeries.LAMINAR

def test_airfoil_series_from_string_other():
    assert AirfoilSeries.from_string("some.OTHER.something") == AirfoilSeries.OTHER

def test_airfoil_series_from_string_none():
    assert AirfoilSeries.from_string("some.unknown.something") is None

def test_identify_airfoil_series_naca():
    assert AirfoilSeries.identify_airfoil_series("naca 4412") == AirfoilSeries.NACA
    assert AirfoilSeries.identify_airfoil_series("NACA0012") == AirfoilSeries.NACA

def test_identify_airfoil_series_wortmann_fx():
    assert AirfoilSeries.identify_airfoil_series("FX 60-126") == AirfoilSeries.WORTMANN_FX
    assert AirfoilSeries.identify_airfoil_series("fx60-126") == AirfoilSeries.WORTMANN_FX
    assert AirfoilSeries.identify_airfoil_series("FX60-126") == AirfoilSeries.WORTMANN_FX
    assert AirfoilSeries.identify_airfoil_series("FX 60-126 some other text") == AirfoilSeries.WORTMANN_FX

def test_identify_airfoil_series_eppler():
    assert AirfoilSeries.identify_airfoil_series("epl 123") == AirfoilSeries.EPPLER
    assert AirfoilSeries.identify_airfoil_series("EPL123") == AirfoilSeries.EPPLER

def test_identify_airfoil_series_selig():
    assert AirfoilSeries.identify_airfoil_series("s1223") == AirfoilSeries.SELIG
    assert AirfoilSeries.identify_airfoil_series("S1223") == AirfoilSeries.SELIG

def test_identify_airfoil_series_goettingen():
    assert AirfoilSeries.identify_airfoil_series("goe 490") == AirfoilSeries.GOETTINGEN
    assert AirfoilSeries.identify_airfoil_series("GOE490") == AirfoilSeries.GOETTINGEN
    assert AirfoilSeries.identify_airfoil_series("goe490") == AirfoilSeries.GOETTINGEN
    assert AirfoilSeries.identify_airfoil_series("GOE 490 some other text") == AirfoilSeries.GOETTINGEN

def test_identify_airfoil_series_risk():
    assert AirfoilSeries.identify_airfoil_series("risk 123") == AirfoilSeries.RISK
    assert AirfoilSeries.identify_airfoil_series("RISK123") == AirfoilSeries.RISK
    assert AirfoilSeries.identify_airfoil_series("risk123") == AirfoilSeries.RISK
    assert AirfoilSeries.identify_airfoil_series("RISK 123 some other text") == AirfoilSeries.RISK

def test_identify_airfoil_series_laminar():
    assert AirfoilSeries.identify_airfoil_series("lam 123") == AirfoilSeries.LAMINAR
    assert AirfoilSeries.identify_airfoil_series("LAM123") == AirfoilSeries.LAMINAR

def test_identify_airfoil_series_other():
    assert AirfoilSeries.identify_airfoil_series("some unknown airfoil") == AirfoilSeries.OTHER
    assert AirfoilSeries.identify_airfoil_series("") == AirfoilSeries.OTHER
    assert AirfoilSeries.identify_airfoil_series("123") == AirfoilSeries.OTHER