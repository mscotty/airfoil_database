from enum import Enum
import re

class AirfoilSeries(Enum):
    NACA = "NACA"
    WORTMANN_FX = "Wortmann FX"
    EPPLER = "Eppler"
    SELIG = "Selig"
    GOETTINGEN = "Goettingen"
    RISK = "RISK"
    ALTHAUS = "Althaus"
    QUABECK_HQ = "Quabeck HQ"
    MARTIN_HEPPERLE = "Martin Hepperle"
    DRELA = "Drela"
    LAMINAR = "Laminar"
    OTHER = "Other"

    @staticmethod
    def from_string(airfoil_series_str):
        airfoil_series_str = airfoil_series_str.upper()
        if "NACA" in airfoil_series_str:
            return AirfoilSeries.NACA
        elif 'WORTMANN_FX' in airfoil_series_str or 'WORTMANN FX' in airfoil_series_str or 'WORTMANN' in airfoil_series_str:
            return AirfoilSeries.WORTMANN_FX
        elif "EPL" in airfoil_series_str:
            return AirfoilSeries.EPPLER
        elif "SELIG" in airfoil_series_str:  # Covers Selig S series
            return AirfoilSeries.SELIG
        elif 'GOETTINGEN' in airfoil_series_str:
            return AirfoilSeries.GOETTINGEN
        elif 'RISK' in airfoil_series_str:
            return AirfoilSeries.RISK
        elif 'ALTHAUS' in airfoil_series_str:
            return AirfoilSeries.ALTHAUS
        elif 'QUABECK' in airfoil_series_str:
            return AirfoilSeries.QUABECK_HQ
        elif 'MARTIN HEPPERLE' in airfoil_series_str:
            return AirfoilSeries.MARTIN_HEPPERLE
        elif "DRELA" in airfoil_series_str:
            return AirfoilSeries.DRELA
        elif "LAM" in airfoil_series_str:
            return AirfoilSeries.LAMINAR
        elif 'OTHER' in airfoil_series_str:
            return AirfoilSeries.OTHER
        else:
            return AirfoilSeries.OTHER

    @staticmethod
    def identify_airfoil_series(airfoil_name):
        """Identifies the airfoil series based on the airfoil name."""

        name_upper = airfoil_name.upper()

        if "NACA" in name_upper:
            return AirfoilSeries.NACA
        elif re.match(r"FX\s?\d+-\d+", name_upper):
            return AirfoilSeries.WORTMANN_FX
        elif re.match(r"E\d+", name_upper):
            return AirfoilSeries.EPPLER
        elif "S" in name_upper and re.match(r"S\d+", name_upper):  # Covers Selig S series
            return AirfoilSeries.SELIG
        elif re.match(r"GOE\s?\d+", name_upper):
            return AirfoilSeries.GOETTINGEN
        elif re.match(r"RISK\s?\d+", name_upper):
            return AirfoilSeries.RISK
        elif re.match(r"AH\d+\s?\d+", name_upper):
            return AirfoilSeries.ALTHAUS
        elif re.match(r"HQ\d+", name_upper):
            return AirfoilSeries.QUABECK_HQ
        elif re.match(r"MH\d+", name_upper):
            return AirfoilSeries.MARTIN_HEPPERLE
        elif re.match(r"AG\d+", name_upper):
            return AirfoilSeries.DRELA
        elif "LAM" in name_upper:
            return AirfoilSeries.LAMINAR
        else:
            return AirfoilSeries.OTHER