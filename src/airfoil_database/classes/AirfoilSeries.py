from enum import Enum
import re

class AirfoilSeries(Enum):
    NACA = "NACA"
    WORTMANN_FX = "Wortmann FX"
    EPPLER = "Eppler"
    SELIG = "Selig"
    GOETTINGEN = "Goettingen"
    RISK = "RISK"
    LAMINAR = "Laminar"
    OTHER = "Other"

    @staticmethod
    def from_string(airfoil_series_str):
        if "NACA" in airfoil_series_str:
            return AirfoilSeries.NACA
        elif 'WORTMANN_FX' in airfoil_series_str:
            return AirfoilSeries.WORTMANN_FX
        elif "EPL" in airfoil_series_str:
            return AirfoilSeries.EPPLER
        elif "SELIG" in airfoil_series_str:  # Covers Selig S series
            return AirfoilSeries.SELIG
        elif 'GOETTINGEN' in airfoil_series_str:
            return AirfoilSeries.GOETTINGEN
        elif 'RISK' in airfoil_series_str:
            return AirfoilSeries.RISK
        elif "LAM" in airfoil_series_str:
            return AirfoilSeries.LAMINAR
        elif 'OTHER' in airfoil_series_str:
            return AirfoilSeries.OTHER
        else:
            return None

    @staticmethod
    def identify_airfoil_series(airfoil_name):
        """Identifies the airfoil series based on the airfoil name."""

        name_upper = airfoil_name.upper()

        if "NACA" in name_upper:
            return AirfoilSeries.NACA
        elif re.match(r"FX\s?\d+-\d+", name_upper):
            return AirfoilSeries.WORTMANN_FX
        elif "EPL" in name_upper:
            return AirfoilSeries.EPPLER
        elif "S" in name_upper and re.match(r"S\d+", name_upper):  # Covers Selig S series
            return AirfoilSeries.SELIG
        elif re.match(r"GOE\s?\d+", name_upper):
            return AirfoilSeries.GOETTINGEN
        elif re.match(r"RISK\s?\d+", name_upper):
            return AirfoilSeries.RISK
        elif "LAM" in name_upper:
            return AirfoilSeries.LAMINAR
        else:
            return AirfoilSeries.OTHER