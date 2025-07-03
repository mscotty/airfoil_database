# airfoil_database/__init__.py (alternative version)
from airfoil_database.core import AirfoilDatabase, Airfoil, AeroCoeff, AirfoilGeometry
from airfoil_database.analysis import GeometryAnalyzer, AeroAnalyzer
from airfoil_database.plotting import AirfoilPlotter, DataPlotter
from airfoil_database.xfoil import XFoilRunner, PointcloudProcessor

__all__ = [
    'AirfoilDatabase',
    'Airfoil',
    'AeroCoeff',
    'AirfoilGeometry',
    'GeometryAnalyzer',
    'AeroAnalyzer',
    'AirfoilPlotter',
    'DataPlotter',
    'XFoilRunner',
    'PointcloudProcessor'
]
