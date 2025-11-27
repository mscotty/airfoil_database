from typing import List, Optional, Dict, Any, Tuple
from sqlmodel import Field, SQLModel, Column, JSON, String, Float


class Airfoil(SQLModel, table=True):
    __tablename__ = "airfoils"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None
    pointcloud: Optional[str] = None
    airfoil_series: Optional[str] = None
    source: Optional[str] = None


class AeroCoeff(SQLModel, table=True):
    __tablename__ = "aero_coeffs"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, foreign_key="airfoils.name")
    reynolds_number: float
    mach: float
    ncrit: float
    alpha: float
    cl: Optional[float] = None
    cd: Optional[float] = None
    cm: Optional[float] = None

    class Config:
        unique_together = [("name", "reynolds_number", "mach", "alpha")]


class AirfoilGeometry(SQLModel, table=True):
    __tablename__ = "airfoil_geometry"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, foreign_key="airfoils.name")

    # Scalar geometric features
    max_thickness: Optional[float] = None
    max_camber: Optional[float] = None
    chord_length: Optional[float] = None
    aspect_ratio: Optional[float] = None
    leading_edge_radius: Optional[float] = None
    trailing_edge_angle: Optional[float] = None
    thickness_to_chord_ratio: Optional[float] = None

    max_thickness_position: Optional[float] = (
        None  # Chord position (0-1) where max thickness occurs
    )
    max_camber_position: Optional[float] = (
        None  # Chord position (0-1) where max camber occurs
    )

    # Distribution data
    thickness_distribution: Optional[str] = None
    camber_distribution: Optional[str] = None
    normalized_chord: Optional[str] = None
