"""
schemas.py — Contrato de datos de la API (Pydantic).

Pydantic valida automáticamente cada request entrante contra estos tipos.
Si el JSON no encaja (falta un campo, un tipo está mal), FastAPI responde
422 Unprocessable Entity ANTES de que nuestro código de negocio se ejecute.
Esto es lo que se llama "validar en el borde" (validate at the edge).
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class MovieInput(BaseModel):
    """Datos crudos de una película, tal como los rellena el usuario en el formulario."""

    budget: float = Field(..., gt=0, description="Presupuesto en USD", examples=[50_000_000])
    runtime: float = Field(..., gt=0, description="Duración en minutos", examples=[110])
    release_year: int = Field(..., ge=1900, le=2100, examples=[2026])
    season: str = Field(
        ...,
        description="Una de: summer, christmas, spring, off_season",
        examples=["summer"],
    )
    genres: List[str] = Field(
        default_factory=list,
        description="Lista de géneros en minúsculas y con guión bajo, p.ej. 'science_fiction'",
        examples=[["action", "adventure"]],
    )
    director: Optional[str] = Field(None, examples=["Christopher Nolan"])
    production_company: Optional[str] = Field(None, examples=["Warner Bros."])
    cast: List[str] = Field(
        default_factory=list,
        description="Hasta 3 nombres de reparto principal",
        examples=[["Tom Hanks", "Emma Stone"]],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "budget": 50_000_000,
                "runtime": 110,
                "release_year": 2026,
                "season": "summer",
                "genres": ["action", "science_fiction"],
                "director": "Christopher Nolan",
                "production_company": "Warner Bros.",
                "cast": ["Tom Hanks", "Emma Stone"],
            }
        }


class ShapContribution(BaseModel):
    """Una fila de la explicación SHAP: cuánto empujó una feature la predicción."""

    feature: str
    value: float = Field(..., description="Contribución SHAP: positivo empuja hacia éxito")


class PredictionOutput(BaseModel):
    """Lo que la API devuelve tras una predicción."""

    success_probability: float = Field(..., ge=0, le=1)
    top_features: List[ShapContribution]
    unknown_entities: List[str] = Field(
        default_factory=list,
        description="Nombres (director/cast/productora) que no estaban en el histórico "
                    "y por tanto usaron la media global como fallback",
    )


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool