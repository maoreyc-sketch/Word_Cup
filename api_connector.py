"""
api_connector.py
================
Capa de APIs (opcional). Trae datos en vivo de API-Football para ajustar
predicciones por bajas/lesiones. Está DESACOPLADA del motor: si la API falla,
el modelo sigue funcionando con la data histórica pura.

Nota de diseño: identificar selecciones nacionales por nombre en API-Football
es frágil (alias, idiomas). Por eso el conector cachea los IDs y degrada con
elegancia (modificador = 1.0) ante cualquier fallo, sin tumbar la predicción.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

# Cuántas bajas se consideran "significativas" y cuánto penalizan.
INJURY_THRESHOLD = 3
INJURY_PENALTY = 0.95  # -5% de rendimiento si hay muchas bajas
NEUTRAL_MODIFIER = 1.0


class FootballAPIConnector:
    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            "x-rapidapi-host": "v3.football.api-sports.io",
            "x-rapidapi-key": self.api_key,
        }
        self._id_cache: dict[str, int | None] = {}  # evita requests repetidos

    def _get(self, endpoint: str, params: dict) -> dict | None:
        """GET con manejo de errores específico. Devuelve el JSON o None."""
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = requests.get(
                url, headers=self.headers, params=params, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning("Timeout consultando %s", endpoint)
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP %s en %s", e.response.status_code, endpoint)
        except requests.exceptions.RequestException as e:
            logger.warning("Error de red en %s: %s", endpoint, e)
        except ValueError:  # JSON inválido
            logger.warning("Respuesta no-JSON en %s", endpoint)
        return None

    def get_team_id(self, team_name: str) -> int | None:
        """ID interno de la API para una selección (con caché)."""
        if team_name in self._id_cache:
            return self._id_cache[team_name]

        data = self._get("teams", {"name": team_name})
        team_id = None
        if data and data.get("response"):
            team_id = data["response"][0]["team"]["id"]

        self._id_cache[team_name] = team_id
        return team_id

    def get_live_injuries_modifier(self, team_name: str) -> float:
        """
        Modificador de rendimiento según bajas actuales.
        1.0 = sin cambios; < 1.0 = penalización. Nunca lanza excepción.
        """
        team_id = self.get_team_id(team_name)
        if not team_id:
            return NEUTRAL_MODIFIER

        data = self._get("injuries", {"team": team_id})
        if not data:
            return NEUTRAL_MODIFIER

        injuries = data.get("response", [])
        if len(injuries) >= INJURY_THRESHOLD:
            logger.info("%s tiene %d bajas: aplico penalización", team_name, len(injuries))
            return INJURY_PENALTY
        return NEUTRAL_MODIFIER


if __name__ == "__main__":
    # Sin clave válida, debe degradar a 1.0 sin romperse.
    logging.basicConfig(level=logging.INFO)
    api = FootballAPIConnector(api_key="CLAVE_DE_PRUEBA")
    print("Modificador Colombia:", api.get_live_injuries_modifier("Colombia"))
