"""
app_groups.py
=============
Mapeo oficial de los 12 grupos del Mundial 2026 (formato FIFA confirmado:
48 equipos, 12 grupos de 4). Los nombres se resuelven contra la base de datos
mediante los alias de data_manager, así que da igual la variante exacta.

Fuente única de verdad para el simulador de torneo y para la interfaz.
"""

GRUPOS = {
    "A": ["México", "República Checa", "Corea del Sur", "Sudáfrica"],
    "B": ["Canadá", "Bosnia y Herzegovina", "Catar", "Suiza"],
    "C": ["Brasil", "Marruecos", "Escocia", "Haití"],
    "D": ["Estados Unidos", "Paraguay", "Turquía", "Australia"],
    "E": ["Alemania", "Ecuador", "Costa de Marfil", "Curazao"],
    "F": ["Países Bajos", "Suecia", "Japón", "Túnez"],
    "G": ["Bélgica", "Irán", "Egipto", "Nueva Zelanda"],
    "H": ["España", "Uruguay", "Cabo Verde", "Arabia Saudita"],
    "I": ["Francia", "Senegal", "Noruega", "Irak"],
    "J": ["Argentina", "Austria", "Jordania", "Argelia"],
    "K": ["Portugal", "Colombia", "Uzbekistán", "RD Congo"],
    "L": ["Inglaterra", "Croacia", "Panamá", "Ghana"],
}
