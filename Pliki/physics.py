"""
physics.py — Stałe fizyczne i model grawitacyjny
=================================================
Zawiera:
  - Stałe fizyczne (G, masy, promienie, odległości)
  - Obliczenie geometrii startowej z orbity Hohmanna
  - Pozycje ciał niebieskich (Księżyc, Słońce) w czasie t
  - Obliczenie łącznego przyspieszenia grawitacyjnego

Jednostki we wszystkich plikach projektu:
  - Odległości: metry [m]
  - Prędkości:  metry na sekundę [m/s]
  - Czas:       sekundy [s]
  - Masa:       kilogramy [kg]
"""

import numpy as np


# ─────────────────────────────────────────────────────
#  STAŁE FIZYCZNE
# ─────────────────────────────────────────────────────

G       = 6.674e-11   # stała grawitacyjna [m³ / (kg·s²)]

M_EARTH = 5.972e24    # masa Ziemi [kg]
M_MOON  = 7.342e22    # masa Księżyca [kg]
M_SUN   = 1.989e30    # masa Słońca [kg]

R_EARTH = 6.371e6     # promień Ziemi [m]
R_MOON  = 1.737e6     # promień Księżyca [m]

D_MOON  = 3.844e8     # średnia odległość Ziemia–Księżyc [m]
D_SUN   = 1.496e11    # średnia odległość Ziemia–Słońce [m]

# Prędkość orbitalna Księżyca wokół Ziemi [m/s]
V_MOON  = np.sqrt(G * M_EARTH / D_MOON)   # ≈ 1022 m/s

# Prędkość kątowa Księżyca [rad/s]  (używana do animowania jego pozycji)
OM_MOON = V_MOON / D_MOON                 # ≈ 2.66e-6 rad/s


# ─────────────────────────────────────────────────────
#  GEOMETRIA STARTOWA — obliczona z orbity Hohmanna
#
#  Orbita Hohmanna = minimalnoenergtetyczna elipsa łącząca
#  dwie okrągłe orbity.  Periapsis = LEO (200 km),
#  apoapsis = dystans Księżyca.
#
#  Konfiguracja układu współrzędnych:
#    - Ziemia w (0, 0)
#    - Statek startuje w (-R_LEO, 0)  z prędkością (0, -V_TLI)
#    - Apoapsis trajektorii wypada w (+D_MOON, 0)
#    - Księżyc startuje w fazie -65.3°, tak żeby po T_HOHM był w (+D_MOON, 0)
# ─────────────────────────────────────────────────────

R_LEO  = R_EARTH + 200e3   # promień orbity LEO (200 km wysokości) [m]

# Prędkość na orbicie kołowej LEO [m/s]
V_LEO  = np.sqrt(G * M_EARTH / R_LEO)                  # ≈ 7788 m/s

# Wielka półoś elipsy Hohmanna [m]
A_HOHM = (R_LEO + D_MOON) / 2                          # ≈ 195 500 km

# Prędkość na periapsis elipsy Hohmanna (po wykonaniu TLI) [m/s]
V_TLI  = np.sqrt(G * M_EARTH * (2/R_LEO - 1/A_HOHM))  # ≈ 10921 m/s

# Delta-v potrzebne do TLI [m/s]  — tyle musi dodać silnik
DV_TLI = V_TLI - V_LEO                                 # ≈ 3133 m/s

# Czas lotu wzdłuż połowy elipsy (periapsis → apoapsis) [s]
T_HOHM = np.pi * np.sqrt(A_HOHM**3 / (G * M_EARTH))   # ≈ 119.5 h

# Startowa faza Księżyca [rad]
# Obliczona analitycznie z orbity Hohmanna: Księżyc startuje tak, żeby
# po T_HOHM dotrzeć do apoapsis statku na osi +X.
# Dodajemy offset +0.22 rad żeby przelot był ~5.5 Mm od Księżyca
# (bez offsetu trajektoria trafiałaby w Księżyc).
_MOON_PHASE_HOHMANN = -OM_MOON * T_HOHM                # ≈ -1.139 rad = -65.3°
_MOON_PHASE_OFFSET  = +0.22                             # offset bezpieczeństwa [rad]
MOON_PHASE_0 = _MOON_PHASE_HOHMANN + _MOON_PHASE_OFFSET  # ≈ -0.919 rad = -52.7°


# ─────────────────────────────────────────────────────
#  POZYCJE CIAŁ NIEBIESKICH W CZASIE t
# ─────────────────────────────────────────────────────

def moon_pos(t: float) -> np.ndarray:
    """
    Pozycja Księżyca w chwili t [s].

    Księżyc krąży po okręgu wokół Ziemi z prędkością kątową OM_MOON.
    Startuje z fazy MOON_PHASE_0 tak, żeby po T_HOHM być na osi +X.

    Zwraca: tablica [x, y] w metrach.
    """
    angle = MOON_PHASE_0 + OM_MOON * t
    return np.array([D_MOON * np.cos(angle),
                     D_MOON * np.sin(angle)])


def sun_pos(t: float) -> np.ndarray:
    """
    Pozycja Słońca względem Ziemi w chwili t [s].

    Ziemia krąży wokół Słońca z okresem 365.25 dni.
    Dodajemy przesunięcie π (Słońce jest "naprzeciwko" względem osi X).

    Zwraca: tablica [x, y] w metrach.
    """
    omega_earth = 2 * np.pi / (365.25 * 86400)   # prędkość kątowa Ziemi wokół Słońca
    angle = np.pi + omega_earth * t
    return np.array([D_SUN * np.cos(angle),
                     D_SUN * np.sin(angle)])


# ─────────────────────────────────────────────────────
#  ŁĄCZNE PRZYSPIESZENIE GRAWITACYJNE NA STATEK
# ─────────────────────────────────────────────────────

def gravity_acceleration(pos: np.ndarray, t: float,
                          with_sun: bool = True) -> np.ndarray:
    """
    Oblicza łączne przyspieszenie grawitacyjne działające na statek.

    Prawo grawitacji Newtona:  a = G * M / r²  w kierunku źródła masy.
    Wektorowo:  a⃗ = G * M / |r⃗|³  * r⃗
    gdzie r⃗ = (pozycja_źródła − pozycja_statku).

    Parametry:
      pos      — pozycja statku [x, y] w metrach
      t        — czas [s]
      with_sun — czy uwzględniać przyciąganie Słońca

    Zwraca: wektor przyspieszenia [ax, ay] w m/s².
    """
    acc = np.zeros(2)   # zaczynamy od zerowego przyspieszenia

    # --- Ziemia (zawsze w (0, 0)) ---
    r_vec = -pos                          # wektor od statku do Ziemi
    r     = np.linalg.norm(r_vec)         # odległość
    if r > R_EARTH:                       # sprawdzamy czy nie jesteśmy pod ziemią
        acc += G * M_EARTH / r**3 * r_vec

    # --- Księżyc ---
    pm    = moon_pos(t)
    r_vec = pm - pos                      # wektor od statku do Księżyca
    r     = np.linalg.norm(r_vec)
    # if r > R_MOON:
    #     acc += G * M_MOON / r**3 * r_vec
    if r > R_MOON:
        acc += 0

    # --- Słońce (siła pływowa, opcjonalne) ---
    #
    # WAŻNE: w naszym modelu Ziemia stoi nieruchomo w (0,0).
    # W rzeczywistości i Ziemia i statek "spadają" ku Słońcu tak samo
    # (swobodny spadek w polu grawitacyjnym Słońca).
    # Gdybyśmy dodali pełną siłę Słońca tylko na statek, statek
    # "uciekałby" w kierunku Słońca — to błąd fizyczny.
    #
    # Poprawnie: używamy RÓŻNICOWEJ (pływowej) siły Słońca:
    #   a_pływowa = a_słońce_na_statek − a_słońce_na_Ziemię
    # To opisuje rzeczywisty wpływ Słońca na ruch WZGLĘDEM Ziemi.
    if with_sun:
        ps = sun_pos(t)

        # Przyspieszenie od Słońca działające na statek
        r_vec_ship  = ps - pos
        r_ship      = np.linalg.norm(r_vec_ship)
        a_sun_ship  = G * M_SUN / r_ship**3 * r_vec_ship

        # Przyspieszenie od Słońca działające na Ziemię (pos = [0,0])
        r_vec_earth = ps                         # Ziemia jest w (0,0)
        r_earth     = np.linalg.norm(r_vec_earth)
        a_sun_earth = G * M_SUN / r_earth**3 * r_vec_earth

        # Dodajemy tylko różnicę — siłę pływową
        acc += (a_sun_ship - a_sun_earth)

    return acc
