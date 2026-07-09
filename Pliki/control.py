"""
control.py — Sterowanie statkiem kosmicznym
============================================
Tu definiujesz SWOJE manewry silnikowe.

Klasa ManeuverPhase opisuje jeden manewr:
  - kiedy się zaczyna i kończy (t0, t1)
  - jak duże jest przyspieszenie silnika (thrust)
  - w jakim kierunku działa ciąg (theta lub "prograde"/"retrograde")

Klasa MissionControl zbiera listę faz i w każdej chwili t
oblicza wypadkowy wektor przyspieszenia od silnika.

──────────────────────────────────────────────────────
  JAK NAPISAĆ WŁASNE STEROWANIE:
──────────────────────────────────────────────────────

  from control import ManeuverPhase, MissionControl
  import numpy as np

  fazy = [
      ManeuverPhase(
          t0     = 0,           # start manewru [s]
          t1     = 300,         # koniec manewru [s]  (300 s = 5 min)
          thrust = 5.0,         # przyspieszenie [m/s²]
                                #   > 0 = przyspiesza w kierunku theta/prędkości
                                #   < 0 = hamuje (retrograde)
          theta  = np.pi/4,     # kąt ciągu [rad]:
                                #   0       = prawo (+X)
                                #   pi/2    = góra (+Y)
                                #   pi      = lewo (-X)
                                #   -pi/2   = dół (-Y)
                                #   None    = auto: wzdłuż lub przeciw prędkości
          name   = "mój burn"   # nazwa do wyświetlania w animacji
      ),
  ]
  sterowanie = MissionControl(fazy)

──────────────────────────────────────────────────────
  WSKAZÓWKA: Układ współrzędnych
──────────────────────────────────────────────────────
  Ziemia:  (0, 0)
  Statek startuje: (-R_LEO, 0),  prędkość: (0, -V_TLI)
    czyli leci w dół (-Y), zaraz zakręci w prawo (+X) ku Księżycowi.
  Przelot Księżyca: ~88 h po starcie, ok. (384, 0) Mm
──────────────────────────────────────────────────────
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ManeuverPhase:
    """
    Jeden manewr silnikowy — "burn".

    Pola:
      t0     — czas początku manewru [s]
      t1     — czas końca manewru [s]
      thrust — wartość przyspieszenia od silnika [m/s²]
                 > 0  =  przyspieszanie w kierunku theta lub prograde
                 < 0  =  hamowanie (retrograde)
      theta  — stały kąt kierunku ciągu [rad], mierzony od osi +X:
                 0        = prawo  (+X)
                 pi/2     = góra   (+Y)
                 pi       = lewo   (-X)
                 -pi/2    = dół    (-Y)
                 None     = automatyczny:
                              thrust > 0  → wzdłuż prędkości (prograde)
                              thrust < 0  → przeciw prędkości (retrograde)
      name   — nazwa manewru wyświetlana w animacji i na wykresach
    """
    t0:     float
    t1:     float
    thrust: float
    theta:  Optional[float]   # [rad] lub None = prograde/retrograde
    name:   str = "manewr"

    # Do sterowania prędkością
    impulse: bool = False
    delta_v: Optional[float] = None
    executed: bool = False

    def is_active(self, t: float) -> bool:
        """Czy manewr jest aktywny w chwili t?"""
        return self.t0 <= t <= self.t1

    def duration(self) -> float:
        """Czas trwania manewru [s]."""
        return self.t1 - self.t0

    def delta_v_approx(self) -> float:
        """
        Przybliżone delta-v manewru [m/s].
        delta_v = |thrust| × czas_trwania
        """
        return abs(self.thrust) * self.duration()

class MissionControl:
    """
    Zarządza wszystkimi fazami misji i oblicza przyspieszenie od silnika.

    Użycie:
      ctrl   = MissionControl(lista_faz)
      a_vec  = ctrl.thrust_acceleration(t, vel)  →  [ax, ay] [m/s²]
    """

    def __init__(self, phases: list):
        """
        Parametry:
          phases — lista obiektów ManeuverPhase opisujących manewry misji.
                   Pusta lista = lot balistyczny (bez silnika).
        """
        self.phases = phases
        self._print_summary()

    def _print_summary(self):
        """Wypisuje plan misji na konsolę przy tworzeniu obiektu."""
        print("\n" + "─" * 42)
        print("  PLAN MISJI")
        print("─" * 42)
        if not self.phases:
            print("  (brak manewrów — lot balistyczny)")
        for i, ph in enumerate(self.phases, 1):
            dir_str = (f"{np.degrees(ph.theta):.1f}°"
                       if ph.theta is not None
                       else "auto (pro/retrograde)")
            print(f"  Faza {i}: {ph.name}")
            print(f"    t = [{ph.t0/3600:.2f} h — {ph.t1/3600:.2f} h]  "
                  f"({ph.duration():.0f} s)")
            print(f"    ciąg  = {ph.thrust:+.2f} m/s²   "
                  f"Δv ≈ {ph.delta_v_approx():.0f} m/s")
            print(f"    kąt   = {dir_str}")
        print("─" * 42 + "\n")

    def thrust_acceleration(self, t: float, vel: np.ndarray) -> np.ndarray:
        """
        Oblicza wektor przyspieszenia od silnika w chwili t.

        Parametry:
          t   — czas [s]
          vel — aktualny wektor prędkości [vx, vy] [m/s]
                (potrzebny gdy theta=None, żeby znać kierunek prograde)

        Zwraca: wektor [ax, ay] [m/s²].
                Zero gdy żaden manewr nie jest aktywny.
        """
        for phase in self.phases:
            if not phase.is_active(t):
                continue   # ta faza nie jest aktywna — sprawdzamy następną

            if phase.theta is not None:
                # Stały kąt — niezależny od bieżącej prędkości
                direction = np.array([np.cos(phase.theta),
                                      np.sin(phase.theta)])
            else:
                # Kierunek automatyczny: prograde lub retrograde
                speed = np.linalg.norm(vel)
                if speed < 1.0:
                    return np.zeros(2)   # brak prędkości — nie można obliczyć
                direction = vel / speed  # wektor jednostkowy wzdłuż prędkości

            # Znak: dodatni = w kierunku direction, ujemny = przeciw (hamowanie)
            sign = 1.0 if phase.thrust >= 0 else -1.0
            return sign * abs(phase.thrust) * direction

        return np.zeros(2)   # żadna faza nie jest aktywna

    def active_phase(self, t: float) -> Optional[ManeuverPhase]:
        """Zwraca aktywną fazę w chwili t lub None."""
        for phase in self.phases:
            if phase.is_active(t):
                return phase
        return None


# ─────────────────────────────────────────────────────
#  GOTOWE PRZYKŁADY STEROWANIA
# ─────────────────────────────────────────────────────

def make_artemis2_control() -> MissionControl:
    """
    Sterowanie dla misji Artemis 2 — przelot obok Księżyca i powrót.

    Statek startuje z prędkością V_TLI (burn TLI już wykonany jako
    warunek startowy w Simulator — patrz simulator.py).

    Trajektoria jest prawie balistyczna — Księżyc mija ~5.5 Mm od statku.
    Jedyny burn to TEI (powrót), który skraca lot do Ziemi.

    Księżyc jest widoczny w animacji od t≈80 h i mija przy t≈88 h.
    """
    # TEI: burn powrotny ok. 25 h po przelocie Księżyca
    T_TEI = 113 * 3600    # 113 h po starcie

    return MissionControl([
        # Trans-Earth Injection — burn prograde przybliżający ku Ziemi
        ManeuverPhase(
            t0     = T_TEI,
            t1     = T_TEI + 200,
            thrust = 1.2,    # 200 s × 1.2 m/s² = 240 m/s Δv
            theta  = None,   # prograde (wzdłuż prędkości)
            name   = "TEI — Trans-Earth Injection"
        ),
    ])


def make_lunar_orbit_control() -> MissionControl:
    """
    ─────────────────────────────────────────────────────────────
    PRZYKŁAD: Statek wchodzi na orbitę Księżyca i okrąża go.
    ─────────────────────────────────────────────────────────────

    Prędkość orbitalna 100 km nad Księżycem:
      v_orb = sqrt(G × M_MOON / r) ≈ 1633 m/s

    Statek przylatuje z prędkością ~2.5 km/s względem Księżyca
    (w układzie inercjalnym ~1.0 km/s).
    LOI musi zredukować prędkość wzgl. Księżyca do ~1633 m/s.

    UWAGA: Ten przykład jest uproszczony — czas i siła burnów
    może wymagać dostrojenia do konkretnej fazy Księżyca.
    ─────────────────────────────────────────────────────────────
    """
    from physics import G, M_MOON, R_MOON

    r_orbit      = R_MOON + 100e3               # 100 km nad Księżycem
    v_orbit_moon = np.sqrt(G * M_MOON / r_orbit) # ≈ 1633 m/s

    # Statek jest blisko Księżyca ok. t = 88 h
    T_ARRIVAL = 86 * 3600   # 86 h — zaczynam LOI trochę przed przelotem

    return MissionControl([
        # LOI faza 1: silne hamowanie — statek wchodzi w sferę wpływu Księżyca
        # Δv ≈ 3 m/s² × 260 s ≈ 780 m/s — zmniejsza prędkość wzgl. Księżyca
        ManeuverPhase(
            t0     = T_ARRIVAL,
            t1     = T_ARRIVAL + 260,
            thrust = -3.0,
            theta  = None,   # retrograde (hamuj)
            name   = "LOI-1 — wejście na orbitę Księżyca"
        ),

        # LOI faza 2: drobna korekta kołowości orbity (2 h po LOI-1)
        ManeuverPhase(
            t0     = T_ARRIVAL + 7200,
            t1     = T_ARRIVAL + 7260,
            thrust = -0.3,
            theta  = None,
            name   = "LOI-2 — korekta orbity"
        ),

        # TEI: burn powrotny po ~3 okrążeniach (1 okrążenie ≈ 2.1 h)
        # Prograde — opuszcza sferę wpływu Księżyca ku Ziemi
        ManeuverPhase(
            t0     = T_ARRIVAL + 8 * 3600,
            t1     = T_ARRIVAL + 8 * 3600 + 280,
            thrust = 1.5,
            theta  = None,   # prograde
            name   = "TEI — Trans-Earth Injection"
        ),
    ])
