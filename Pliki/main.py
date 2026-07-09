"""
main.py — Punkt wejścia do symulacji Artemis 2
===============================================
Uruchom ten plik żeby zobaczyć symulację:
  python main.py

─────────────────────────────────────────────
  STRUKTURA PROJEKTU
─────────────────────────────────────────────
  main.py        ← TEN PLIK — łączy wszystkie moduły
  physics.py     ← stałe fizyczne, grawitacja, pozycje planet
  control.py     ← klasa ManeuverPhase, MissionControl, przykłady sterowania
  simulator.py   ← całkowanie RK4, klasa Simulator, SimulationResult
  visualizer.py  ← dwa okna matplotlib (animacja + wykresy)

─────────────────────────────────────────────
  JAK ZMIENIĆ STEROWANIE (napisać własne)
─────────────────────────────────────────────
  1. W pliku control.py dopisz funkcję, np. make_moje_sterowanie()
  2. W main.py zmień jedną linię:
       ctrl = make_artemis2_control()
     na:
       ctrl = make_moje_sterowanie()
  3. Uruchom ponownie main.py

  Albo zdefiniuj fazy bezpośrednio tutaj (patrz przykład poniżej).

─────────────────────────────────────────────
  PARAMETRY DO DOSTROJENIA
─────────────────────────────────────────────
  speed_factor — prędkość animacji (sekund symulacji / sekunda animacji)
  t_total      — czas symulacji [s]
  dt           — krok całkowania [s] (mniejszy = dokładniej, wolniej)
  with_sun     — czy uwzględniać grawitację Słońca
"""

import numpy as np

from physics  import T_HOHM, DV_TLI, V_TLI, V_LEO, R_EARTH
from control import MissionControl, ManeuverPhase, make_artemis2_control, make_lunar_orbit_control
from simulator import Simulator
from visualizer import show_all


# ════════════════════════════════════════════════════
#  WYBIERZ STEROWANIE
# ════════════════════════════════════════════════════

def wybierz_sterowanie() -> MissionControl:
    """
    Zmień return na jedną z gotowych opcji, albo zdefiniuj własne fazy.

    Opcje gotowe:
      make_artemis2_control()      — przelot obok Księżyca i powrót
      make_lunar_orbit_control()   — wejście na orbitę Księżyca i okrążenie
    """

    # ── Opcja A: gotowa misja Artemis 2 (przelot obok Księżyca) ──
    # return make_artemis2_control()

    # ── Opcja B: wejście na orbitę Księżyca ──────────────────────
    # return make_lunar_orbit_control()

    # ── Opcja C: własne sterowanie ───────────────────────────────
    # Odkomentuj i dostosuj:
    #

    # Hohmann Transfers
    from physics import G, M_EARTH, R_EARTH, D_MOON, M_MOON, V_MOON

    r1 = R_EARTH + 200e3  # perygeum LEO [m]
    r2 = R_EARTH*10
    r3 = D_MOON  # apogeum do Księżyca [m]
    #r2 = R_EARTH*10  # apogeum do Księżyca [m]

    #Prędkości przed zmianą
    V_before1 = np.sqrt(G*M_EARTH/r1)
    V_after2 = np.sqrt(G*M_EARTH/r2)

    # prędkości na orbicie transferowej Hohmanna
    # Prędkości po zmianie
    V_after1 = np.sqrt(2 * G * M_EARTH * (1 / r1 - 1 / (r1 + r2)))
    V_before2 = np.sqrt(2 * G * M_EARTH * (1 / r2 - 1 / (r1 + r2)))

    # Różnica predkości
    delta_v1 = V_after1 - V_before1
    delta_v2 = V_after2 - V_before2

    # Obliczanie ile czasu potrwa dotarcie do orbity
    a1 = (r1 + r2) / 2  # półosie w metrach
    T_full1 = 2 * np.pi * np.sqrt(a1 ** 3 / (G * M_EARTH) )
    T_transfer1 = T_full1 / 2
    T_hours1 = T_transfer1 / 3600


    # Ile czasu aż przeleci polowe orbity kolowej o promieniu r2
    T_orbita_kolowa=2 * np.pi*r2/V_after2
    T_hours2=T_orbita_kolowa / 2
    T_hours2=T_hours2 / 3600
    print(T_hours2)

    # Ile zmienic predkosc zeby byc na elipsie r3
    V_before3 = V_after2
    V_after3 = np.sqrt(2 * G * M_EARTH * (1 / r2 - 1 / (r2 + r3)))
    delta_v3 = V_after3 - V_before3

    # Ile czasu do elipsy odleglosc r3
    a2 = (r2 + r3) / 2  # półosie w metrach
    T_full3 = 2 * np.pi * np.sqrt(a2 ** 3 / (G * M_EARTH) )
    T_transfer3 = T_full3 / 2
    T_hours3 = T_transfer3 / 3600

    # Jaka predkosc jest na orbicie r3 (przy księżycu)?
    V_before4 = np.sqrt(2 * G * M_EARTH * (1 / r3 - 1 / (r2 + r3)))

    #Predkosc jaka msui byc aby okrazac ksiezyc
    #V_after4=np.sqrt(G*M_MOON/(D_MOON-r3))

    #delta_v4 = V_after4 - V_before4

    #dodac jeszcze predkosc ksiezyca (relatywnie delta_v4)
    #delta_v4=delta_v4 + V_MOON

    from physics import MOON_PHASE_0, moon_pos, R_LEO
    # print(f"Księżyc startuje pod kątem: {angle_moon:.1f}° od osi +X")
    # print(f"Statek startuje pod kątem:  180.0° od osi +X (zawsze w (-R_LEO, 0))")
    # print(f"Kąt między statkiem a Księżycem: {180 - angle_moon:.1f}°")


    # Startuje na 180 stopni, czyli po lewej stronie a lądowanie będzie po prawej czyli 0 stopni
    # w czasie t_start + T_hours1 + T_orbita_kolowa + T_hours2

    #t_calk=t_start + T_hours1 + T_orbita_kolowa + T_hours2
    t_reszta = T_hours1 + T_hours2 + T_hours3

    T_moon_calk = 2 * np.pi * r3 / V_MOON
    T_moon_calk=T_moon_calk/3600

    T_ship = 2 * np.pi * r1 / V_before1
    T_ship=T_ship/3600

    MOON_PHASE_0_mod=MOON_PHASE_0 % (2*np.pi)
    print(MOON_PHASE_0)

    t_start=(MOON_PHASE_0_mod+2*np.pi/T_moon_calk * t_reszta) / (2*np.pi/T_ship - 2*np.pi/T_moon_calk)
    print(t_start)
    #t_start*=1.01


    return MissionControl([
        ManeuverPhase(
            t0=t_start * 3600,
            t1=t_start * 3600,
            thrust=0,
            impulse=True,
            delta_v=delta_v1,
            theta=None,
            name="TLI"
        ),
        ManeuverPhase(
            t0=(t_start + T_hours1) * 3600,
            t1=(t_start + T_hours1) * 3600,
            thrust=0,
            impulse=True,
            delta_v=delta_v2,
            theta=None,
            name="TLI"
        ),
        ManeuverPhase(
            t0=(t_start + T_hours1 + T_hours2) * 3600,
            t1=(t_start + T_hours1 + T_hours2) * 3600,
            thrust=0,
            impulse=True,
            delta_v=delta_v3,
            theta=None,
            name="TLI"
        ),
        # ManeuverPhase(
        #     t0=(t_start + T_hours1 + T_hours2 + T_hours3) * 3600,
        #     t1=(t_start + T_hours1 + T_hours2 + T_hours3) * 3600,
        #     thrust=0,
        #     impulse=True,
        #     delta_v=V_MOON-V_before4,
        #     theta=None,
        #     name="TLI"
        # )
        # ManeuverPhase(
        #     # t0=19.5 * 3600,
        #     # t1=19.5 * 3600 + delta_t,
        #     t0=15.1 * 3600,
        #     t1=15.1 * 3600 + delta_t,
        #     thrust=thrust2,
        #     theta=None,
        #     name="Hohmann LEO → Księżyc"
        # ),

        # ManeuverPhase(
        #     t0=53.7 * 3600,
        #     t1=53.7 * 3600 + delta_t,
        #     thrust=thrust3,
        #     theta=None,
        #     name="Hohmann LEO → Księżyc"
        # ),
        #
        # ManeuverPhase(
        #     t0=184 * 3600,
        #     t1=184 * 3600 + delta_t,
        #     thrust=thrust4,
        #     theta=None,
        #     name="Hohmann LEO → Księżyc"
        # ),
    ])


# ════════════════════════════════════════════════════
#  PARAMETRY SYMULACJI
# ════════════════════════════════════════════════════
T_TOTAL      = 86400 * 100    # czas symulacji [s]  →  100 dni
DT           = 60.0         # krok RK4 [s]  →  1 minuta
WITH_SUN     = False         # czy uwzględniać Słońce
SPEED_FACTOR = 100000         # prędkość animacji
                            #   2000 → animacja ~7 min
                            #   4000 → animacja ~3.5 min
                            #    500 → animacja ~30 min (szczegółowa)
# ════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Wybierz sterowanie
    ctrl = wybierz_sterowanie()

    # 2. Utwórz symulator i uruchom
    sim    = Simulator(control=ctrl, dt=DT, with_sun=WITH_SUN)
    result = sim.run(t_total=T_TOTAL)

    # 3. Wyświetl animację i wykresy (dwa osobne okna)
    show_all(result, speed_factor=SPEED_FACTOR)
