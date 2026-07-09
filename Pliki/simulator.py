"""
simulator.py — Całkowanie równań ruchu (RK4)
=============================================
Zawiera klasę Simulator, która:
  1. Przyjmuje obiekt MissionControl (skąd bierze przyspieszenie silnika)
  2. Całkuje równania ruchu metodą Runge-Kutta 4. rzędu
  3. Przechowuje pełną historię trajektorii (pozycje, prędkości, czas)
  4. Wykrywa kolizje z Ziemią i Księżycem

Stan symulacji:
  Wektor stanu y = [x, y, vx, vy]  (4 liczby)

Równania ruchu:
  dx/dt  = vx
  dy/dt  = vy
  dvx/dt = a_grav_x + a_thrust_x
  dvy/dt = a_grav_y + a_thrust_y

gdzie a_grav pochodzi z physics.py, a a_thrust z control.py.
"""

import numpy as np
from physics import (
    G, M_EARTH, R_EARTH, R_MOON,
    R_LEO, V_TLI,
    gravity_acceleration, moon_pos
)
from control import MissionControl


class SimulationResult:
    """
    Przechowuje wyniki symulacji jako tablice numpy.

    Wszystkie tablice mają tę samą długość N (liczba kroków + 1).
    Indeks 0 = warunki startowe.

    Atrybuty:
      times      — czasy kolejnych kroków [s],             shape: (N,)
      positions  — pozycje [x, y] [m],                     shape: (N, 2)
      velocities — prędkości [vx, vy] [m/s],               shape: (N, 2)
      moon_traj  — pozycja Księżyca w każdym kroku [m],    shape: (N, 2)
      thrust_log — wielkość przyspieszenia silnika [m/s²], shape: (N,)
      phases     — lista faz manewrowych (z MissionControl)
    """

    def __init__(self, times, positions, velocities, moon_traj,
                 thrust_log, phases):
        self.times      = times
        self.positions  = positions
        self.velocities = velocities
        self.moon_traj  = moon_traj
        self.thrust_log = thrust_log
        self.phases     = phases

    # ── wygodne właściwości obliczane na żądanie ──────────────────

    @property
    def dist_from_earth(self) -> np.ndarray:
        """Odległość od Ziemi w każdym kroku [m]."""
        return np.linalg.norm(self.positions, axis=1)

    @property
    def dist_from_moon(self) -> np.ndarray:
        """Odległość od Księżyca w każdym kroku [m]."""
        return np.linalg.norm(self.positions - self.moon_traj, axis=1)

    @property
    def speed(self) -> np.ndarray:
        """Prędkość (skalar) w każdym kroku [m/s]."""
        return np.linalg.norm(self.velocities, axis=1)

    @property
    def t_hours(self) -> np.ndarray:
        """Czas w godzinach (wygodne do wykresów)."""
        return self.times / 3600

    def closest_moon_approach(self) -> tuple:
        """
        Zwraca (minimalna_odległość [m], czas [s]) dla najbliższego przelotu Księżyca.
        """
        dm   = self.dist_from_moon
        imin = int(np.argmin(dm))
        return dm[imin], self.times[imin]

    def print_stats(self):
        """Wypisuje statystyki misji na konsolę."""
        min_dm, t_min = self.closest_moon_approach()
        print("\n" + "=" * 54)
        print("  STATYSTYKI MISJI")
        print("=" * 54)
        print(f"  Czas symulacji:          {self.times[-1]/86400:.2f} dni "
              f"({self.times[-1]/3600:.1f} h)")
        print(f"  Maks. odl. od Ziemi:     "
              f"{self.dist_from_earth.max()/1e6:.0f} Mm  "
              f"({self.dist_from_earth.max()/3.844e8:.2f}× D_Moon)")
        print(f"  Min. odl. od Księżyca:   "
              f"{min_dm/1e6:.3f} Mm   (t = {t_min/3600:.1f} h)")
        print(f"  Maks. prędkość:          "
              f"{self.speed.max()/1e3:.2f} km/s")
        print(f"  Min. prędkość:           "
              f"{self.speed.min()/1e3:.2f} km/s")
        print("=" * 54 + "\n")


class Simulator:
    """
    Całkuje równania ruchu statku kosmicznego metodą RK4.

    Użycie:
      from control import make_artemis2_control
      ctrl = make_artemis2_control()

      sim = Simulator(control=ctrl, dt=60.0, with_sun=True)
      result = sim.run(t_total=86400 * 8)
      result.print_stats()
    """

    def __init__(self,
                 control:   MissionControl,
                 dt:        float = 10.0,
                 with_sun:  bool  = True):
        """
        Parametry:
          control   — obiekt MissionControl z zdefiniowanymi fazami manewrów
          dt        — krok czasowy całkowania [s]
                       60 s  = dobra dokładność, szybki czas obliczeń
                       10 s  = lepsza dokładność (wolniejsze obliczenia ~6×)
          with_sun  — czy uwzględniać grawitację Słońca
        """
        self.control  = control
        self.dt       = dt
        self.with_sun = with_sun

        # Warunki startowe:
        # Pozycja: (-R_LEO, 0) — po lewej stronie Ziemi
        # Prędkość: (0, -V_TLI) — w dół, prograde na tej orbicie
        # Geometria dobrana tak, żeby apoapsis trafił w Księżyc (patrz physics.py)
        # self.initial_state = np.array([-R_LEO, 0.0, 0.0, -V_TLI])
        from physics import T_HOHM, DV_TLI, V_TLI, V_LEO
        # self.initial_state = np.array([-R_LEO, 0.0, 0.0, -1.1*V_LEO])
        self.initial_state = np.array([-R_LEO, 0.0, 0.0, -1.0 * V_LEO])
        #self.initial_state = np.array([-R_LEO, 0.0, 0.0, -10918])
        ####################3 R_LEO = R_EARTH + 200e3

    def _equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Prawe strony równań ruchu — wywoływane przez krok RK4.

        Parametry:
          t     — aktualny czas [s]
          state — wektor stanu [x, y, vx, vy]

        Zwraca: pochodna stanu [vx, vy, ax, ay]
        """
        pos = state[:2]   # pozycja [x, y]
        vel = state[2:4]  # prędkość [vx, vy]

        # Przyspieszenie od grawitacji
        a_grav = gravity_acceleration(pos, t, self.with_sun)

        # Przyspieszenie od silnika (zero jeśli żaden manewr nie jest aktywny)
        a_thrust = self.control.thrust_acceleration(t, vel)

        # Łączne przyspieszenie
        a = a_grav + a_thrust

        # Zwracamy pochodną stanu: [dx/dt, dy/dt, dvx/dt, dvy/dt]
        return np.array([vel[0], vel[1], a[0], a[1]])

    @staticmethod
    def _rk4_step(f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        """
        Jeden krok metody Runge-Kutta 4. rzędu.

        Wzór:
          k1 = f(t,       y)
          k2 = f(t+dt/2,  y + dt/2 * k1)
          k3 = f(t+dt/2,  y + dt/2 * k2)
          k4 = f(t+dt,    y + dt   * k3)
          y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        Parametry:
          f  — funkcja f(t, y) zwracająca pochodną stanu
          t  — aktualny czas
          y  — aktualny wektor stanu
          dt — krok czasowy
        """
        k1 = f(t,        y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt,   y + dt   * k3)
        return y + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

    def run(self, t_total: float = 86400 * 8) -> SimulationResult:
        """
        Uruchamia symulację od t=0 do t=t_total.

        Parametry:
          t_total — całkowity czas symulacji [s]
                    86400 * 8 = 8 dni

        Zwraca: obiekt SimulationResult z pełną historią trajektorii.
        """
        n = int(t_total / self.dt)   # liczba kroków

        # Alokacja tablic wynikowych (szybsze niż append w pętli)
        times      = np.empty(n + 1)
        positions  = np.empty((n + 1, 2))
        velocities = np.empty((n + 1, 2))
        moon_traj  = np.empty((n + 1, 2))
        thrust_log = np.empty(n + 1)

        # Warunki startowe (krok 0)
        state = self.initial_state.copy()
        times[0]      = 0.0
        positions[0]  = state[:2]
        velocities[0] = state[2:4]
        moon_traj[0]  = moon_pos(0.0)
        thrust_log[0] = np.linalg.norm(
            self.control.thrust_acceleration(0.0, state[2:4])
        )

        print(f"Uruchamianie symulacji: {n} kroków × {self.dt:.0f} s = "
              f"{t_total/86400:.1f} dni ...")

        stop = n   # indeks ostatniego zapisanego kroku

        for i in range(n):
            t = times[i]
            # DO STEROWANIA PREDKOSCIĄ
            for phase in self.control.phases:
                if getattr(phase, "impulse", False) and not phase.executed:

                    if abs(t - phase.t0) <= self.dt:

                        vel = state[2:4]
                        v_norm = np.linalg.norm(vel)

                        # kierunek
                        if phase.theta is not None:
                            direction = np.array([
                                np.cos(phase.theta),
                                np.sin(phase.theta)
                            ])
                        else:
                            if v_norm == 0:
                                direction = np.array([1.0, 0.0])
                            else:
                                direction = vel / v_norm
                                if phase.delta_v < 0:
                                    direction = -direction

                        # 🚀 impuls zmieniający prędkość
                        state[2:4] += abs(phase.delta_v) * direction

                        phase.executed = True
                        print(f"[IMPULSE] {phase.name}: Δv={phase.delta_v:.1f} m/s")

            # Jeden krok RK4
            state = self._rk4_step(self._equations_of_motion, t, state, self.dt)
            t_new = t + self.dt

            # Zapisz krok
            times[i + 1]      = t_new
            positions[i + 1]  = state[:2]
            velocities[i + 1] = state[2:4]
            moon_traj[i + 1]  = moon_pos(t_new)
            thrust_log[i + 1] = np.linalg.norm(
                self.control.thrust_acceleration(t_new, state[2:4])
            )

            # ── Detekcja kolizji z Ziemią ───────────────────────────
            r_earth = np.linalg.norm(state[:2])
            if r_earth < R_EARTH:
                print(f"  [!] Kolizja z Ziemią przy t = {t_new/3600:.1f} h")
                stop = i + 1
                break

            # ── Detekcja bliskiego przelotu Księżyca ────────────────
            r_moon = np.linalg.norm(state[:2] - moon_traj[i + 1])
            if r_moon < R_MOON * 5:
                # R_MOON * 5 ≈ 8685 km — blisko, ale nie kolizja
                print(f"  [*] Bliski przelot Księżyca!  "
                      f"t = {t_new/3600:.1f} h   d = {r_moon/1e6:.3f} Mm")
            if r_moon < R_MOON:
                print(f"  [!] Kolizja z Księżycem przy t = {t_new/3600:.1f} h")
                stop = i + 1
                break

        print("  Gotowe.")

        # Przytnij tablice do faktycznej długości
        result = SimulationResult(
            times      = times[:stop + 1],
            positions  = positions[:stop + 1],
            velocities = velocities[:stop + 1],
            moon_traj  = moon_traj[:stop + 1],
            thrust_log = thrust_log[:stop + 1],
            phases     = self.control.phases,
        )
        result.print_stats()
        return result
