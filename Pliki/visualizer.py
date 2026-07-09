"""
visualizer.py — Wizualizacja symulacji
=======================================
Dwa osobne okna matplotlib:

  Okno 1 (AnimationWindow):
    - Animacja w czasie rzeczywistym
    - Statek (biała kropka) poruszający się po trajektorii
    - Zielona strzałka: wektor prędkości
    - Pomarańczowa strzałka: wektor ciągu silnika
    - Ruchomy Księżyc
    - Panel HUD z aktualną telemetrią
    - Wyświetlana nazwa aktywnego manewru

  Okno 2 (ChartWindow):
    - Wykres 1: odległość od Ziemi i od Księżyca vs. czas
    - Wykres 2: prędkość vs. czas
    - Wykres 3: przyspieszenie silnika vs. czas
    - Zaznaczone kolorowymi pasami: aktywne manewry

Użycie:
  from visualizer import show_all
  show_all(result, speed_factor=2000)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from physics import R_EARTH, R_MOON, D_MOON
from simulator import SimulationResult
from matplotlib.widgets import Button


# ─────────────────────────────────────────────────────
#  PALETA KOLORÓW (ciemny motyw kosmiczny)
# ─────────────────────────────────────────────────────
BG_COLOR    = '#0a0a1a'   # tło (prawie czarne, lekko niebieskie)
GRID_COLOR  = '#1a1a2e'   # linie siatki
TEXT_COLOR  = '#e0e0f0'   # podstawowy kolor tekstu

# Kolory manewrów (dla kolejnych faz)
PHASE_COLORS = ['#ff6600', '#cc44ff', '#00bbff', '#ffaa00', '#00ff88']

# Kolory elementów animacji
COLOR_EARTH    = '#1a6fcc'
COLOR_MOON_SRF = '#c8c4b8'
COLOR_TRAIL    = '#7733cc'
COLOR_VEL_ARR  = '#00ff88'
COLOR_THR_ARR  = '#ff6600'
COLOR_SHIP     = 'white'
COLOR_STARS    = 'white'


# ─────────────────────────────────────────────────────
#  OKNO 2: WYKRESY TELEMETRII
# ─────────────────────────────────────────────────────

class ChartWindow:
    """
    Statyczne okno z trzema wykresami telemetrii misji.
    Otwierać przed AnimationWindow (będzie w tle).
    """

    def __init__(self, result: SimulationResult):
        self.result = result
        self.fig = self._build()

    def _build(self) -> plt.Figure:
        r   = self.result
        t_h = r.t_hours                        # czas [h]
        d_m = r.dist_from_moon  / 1e6          # odległość od Księżyca [Mm]
        d_e = r.dist_from_earth / 1e6          # odległość od Ziemi [Mm]
        spd = r.speed           / 1e3          # prędkość [km/s]
        thr = r.thrust_log                     # ciąg [m/s²]

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), facecolor=BG_COLOR)
        try:
            fig.canvas.manager.set_window_title("Artemis 2 — wykresy telemetrii")
        except Exception:
            pass
        fig.suptitle("Artemis 2 — dane telemetryczne misji",
                     color=TEXT_COLOR, fontsize=13, fontweight='bold')

        self._style_axes(axes)

        # ── Wykres 1: Odległości ─────────────────────────────────
        ax = axes[0]
        ax.plot(t_h, d_e, color='#1a7adb', lw=1.6, label='od Ziemi')
        ax.plot(t_h, d_m, color=COLOR_MOON_SRF, lw=1.6, label='od Księżyca')
        ax.axhline(D_MOON/1e6, color='#444466', ls='--', lw=0.8, alpha=0.6,
                   label=f'Dystans Ziemia–Księżyc ({D_MOON/1e6:.0f} Mm)')

        # Zaznacz fazy manewrów kolorowymi pasami
        for ph, col in zip(r.phases, PHASE_COLORS):
            ax.axvspan(ph.t0/3600, ph.t1/3600, alpha=0.2, color=col,
                       label=ph.name.split(' — ')[0])

        ax.set_title("Odległości od ciał niebieskich", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Odległość [Mm]", color=TEXT_COLOR, fontsize=9)
        ax.legend(fontsize=7, facecolor='#111122', labelcolor=TEXT_COLOR, ncol=3)

        # ── Wykres 2: Prędkość ───────────────────────────────────
        ax = axes[1]
        ax.plot(t_h, spd, color='#00ff88', lw=1.6)
        for ph, col in zip(r.phases, PHASE_COLORS):
            ax.axvspan(ph.t0/3600, ph.t1/3600, alpha=0.2, color=col)
            # Etykieta manewru nad pasem
            mid = (ph.t0 + ph.t1) / 2 / 3600
            ax.text(mid, spd.max() * 0.93,
                    ph.name.split(' — ')[0],
                    ha='center', color=col, fontsize=8, fontweight='bold')
        ax.set_title("Prędkość statku", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Prędkość [km/s]", color=TEXT_COLOR, fontsize=9)

        # ── Wykres 3: Ciąg silnika ───────────────────────────────
        ax = axes[2]
        ax.fill_between(t_h, thr, color=COLOR_THR_ARR, alpha=0.3)
        ax.plot(t_h, thr, color='#ff8833', lw=1.4)
        ax.set_title("Przyspieszenie od silnika  |u₁(t)|",
                     color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("|u₁| [m/s²]", color=TEXT_COLOR, fontsize=9)

        plt.tight_layout()
        return fig

    def _style_axes(self, axes):
        """Stosuje jednolity styl do wszystkich podwykresów."""
        for ax in axes:
            ax.set_facecolor(BG_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.set_xlabel("Czas [h]", color=TEXT_COLOR, fontsize=9)
            ax.grid(True, color=GRID_COLOR, lw=0.6)
            for sp in ax.spines.values():
                sp.set_color(GRID_COLOR)


# ─────────────────────────────────────────────────────
#  OKNO 1: ANIMACJA TRAJEKTORII
# ─────────────────────────────────────────────────────

class AnimationWindow:
    """
    Okno z animacją trajektorii w czasie rzeczywistym.

    speed_factor — ile sekund symulacji = 1 sekunda animacji
      2000  → pełna animacja (8 dni) trwa ~7 min
      4000  → ~3.5 min  (szybszy podgląd)
      500   → ~30 min   (wolna, szczegółowa)
    """

    def __init__(self, result: SimulationResult, speed_factor: float = 2000):

        self.result = result
        self.speed_factor = speed_factor
        self.paused = True  # do button

        self.result       = result
        self.speed_factor = speed_factor

        # Przelicz indeksy klatek
        dt      = float(result.times[1] - result.times[0])
        INT_MS  = 33   # interwał między klatkami [ms] → ~30 fps
        spf     = max(1, int(speed_factor / dt * (INT_MS / 1000)))
        self._fidx = list(range(0, len(result.times), spf))
        if self._fidx[-1] != len(result.times) - 1:
            self._fidx.append(len(result.times) - 1)

        self.fig, self._ax = self._setup_figure()
        self._artists      = self._build_scene()
        self._anim         = self._make_animation(INT_MS)
        # zatrzymaj od razu, żeby nie zdążyło nic przeskoczyć
        self._anim.event_source.stop()

    # ── Konfiguracja figury ───────────────────────────────────────

    def _setup_figure(self):

        fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG_COLOR)
        try:
            fig.canvas.manager.set_window_title(
                "Artemis 2 — animacja trajektorii"
            )
        except Exception:
            pass

        ax.set_facecolor(BG_COLOR)
        ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

        # Gwiazdki w tle
        np.random.seed(7)
        sx = np.random.uniform(-D_MOON*1.3, D_MOON*1.3, 500)
        sy = np.random.uniform(-D_MOON*1.3, D_MOON*1.3, 500)
        ax.scatter(sx, sy, s=0.3, c=COLOR_STARS, alpha=0.3, zorder=0)

        # Granice widoku
        M = D_MOON * 1.25
        ax.set_xlim(-M, M)
        ax.set_ylim(-M, M)

        # Orbita Księżyca — przerywana elipsa
        phi = np.linspace(0, 2 * np.pi, 300)
        ax.plot(D_MOON * np.cos(phi), D_MOON * np.sin(phi),
                '--', color='#222244', lw=0.7, zorder=1)

        # Ziemia: halo (atmosfera) + dysk
        # ax.add_patch(Circle((0, 0), R_EARTH * 7,   color=COLOR_EARTH, alpha=0.18, zorder=2))
        # ax.add_patch(Circle((0, 0), R_EARTH * 5.5, color=COLOR_EARTH, zorder=3))
        ax.add_patch(Circle((0, 0), R_EARTH, color=COLOR_EARTH, alpha=0.18, zorder=2))
        ax.add_patch(Circle((0, 0), R_EARTH, color=COLOR_EARTH, zorder=3))
        ax.text(0, -R_EARTH * 11, 'Ziemia',
                ha='center', color='#88bbff', fontsize=8, zorder=4)

        # Tytuł i legenda
        ax.set_title("Symulacja misji Artemis 2",
                     color=TEXT_COLOR, fontsize=12, fontweight='bold')
        ax.legend(handles=[
            Line2D([0], [0], color=COLOR_VEL_ARR, lw=1.5, label='Wektor prędkości'),
            Line2D([0], [0], color=COLOR_THR_ARR, lw=2.0, label='Ciąg silnika'),
            Line2D([0], [0], color=COLOR_TRAIL,   lw=1.5, label='Trajektoria'),
        ], loc='lower right', fontsize=8,
           facecolor='#111133', labelcolor=TEXT_COLOR,
           framealpha=0.75, edgecolor='#333366')

        return fig, ax

    # ── Budowa elementów sceny ────────────────────────────────────

    def _build_scene(self) -> dict:
        """
        Tworzy wszystkie dynamiczne elementy graficzne.
        Zwraca słownik {nazwa: obiekt_matplotlib}.
        """
        ##########BUTTON#######
        # ── PRZYCISK START/PAUSE ──
        ax_button = plt.axes([0.8, 0.02, 0.12, 0.05])
        self.btn = Button(ax_button, 'Start/Pause')
        self.btn.on_clicked(self._toggle_pause)
        ######################

        ax = self._ax
        mt0 = self.result.moon_traj[0]   # startowa pozycja Księżyca

        # Księżyc: halo + dysk + etykieta
        # moon_c   = Circle(mt0, R_MOON * 8,  color=COLOR_MOON_SRF, zorder=3)
        # moon_h   = Circle(mt0, R_MOON * 10, color=COLOR_MOON_SRF, alpha=0.12, zorder=2)
        moon_c   = Circle(mt0, R_MOON,  color=COLOR_MOON_SRF, zorder=3)
        moon_h   = Circle(mt0, R_MOON, color=COLOR_MOON_SRF, alpha=0.12, zorder=2)
        ax.add_patch(moon_c)
        ax.add_patch(moon_h)
        moon_lbl = ax.text(mt0[0], mt0[1] + R_MOON * 14, 'Księżyc',
                           ha='center', color='#ddddcc', fontsize=8, zorder=4)

        # Ślad trajektorii (narasta w czasie)
        trail, = ax.plot([], [], '-', color=COLOR_TRAIL, lw=1.0, alpha=0.7, zorder=5)

        # Punkt statku
        ship, = ax.plot([], [], 'o', color=COLOR_SHIP, ms=5, zorder=10,
                        markeredgecolor='#aaaaff', markeredgewidth=0.7)

        # Strzałka wektora prędkości
        # (annotation z arrowprops, aktualizowana przez zmianę .xy i .set_position)
        vel_arr = ax.annotate('', xy=(0, 0), xytext=(0, 0), zorder=8,
                              arrowprops=dict(arrowstyle='->',
                                              color=COLOR_VEL_ARR,
                                              lw=1.5, mutation_scale=12))

        # Strzałka wektora ciągu silnika
        thr_arr = ax.annotate('', xy=(0, 0), xytext=(0, 0), zorder=8,
                              arrowprops=dict(arrowstyle='->',
                                              color=COLOR_THR_ARR,
                                              lw=2.2, mutation_scale=15))

        # Panel HUD (telemetria w lewym górnym rogu)
        hud = ax.text(
            0.02, 0.97, '',
            transform=ax.transAxes,
            color=TEXT_COLOR, fontsize=9, va='top',
            fontfamily='monospace', zorder=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=BG_COLOR,
                      edgecolor='#333355', alpha=0.75)
        )

        # Etykieta aktywnego manewru (prawy górny róg)
        phase_lbl = ax.text(
            0.98, 0.97, '',
            transform=ax.transAxes,
            color='#ffaa00', fontsize=9, va='top', ha='right',
            fontweight='bold', zorder=11
        )

        return dict(
            moon_c=moon_c, moon_h=moon_h, moon_lbl=moon_lbl,
            trail=trail, ship=ship,
            vel_arr=vel_arr, thr_arr=thr_arr,
            hud=hud, phase_lbl=phase_lbl,
        )

    # ── Krok animacji ─────────────────────────────────────────────

    def _update_frame(self, frame_no: int):

        #button
        if self.paused:
            return tuple(self._artists.values())

        """
        Wywoływana przez FuncAnimation dla każdej klatki.
        Aktualizuje pozycję wszystkich dynamicznych elementów.
        """
        i   = self._fidx[frame_no]
        r   = self.result
        p   = r.positions[i]      # pozycja statku [x, y]
        v   = r.velocities[i]     # prędkość statku [vx, vy]
        m   = r.moon_traj[i]      # pozycja Księżyca [x, y]
        thr = r.thrust_log[i]     # wartość ciągu [m/s²]
        t   = r.times[i]          # aktualny czas [s]

        a = self._artists   # skrót

        # ── Trajektoria (rośnie klatka po klatce) ────────────────
        a['trail'].set_data(r.positions[:i+1, 0], r.positions[:i+1, 1])

        # ── Pozycja statku ────────────────────────────────────────
        a['ship'].set_data([p[0]], [p[1]])

        # ── Księżyc ───────────────────────────────────────────────
        a['moon_c'].center = m
        a['moon_h'].center = m
        a['moon_lbl'].set_position((m[0], m[1] + R_MOON * 14))

        # ── Wektor prędkości (skalowany do stałej długości graficznej) ──
        spd = np.linalg.norm(v)
        VS  = D_MOON * 0.13   # długość strzałki na ekranie [m]
        if spd > 100:
            # Strzałka zaczyna w p i kończy w p + kierunek * VS
            a['vel_arr'].set_position(p)
            a['vel_arr'].xy = p + (v / spd) * VS
        else:
            # Zerowa prędkość — ukryj strzałkę (wskaż na siebie)
            a['vel_arr'].set_position(p)
            a['vel_arr'].xy = p

        # ── Wektor ciągu silnika ──────────────────────────────────
        TS = D_MOON * 0.14   # długość strzałki ciągu [m]
        if thr > 0.01 and spd > 100:
            # Wyznacz kierunek ciągu z aktywnej fazy
            active_ph = r.control.active_phase(t) if hasattr(r, 'control') else None
            # Jeśli brak dostępu do control, odtwarzamy kierunek z prędkości
            phase_obj = self._get_active_phase(t)
            if phase_obj is not None:
                if phase_obj.theta is not None:
                    td = np.array([np.cos(phase_obj.theta),
                                   np.sin(phase_obj.theta)])
                elif phase_obj.thrust >= 0:
                    td = v / spd          # prograde
                else:
                    td = -(v / spd)       # retrograde
                scale = min(thr / 9.0, 1.0) * TS
                a['thr_arr'].set_position(p)
                a['thr_arr'].xy = p + td * scale
            else:
                a['thr_arr'].set_position(p)
                a['thr_arr'].xy = p
        else:
            a['thr_arr'].set_position(p)
            a['thr_arr'].xy = p

        # ── Panel HUD ─────────────────────────────────────────────
        d_e = np.linalg.norm(p) / 1e6
        d_m = np.linalg.norm(p - m) / 1e6
        a['hud'].set_text(
            f"t        = {t/3600:7.1f} h\n"
            f"prędkość = {spd/1e3:7.2f} km/s\n"
            f"od Ziemi = {d_e:7.1f} Mm\n"
            f"od Ks.   = {d_m:7.1f} Mm\n"
            f"ciąg     = {thr:7.3f} m/s²"
        )

        # ── Nazwa aktywnego manewru ───────────────────────────────
        phase_obj = self._get_active_phase(t)
        a['phase_lbl'].set_text(
            f"▶ {phase_obj.name}" if phase_obj else ""
        )

        return tuple(a.values())

    def _get_active_phase(self, t: float):
        """Zwraca aktywną fazę w chwili t lub None."""
        for ph in self.result.phases:
            if ph.is_active(t):
                return ph
        return None

    # ── Tworzenie animacji ────────────────────────────────────────

    def _make_animation(self, interval_ms: int) -> animation.FuncAnimation:
        """Tworzy obiekt FuncAnimation."""
        return animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames   = len(self._fidx),
            interval = interval_ms,
            blit     = False,   # True jest szybsze, ale bywa wadliwe na Windows
            repeat   = False,
        )

    # Button
    def _toggle_pause(self, event):
        self.paused = not self.paused

        if self.paused:
            self._anim.event_source.stop()
        else:
            self._anim.event_source.start()


# ─────────────────────────────────────────────────────
#  FUNKCJA SKRÓTOWA: otwiera oba okna
# ─────────────────────────────────────────────────────

def show_all(result: SimulationResult, speed_factor: float = 2000):
    """
    Otwiera oba okna naraz i uruchamia pętlę zdarzeń matplotlib.

    Parametry:
      result       — wyniki z Simulator.run()
      speed_factor — ile sekund symulacji = 1 sekunda animacji
                     2000 → ~7 min  |  4000 → ~3.5 min  |  500 → ~30 min

    Okno wykresów otwiera się pierwsze (jest pod animacją).
    """
    charts  = ChartWindow(result)
    anim_w  = AnimationWindow(result, speed_factor=speed_factor)
    # Przechowujemy referencję do animacji żeby matplotlib jej nie usunął
    _anim_keep = anim_w._anim
    plt.show()
