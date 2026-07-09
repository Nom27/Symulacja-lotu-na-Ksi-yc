# Symulacja lotu na Księżyc

To mój prywatny projekt: symulacja 2D lotu statku kosmicznego z Ziemi w stronę Księżyca, oparta o numeryczne całkowanie równań ruchu (metoda RK4) w polu grawitacyjnym Ziemi, z możliwością włączenia też grawitacji Księżyca i pływowego wpływu Słońca. Projekt pozwala zaplanować i przetestować własne manewry silnikowe, a potem obejrzeć wynik jako animację i wykresy telemetrii.

## Struktura projektu

| Plik | Rola |
|---|---|
| `main.py` | Punkt wejścia — ustawia parametry symulacji, definiuje sterowanie i uruchamia całość |
| `physics.py` | Stałe fizyczne, model grawitacji, pozycje Księżyca/Słońca w funkcji czasu |
| `control.py` | Definicja manewrów silnikowych (`ManeuverPhase`, `MissionControl`) + gotowe przykłady |
| `simulator.py` | Całkowanie RK4, wykrywanie kolizji z Ziemią/Księżycem, zapis pełnej historii trajektorii |
| `visualizer.py` | Wizualizacja: animacja lotu + wykresy telemetrii (matplotlib) |
| `plan.png`, `plan2.png` | Moje odręczne notatki/obliczenia (rysunki w Paint) z planowaniem trajektorii, zanim trafiły do kodu |

## Uruchomienie

```bash
python main.py
```

Odpalenie `main.py` uruchamia dokładnie tę symulację, którą zaplanowałem na rysunkach (`plan.png`, `plan2.png`) i zaimplementowałem w funkcji `wybierz_sterowanie()`. Po uruchomieniu otwierają się dwa okna matplotlib:

- **Animacja trajektorii** — statek (biała kropka), Ziemia, poruszający się Księżyc, ślad lotu, wektor prędkości (zielony) i wektor ciągu silnika (pomarańczowy), panel HUD z telemetrią oraz przycisk Start/Pause.

<img width="399" height="401" alt="symulacja" src="https://github.com/user-attachments/assets/981841b1-be32-431e-8c97-f1fa25d0e19a" />

- **Wykresy telemetrii** — odległość od Ziemi i od Księżyca w czasie, prędkość statku, oraz przyspieszenie od silnika, z zaznaczonymi kolorowo fazami manewrów.

<img width="659" height="461" alt="wykresy" src="https://github.com/user-attachments/assets/3590cabb-a93a-4b56-b290-069c6eb4fbbc" />

## Idea trajektorii (to, co jest na `plan.png` / `plan2.png`)

Cel: dolecieć do Księżyca po trasie zbudowanej z **transferów Hohmanna** — czyli najbardziej ekonomicznych energetycznie elips przejściowych między dwiema orbitami kołowymi. Dokładnie to realizuje kod:

1. **Start → pierwsza elipsa.** Statek startuje z orbity o promieniu `r1` (LEO) i w jednym impulsie (`delta_v1`) dostaje prędkość, która wprowadza go na elipsę Hohmanna sięgającą do promienia `r2` (orbita pośrednia, w kodzie `r2 = 10 × R_EARTH`).
2. **Krążenie po okręgu `r2`.** Po dotarciu do apogeum tej elipsy drugi impuls (`delta_v2`) robi orbitę kołową — statek leci po okręgu o promieniu `r2` i "czeka" tam obliczony czas, aż geometria (kąt między statkiem a Księżycem) będzie odpowiednia do dalszego lotu.
3. **Druga elipsa → w stronę Księżyca.** Kolejny impuls (`delta_v3`) wprowadza statek na drugą elipsę Hohmanna, tym razem sięgającą do `r3 = D_MOON`, czyli praktycznie do orbity Księżyca.
4. **Docelowy okrąg przy Księżycu.** Ostatni krok to nadanie prędkości kołowej na promieniu `r3`, żeby statek zamiast przelecieć przez apogeum "na styk" — krążył tam, gdzie spotka Księżyc.

Najtrudniejszą częścią całego planowania jest **synchronizacja w czasie** pozycji statku i Księżyca: trzeba tak dobrać moment startu (`t_start`), żeby po przejściu przez wszystkie fazy (dwie elipsy + czas na okręgu `r2`) statek i Księżyc spotkały się w tym samym miejscu i czasie. Stąd obliczenia okresów orbitalnych (`T_s`, `T_k`), kątów fazowych (`α_k`, `α_start`) i równania na `t_start` widoczne na rysunkach i przełożone na kod w `main.py` (m.in. `MOON_PHASE_0` z `physics.py`, `t_start` liczony z warunku zgodności kątowej Księżyca i statku).

## Silnik i manewry (`control.py`)

`MissionControl` przechowuje listę faz `ManeuverPhase`, z których każda może być:
- **impulsowa** (`impulse=True`, `delta_v=...`) — natychmiastowa zmiana prędkości w danym momencie (tak działa opisany wyżej Hohmann w `main.py`),
- **ciągła** (`thrust`, `t0`–`t1`) — silnik działa przez określony czas z zadanym przyspieszeniem, w kierunku stałego kąta (`theta`) albo automatycznie prograde/retrograde (`theta=None`).

W pliku znajdują się też gotowe, alternatywne przykłady sterowania: `make_artemis2_control()` (prosty przelot obok Księżyca z jednym burnem powrotnym TEI) oraz `make_lunar_orbit_control()` (wejście na orbitę Księżyca — LOI + korekta + TEI). Dzięki takiej strukturze można zaprogramować dowolną inną trajektorię, nie tylko wariant z rysunków.

## Fizyka (`physics.py`) i całkowanie (`simulator.py`)

Grawitacja liczona jest z prawa Newtona dla Ziemi (zawsze) i opcjonalnie dla Słońca (jako siła pływowa względem Ziemi, żeby układ współrzędnych z nieruchomą Ziemią w (0,0) nie "uciekał"). Przyciąganie Księżyca jest w kodzie obecne, ale obecnie wyłączone (`acc += 0`) — łatwo je włączyć z powrotem. `Simulator` całkuje równania ruchu metodą RK4 z krokiem `dt`, obsługuje impulsowe zmiany prędkości w odpowiednim momencie oraz wykrywa kolizje ze statkiem/Ziemią/Księżycem, przerywając symulację i wypisując log.

## Parametry do dostrojenia (`main.py`)

- `T_TOTAL` — całkowity czas symulacji,
- `DT` — krok całkowania (mniejszy = dokładniej, wolniej),
- `WITH_SUN` — czy uwzględniać grawitację Słońca,
- `SPEED_FACTOR` — mnożnik prędkości animacji.

## Własne sterowanie

Nowy manewr definiuje się jako `ManeuverPhase` (impulsowy `delta_v` albo ciągły `thrust` w przedziale `t0`–`t1`) i dodaje do listy przekazywanej do `MissionControl`. Szczegółowa instrukcja z przykładami znajduje się w komentarzach na górze `control.py`.
