# MK-BPX: balanced partition crossover for the p-median problem

Code, data, run-level logs and manuscript of the study
**"Balanced Partition Crossover for the p-Median Problem: A Multi-Kick Memetic Algorithm"**
(F. A. Rivero Sotelo, N. D. Montes Villalba, H. E. Hernández Riaño, J. M. López Pereira — Universidad de Córdoba, Colombia).

> **English quick start.** `pip install -r requirements.txt`, then `python tests/smoke_test.py`
> (≈1 min, must end with `ALL CHECKS PASSED`). Run the algorithm with
> `python run_mkbpx.py fl1400 --p 100 --time 10`. Regenerate every number, table and figure of the
> paper from the shipped logs with `python experiments/analyze.py`, `python experiments/analyze_sota.py`
> and `python experiments/analyze_orig.py`. Section 5 below maps every table and figure of the paper
> to the command that produces it. The rest of this file is in Spanish.

Este repositorio es autocontenido: no necesita conexión a internet para ejecutar el algoritmo, repetir
los experimentos propios o regenerar las cifras, tablas y figuras del artículo. La única pieza externa
es el programa original POPSTAR de Resende y Werneck, que es de sus autores y **no se redistribuye**:
se descarga y compila con un comando (sección 7).

---

## 1. Instalación

Requiere Python 3.10 o superior (el artículo usó Python 3.13, NumPy 2.5 y Numba 0.68).

```bash
git clone https://github.com/ferminriv20/HGA-EBMKI.git
cd HGA-EBMKI
python -m venv .venv && source .venv/bin/activate     # opcional
pip install -r requirements.txt
python tests/smoke_test.py
```

La prueba tarda cerca de un minuto y debe terminar con `ALL CHECKS PASSED`. La primera ejecución de
cualquier script compila el código con Numba y tarda unos segundos más. **Todos los comandos se
ejecutan desde la raíz del repositorio.**

## 2. Estructura

```
├── README.md, LICENSE, CITATION.cff, requirements.txt
├── run_mkbpx.py                    Ejecuta MK-BPX (o VNS) sobre una instancia
├── mkbpx/                          Los algoritmos
│   ├── core.py                     Búsqueda local, kick, cruce BPX (px) y bucle principal (run_main, run_main_ext)
│   ├── instances.py                Carga de instancias y valores de referencia
│   ├── whitaker.py                 Intercambio rápido de Whitaker (solo para medir velocidad)
│   ├── prototypes.py               Prototipos con varios élites usados durante el desarrollo
│   ├── popstar.py                  Reimplementación de PopStar sobre los mismos núcleos (2.ª campaña)
│   └── popstar_orig.py             Lanzador del programa POPSTAR original (3.ª campaña)
├── experiments/                    Todo lo que se ejecutó para el artículo (ver sección 5)
├── tools/
│   ├── build_orlib_matrices.py     Reconstruye las matrices de OR-Library desde los archivos originales
│   ├── export_pmm.py               Exporta una instancia al formato PMM de POPSTAR
│   └── get_popstar.sh              Descarga y compila el POPSTAR original
├── data/
│   ├── orlib/, orlib_raw/          pmed1–pmed40 (matrices y archivos originales)
│   ├── tsplib/                     Los siete archivos .tsp usados
│   └── reference/                  Límites de tiempo de OR-Library y corridas del estudio anterior
├── results/                        Registros de la campaña principal (VNS, MK-BPX, VNS20, MK-BPX4)
├── results_sota/                   Registros de la 2.ª campaña (PopStar reimplementado y MK-BPX)
├── results_orig/                   Registros de la 3.ª campaña (POPSTAR original, MK-BPX y VNS)
├── results_replay/                 Repetición exacta de la campaña principal (trazas completas)
├── third_party/popstar/            Aquí queda el ejecutable de POPSTAR (no se incluye)
├── paper/                          Manuscrito (main.tex, main.pdf) y todo lo que generan los análisis
└── tests/smoke_test.py             Comprobación rápida de la instalación
```

## 3. Ejecutar el algoritmo

```bash
python run_mkbpx.py pmed40                                  # instancia de OR-Library (p propio)
python run_mkbpx.py fl1400 --p 100 --time 10                # instancia TSPLIB incluida
python run_mkbpx.py u1432 --p 200 --algo VNS --seed 7 --runs 5
python run_mkbpx.py mi_instancia.tsp --p 50 --save solucion.txt
python run_mkbpx.py distancias.npy --p 30                   # matriz de distancias simétrica
```

| Opción | Significado | Valor por defecto |
|---|---|---|
| `--p` | Número de medianas (obligatorio salvo en `pmedX`) | el de la instancia |
| `--time` | Límite de tiempo por corrida, en segundos | 10 |
| `--algo` | `BPX` (MK-BPX), `VNS`, `VNS20` o `BPX4` | `BPX` |
| `--seed` | Semilla de la primera corrida; la corrida r usa semilla + r | 1 |
| `--runs` | Número de corridas | 1 |
| `--save` | Archivo donde se escribe la mejor solución (índices desde 0) | no se guarda |

Acepta un nombre conocido (`pmed1`–`pmed40` o una de las siete de `data/tsplib`), un archivo `.tsp`
con coordenadas, o una matriz de distancias en `.npy`. Cada cliente es también un sitio candidato.

Uso desde Python:

```python
from mkbpx import core, instances
C = instances.load_matrix('fl1400')            # matriz de distancias
order = core.build_order(C)                    # listas ordenadas, una vez por instancia
costo, solucion, stats, traza = core.run_main(
    C, order, 100,      # p
    1,                  # use_px: 1 = MK-BPX, 0 = VNS
    0,                  # neutral: dejar en 0
    100,                # k_max (p en el artículo)
    1,                  # copias élite (1 en el artículo)
    123,                # semilla
    0.0,                # valor objetivo (solo para registrar cuándo se alcanza)
    10.0)               # límite de tiempo en segundos
```

`traza` es la lista de pares (tiempo, costo) de cada mejora. `core.run_main_ext` es la misma búsqueda
con tres opciones más (detenerse tras un número de iteraciones, continuar hasta un mínimo de
iteraciones y registrar la iteración de cada mejora); se usa para repetir exactamente una corrida.

## 4. Las campañas experimentales

| Campaña | Pregunta | Algoritmos | Registros | Análisis |
|---|---|---|---|---|
| Principal | ¿Qué aporta el cruce a una búsqueda por perturbación? | VNS, MK-BPX, VNS20, MK-BPX4 (mismo código) | `results/` | `analyze.py` |
| Segunda | MK-BPX frente a PopStar con la misma búsqueda local y el mismo reloj | PopStar reimplementado, MK-BPX | `results_sota/` | `analyze_sota.py` |
| Tercera | MK-BPX y VNS frente al **programa POPSTAR original**, en la misma máquina y a igual tiempo, por bloques | POPSTAR (programa de los autores), MK-BPX, VNS | `results_orig/` | `analyze_orig.py` |
| Repetición | Repetición exacta de la campaña principal para completar sus trazas | VNS, MK-BPX | `results_replay/` | `complete_traces.py` |

Conjuntos de instancias (`SET` en los scripts): `S0` = 40 instancias de OR-Library (30 corridas);
`S1` = fl1400, u1432, rl1304 (10 s) y d2103 (20 s), 10 corridas; `S2` = pcb3038 (30 s), fl3795 (40 s)
y rl5934 (60 s), 5 corridas; `S3` = corridas con cinco veces el límite. Desarrollo: rl1304 y pcb3038.
Prueba: fl1400, u1432, d2103, fl3795 y rl5934.

## 5. Cómo reproducir cada resultado del artículo

Los registros con los que se escribió el artículo están incluidos, así que **regenerar todas las
cifras, tablas y figuras no requiere repetir ningún experimento**:

```bash
python experiments/analyze.py          # campaña principal  -> paper/numbers.tex, tables/tab_*.tex, images/fig_*.pdf
python experiments/analyze_sota.py     # segunda campaña    -> paper/numbers_sota.tex, tables/tab_sota_*.tex, images/fig_sota_*.pdf
python experiments/analyze_orig.py     # tercera campaña    -> paper/numbers_orig.tex, tables/tab_orig_*.tex, images/fig_orig_*.pdf
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Para repetir los experimentos, cada elemento del artículo se obtiene así (añadir `--out DIR` para no
sobrescribir los registros incluidos; las campañas se reanudan donde quedaron):

| Elemento del artículo | Comando | Duración (2 núcleos) |
|---|---|---|
| Verificación exhaustiva de BPX (sec. 4.2) | `python experiments/verify_bpx.py` | 5 min |
| Figura 1 (componentes en rl1304, p = 100) | `python experiments/fig_components.py 9 plot` | 1 min |
| Tabla 1 (velocidad de la búsqueda local) | `python experiments/speed_benchmark.py` | 3 min |
| Tabla 3 (configuraciones de desarrollo) | `python experiments/development.py` (ver su encabezado) | 20 min |
| Tabla 4 (ajuste de PopStar reimplementado) | `python experiments/development_popstar.py` | 15 min |
| Tablas 5–9, figuras 2–3 (efecto del cruce) | `campaign.py VNS S1` y `campaign.py BPX S1`; ídem `S2`; luego `analyze.py` | 75 + 70 min |
| Tabla 10 (corridas largas) | `campaign.py VNS S3` y `campaign.py BPX S3` | 85 min |
| Control ajustado y 4 élites (sec. 7.4) | `campaign.py VNS20 S1` y `campaign.py BPX4 S1` | 75 min |
| Tabla 11 (OR-Library) | `campaign.py VNS S0` y `campaign.py BPX S0` | 40 min |
| Sec. 7.6 (PopStar reimplementado) | sección 6 de este archivo | 4,5 h |
| Sec. 7.7 (POPSTAR original, igual tiempo) | sección 7 de este archivo | 5 h |
| Trazas completas y repetición exacta | sección 8 de este archivo | — |

`python experiments/run_all.py --out results_nuevos` ejecuta de una vez toda la campaña principal
(unas 6,5 horas) y deja cifras, tablas y figuras en `results_nuevos/paper/`. En cada línea de la tabla
los dos procesos (por ejemplo VNS y BPX) se lanzan a la vez, uno por núcleo, como en el artículo:

```bash
python experiments/campaign.py VNS S1 --out results_nuevos &
python experiments/campaign.py BPX S1 --out results_nuevos ; wait
python experiments/analyze.py results_nuevos/paper --results results_nuevos
```

## 6. Segunda campaña: PopStar reimplementado

MK-BPX frente a PopStar (GRASP con path-relinking de Resende y Werneck, 2004) reimplementado sobre los
mismos núcleos: misma búsqueda local, compilador, reloj y generador aleatorio, y el mismo límite de
tiempo. Los dos algoritmos se ejecutan a la vez, uno por núcleo:

```bash
python experiments/campaign_sota.py S1 --out results_sota & python experiments/campaign.py BPX S1 --out results_sota ; wait
#   (igual con S2, S0 y S3)
python experiments/popstar_fidelity.py fl1400 ; python experiments/popstar_fidelity.py orlib
cat results_sota/fidelity_fl1400.csv > results_sota/fidelity_popstar.csv
tail -n +2 results_sota/fidelity_orlib.csv >> results_sota/fidelity_popstar.csv
python experiments/verify_popstar.py        # path-relinking contra fuerza bruta
python experiments/analyze_sota.py
```

| Archivo | Contenido |
|---|---|
| `mkbpx/popstar.py` | Construcción *sample greedy*, path-relinking, conjunto élite, post-optimización, bucle con límite de tiempo |
| `experiments/campaign_sota.py SET` | Campaña de PopStar (S0–S3), mismos pares, tiempos y corridas que `campaign.py` |
| `experiments/popstar_fidelity.py` | PopStar tal como se publicó (un ciclo, sin límite) frente a los resultados publicados |
| `results_sota/` | Registros y trazas, ajuste (`dev_popstar.csv`), fidelidad y calibración |

## 7. Tercera campaña: el programa POPSTAR original

Aquí no se reimplementa nada: se ejecuta el programa de los autores, sin modificar, y se compara con
MK-BPX y VNS **en la misma máquina, con la misma carga y a igual tiempo transcurrido**.

**7.1 Obtener POPSTAR.** Se descarga de la página de los autores,
<https://mauricio.resende.info/popstar/downloads.html>, y se compila sin cambios:

```bash
tools/get_popstar.sh                        # descarga popstar.tar.gz y compila
tools/get_popstar.sh /ruta/popstar.tar.gz   # o usa una copia ya descargada
```

El ejecutable queda en `third_party/popstar/popstar` (o indique otro con la variable `POPSTAR_BIN`).
El artículo usó la versión `20041104`; el script avisa si el archivo descargado es distinto. El
binario Linux precompilado que ofrece la misma página es un ejecutable de 32 bits de 2002 que no
arranca en sistemas actuales (falla dentro del cargador dinámico de glibc 2.39), por lo que se
compila el código fuente de la misma distribución con `g++ -O3` (necesita `g++` y `make`).

**7.2 Instancias.** POPSTAR lee la misma matriz de distancias que usa MK-BPX, exportada a su formato
PMM. La campaña lo hace sola; a mano:

```bash
python tools/export_pmm.py fl1400 pmed40 --out data/pmm
third_party/popstar/popstar data/pmm/fl1400.pmm -p 100 -seed 1 -graspit 32 -elite 10
```

(`data/pmm/` ocupa varios GB y está excluido del repositorio.)

**7.3 Ejecutar la campaña** (unas 5 horas; necesita al menos dos núcleos; es reanudable):

```bash
experiments/run_campaign3.sh results_orig_nuevos
python experiments/analyze_orig.py results_orig_nuevos/paper --results results_orig_nuevos
```

El script llama a `experiments/campaign_blocked.py GRUPO` para los grupos `test_small` (fl1400,
u1432, d2103), `test_big` (fl3795, rl5934), `dev` (rl1304, pcb3038) y `orlib` (pmed1–pmed40).

**7.4 Diseño por bloques.** La velocidad de una máquina virtual compartida cambia entre un 10 y un
20 % a lo largo de las horas que dura una campaña (y al reiniciarse la máquina), tanto como las
diferencias que una comparación a igual tiempo debe resolver. Por eso todo lo que se compara se mide
dentro de un mismo bloque de pocos minutos. Para cada par instancia–p, siempre con dos procesos
activos, cada uno fijado a un núcleo, y alternando las corridas de cada algoritmo entre los núcleos:

| Paso | Qué se ejecuta |
|---|---|
| 1. POPSTAR publicado (`P32`) | 32 iteraciones multiarranque, 10 élites; semilla = corrida + 1 |
| 2. POPSTAR escalado (`PT`) | k = ⌊(0,9 T − o − b)/a⌋ iteraciones, con a, b, o medidos en el paso 1 (tiempo por iteración, post-optimización e inicialización). Solo si k ≥ 40; si no, el paso 1 ya usa el límite T. No se aplica a OR-Library |
| 3. MK-BPX y VNS | Las corridas de la campaña principal (mismas semillas), prolongadas hasta H = máx(T, 1,1 × corrida más larga de POPSTAR en el bloque). VNS solo en las instancias de prueba |

Un bloque se escribe en disco solo cuando termina; `blocks.csv` guarda su hora, duración, k, H y las
sondas de velocidad (una operación fija cronometrada en ambos núcleos al inicio y al final).

**7.5 Regla de igual tiempo.** POPSTAR no tiene límite de tiempo. Para cada par, *t* es el tiempo
transcurrido medio de sus corridas; el valor de una corrida de MK-BPX o VNS es el mejor que había
encontrado en el instante *t*, leído de su traza después de sumarle el tiempo de ordenar las listas
de sitios (`order_time.json`), porque el tiempo de POPSTAR incluye la inicialización de su oráculo de
distancias. El tiempo de POPSTAR se mide con reloj de pared desde que termina de leer la instancia
(`mkbpx/popstar_orig.py` explica cómo). Cada solución que entrega POPSTAR se reevalúa con la matriz
de este estudio y debe dar el valor que el programa informa.

**7.6 Control de perturbaciones.** Una corrida de POPSTAR cuyo tiempo transcurrido supera su tiempo
de CPU en más de 5 % + 0,15 s, o una corrida de MK-BPX o VNS cuyo tiempo de CPU es inferior al 98,5 %
del transcurrido, se repite (dos veces como máximo); las columnas `attempts` y `cpu_ratio` lo
registran. `results_orig/machine.txt` describe el equipo.

## 8. Trazas completas y repetición exacta de una corrida

Una corrida de VNS o MK-BPX queda determinada por su semilla; lo único que depende de la máquina es
el punto en que el límite de tiempo la corta. Por eso cualquier corrida registrada se puede repetir
exactamente en cualquier equipo:

```bash
python experiments/replay.py results/S1_BPX.csv fl1400 100 3     # termina con IDENTICAL
```

La campaña principal no había guardado las trazas de VNS y MK-BPX en fl1400, u1432 y rl1304 (270
corridas de cada uno). Se reconstruyeron repitiendo las 720 corridas del conjunto S1 con sus semillas:

```bash
python experiments/campaign_replay.py BPX S1 --core 0 &     # escribe en results_replay/ (≈95 min)
python experiments/campaign_replay.py VNS S1 --core 1 ; wait
python experiments/complete_traces.py                       # completa results/ y escribe results/trace_completion.json
```

`campaign_replay.py` prolonga cada corrida hasta el número de iteraciones registrado y comprueba que
el valor en ese punto es el registrado (`replay_ok`). `complete_traces.py` verifica además las trazas
que sí existían contra la repetición y añade las que faltaban. Los valores de una traza reconstruida
son exactos; sus tiempos se reescalan al reloj de la corrida original y son estimaciones. Hoy
`results/S*_VNS_trace.json` y `results/S*_BPX_trace.json` contienen la traza de todas las corridas.
Los registros incluidos en `results_replay/` son los de esa repetición.

## 9. Formato de los registros

`results/<SET>_<ALGO>.csv` y `results_sota/…` (una fila por corrida):

| Columna | Significado |
|---|---|
| `algo`, `inst`, `n`, `p` | Algoritmo, instancia, número de nodos y de medianas |
| `opt` | Valor de referencia publicado |
| `tl`, `run`, `seed` | Límite de tiempo, índice de corrida y semilla |
| `best`, `gap` | Costo final y brecha porcentual frente a `opt` |
| `iters` | Iteraciones (kicks) |
| `imp_plain`, `imp_px` | Mejoras por aceptación directa y por el cruce |
| `px_calls`, `px_take` | Llamadas a BPX y llamadas con selección no vacía |
| `comps`, `comps_taken`, `dsum` | Componentes encontrados, tomados y suma de \|D\| |
| `scans` | Barridos del vecindario |
| `time`, `t_hit` | Tiempo total y tiempo en que se alcanzó `opt` (−1 si no) |
| `viol` | Violaciones de la cota (debe ser 0) |

`results_orig/P32_<SET>.csv` y `PT_<SET>.csv` (POPSTAR original): `graspit`, `elite` (parámetros);
`best`, `gap`; `wall` (tiempo transcurrido de la búsqueda), `cpu` (el `cputime` que informa el
programa), `init`, `oracle`, `prtime`, `ms_end` (fin de la fase multiarranque), `generations`,
`grasp` (valor antes de la post-optimización), `user`, `sys`, `wall_total`, `ms_iters`,
`po_relinks`, `attempts`, `core`. `PT_<SET>_scaling.csv` guarda el cálculo del número de iteraciones.

`results_orig/<SET>_BPX.csv` y `<SET>_VNS.csv`: `tl` (límite de la tabla 2), `tl3` (horizonte H de la
corrida), `best_tl`, `best_tl3`, `best_end` (valor en cada instante), `iters_tl`, `iters_tl3`,
`iters_end`, `main_iters`, `main_best` (la misma corrida en la campaña principal), contadores como
arriba, `cpu_ratio`, `attempts`, `core`. En `results_replay/` las mismas columnas más `replay_best`,
`replay_ok` (1 si la repetición da el valor registrado) y `t_replay`. `results_orig/blocks.csv`: una
fila por bloque (hora, duración, k, H, sondas).

`*_trace.json` guarda, para cada corrida `instancia_p_corrida`, la lista de (tiempo, costo) de cada
mejora; en la tercera campaña, (tiempo, costo, iteración). Los análisis recalculan la brecha frente
al mejor valor entre el publicado y el hallado en cualquier corrida (`experiments/refvalues.py`).

## 10. Reproducibilidad

- **Semillas.** Semilla = 10⁷ × id del algoritmo + 100 × id del par + índice de corrida. La secuencia
  de soluciones de una corrida es determinista (sección 8). POPSTAR usa semilla = corrida + 1.
- **Límite de tiempo.** El punto en que el límite corta una corrida depende de la máquina: en otro
  equipo los valores finales pueden diferir y las conclusiones deben mantenerse. Por eso los
  algoritmos que se comparan se ejecutaron siempre a la vez en la misma máquina.
- **Hardware.** Las tres campañas se ejecutaron en máquinas virtuales Linux del mismo tipo: dos
  núcleos Intel Xeon a 2,1 GHz, 8 GB de RAM, Python 3.13, NumPy 2.5 y Numba 0.68, un proceso por
  núcleo; POPSTAR se compiló con g++ 13.3 `-O3`. Las máquinas virtuales no son físicamente la misma
  y su velocidad varía con el tiempo, así que ningún registro de una campaña se comparó con registros
  de otra: en cada comparación todos los algoritmos se volvieron a ejecutar lado a lado, y en la
  tercera campaña dentro de un mismo bloque de minutos (sección 7.4). El artículo informa cuánto
  difieren las máquinas (iteraciones en el mismo tiempo con las mismas semillas).
- **Distancias de TSPLIB.** Distancia euclidiana truncada a entero en doble precisión, sin tolerancia;
  así se reproducen los valores de Duran-Mateluna, Ales y Elloumi (2023).
- **Memoria.** Unos 12·n² bytes: cerca de 430 MB para rl5934.

## 11. Procedencia de los datos y licencias

- **OR-Library:** pmed1–pmed40 de J. E. Beasley. `python tools/build_orlib_matrices.py --check`
  confirma que las matrices incluidas coinciden con las de los archivos originales.
- **TSPLIB:** G. Reinelt.
- **Valores de referencia:** óptimos de OR-Library y valores óptimos o mejores conocidos de
  Duran-Mateluna, Ales y Elloumi (2023), en `mkbpx/instances.py`.
- **POPSTAR:** M. G. C. Resende y R. F. Werneck, *A hybrid heuristic for the p-median problem*,
  Journal of Heuristics 10:59–88, 2004. No se incluye; ver sección 7.1.
- **Código de este repositorio:** licencia MIT (`LICENSE`). Para citar, ver `CITATION.cff`.
