# NCREP Internship — Operational Modal Analysis & Structural Health Monitoring

This repository contains the code produced during an internship at **NCREP** (or affiliated institution), focused on **Operational Modal Analysis (OMA)** of civil engineering structures using ambient vibration data. The work is divided into three independent parts, each targeting a different structure or analysis objective.

---

## Scientific context

The central problem is **identifying the natural frequencies and mode shapes of structures** without applying a known excitation — using only ambient vibrations (wind, traffic, footfall). This is known as **output-only modal analysis** or OMA.

The method implemented is **Frequency Domain Decomposition (FDD)**, which works as follows:

1. Compute the **cross-spectral density (CSD) matrix** between all sensor pairs using Welch's method
2. Apply **Singular Value Decomposition (SVD)** frequency by frequency
3. Identify structural modes at frequencies where the first singular value peaks — these correspond to resonance
4. Extract **mode shapes** from the associated left singular vectors

A custom **peak-picking (PP) automation layer** is built on top of FDD, enabling both static single-window analysis and **temporal tracking** of modal frequencies across long recordings (whole-day datasets). A MAC-based (Modal Assurance Criterion) cluster method allows tracking mode identity across consecutive time windows.

A separate **statistical damage detection** pipeline (Part II) uses a novelty index derived from SDO (Statistical Deformation Operator) descriptors and k-medoids clustering.

---

## Repository structure

```
NCREP_internship/
│
├── PART_I/                   # Modal analysis — Lello Library (Brussels)
│   ├── helper/
│   │   ├── processor.py      # Core OMA engine (ModalFrequencyAnalyzer, PeakPicker)
│   │   ├── visualizer.py     # All plotting and figure-saving utilities
│   │   ├── data_loader1.py   # CSV loader for multi-sensor batched files
│   │   ├── folder_processor.py  # Sliding-window processing over a full folder
│   │   ├── create_gif.py     # Assembles result frames into animated GIFs
│   │   └── utils.py
│   ├── exp0.py               # Plot raw time series
│   ├── exp1.py               # Single time-window full analysis (FDD + peak picking + MAC)
│   ├── exp2_nperseg.py       # Sensitivity analysis: nperseg parameter
│   ├── exp2_sigma.py         # Sensitivity analysis: Gaussian smoothing sigma
│   ├── exp3.py               # Whole-day sliding window — modal frequency tracking
│   └── results/              # Output figures and GIFs (generated at runtime)
│
├── PART_II/                  # Damage detection — Alfredo lab specimen
│   └── code/
│       ├── alfredo.py        # Main class: data loading, freq. analysis, stat. analysis
│       ├── process_array.py  # Statistical processing over an array (NI / CB / DI)
│       ├── process_tw.py     # Per-window: SDO computation, k-medoids, novelty index
│       ├── process_folder.py # Orchestration over folder-based datasets
│       ├── data_loader2.py   # Data loader for Alfredo .txt format
│       ├── utils.py
│       ├── exp0.py – exp4.py # Progressive experiments (time domain → damage tracking)
│       └── yellow.py
│
├── PART_chimney/
│   └── main.py               # Modal analysis of a chimney (2 channels, 200 Hz, FDD)
│
├── MAC_Matrix0.png           # Example MAC matrix — undamaged state
├── MAC_Matrix20.png          # Example MAC matrix — later time step
└── NCREP_report.pdf          # Full internship report
```

> **Note:** Data files are not included in this repository (see [Data](#data) section below).

---

## Parts overview

### Part I — Lello Library (Porto)

Accelerometers were placed at the **staircase** and **base** of the [Lello Library](https://en.wikipedia.org/wiki/Livraria_Lello) building. Eight sensors recorded 3-axis acceleration; this code analyses a subset of 4 (staircase sensors, indices 3–6). Recordings are batched in 10-minute `.csv` files sampled at ~100 Hz, with physical scaling factors converting raw ADC values to mG.

**Three structural modes** are targeted in the 8–24 Hz band, nominally around 11.5 Hz, 16.5 Hz, and 22.5 Hz.

**Peak-picking methods implemented:**

| Method | Description |
|--------|-------------|
| Method 0 | Gaussian smoothing + `find_peaks` + local refinement in ±1 Hz window |
| Method 1 | Combined PP index + first PSD singular value; user-defined frequency ranges; curvature-based ranking |
| Method 2 | MAC-based mode clustering: groups frequencies by mode shape similarity, scores clusters against a rolling memory of past detections, enables robust temporal tracking |

---

### Part II — Alfredo Specimen

A **laboratory reinforced concrete specimen** ("Alfredo") was tested under progressive damage states (M_0 = undamaged, M_1–M_3 = increasing damage). Four-channel accelerations (x₁, y₁, x₂, y₂) were recorded. Multiple sensor placements and ambient excitation conditions are available.

Two analysis approaches are applied:

- **Frequency-domain (FDD):** same pipeline as Part I, tracking modal frequency shifts across damage states
- **Statistical (novelty detection):** a Damage Index DI = NI − CB, where NI is a Novelty Index derived from SDO descriptors of windowed data and CB is a control baseline computed via k-medoids clustering

---

### Part chimney

A standalone script applies the full FDD pipeline (Method 2 peak picker, up to 6 modes) to two recordings from a **chimney structure**, sampled at 200 Hz, in the 0–10 Hz frequency band.

---

## Installation

```bash
git clone https://github.com/Colin16655/NCREP_internship.git
cd NCREP_internship
pip install -r requirements.txt
```

**Dependencies:**

```
numpy
scipy
matplotlib
scikit-learn
tqdm
```

> Optional: `PyOMA` (commented-out in code, for cross-validation with the FDD implementation)

---

## Data

Data files are **not included** in this repository due to size and confidentiality constraints. To run the experiments, place the data in the expected directories:

```
PART_I/data/Lello/Jul23/          # CSV files: data_2023_07_10_*.csv
PART_I/data/Lello/Jul24/          # CSV files: data_2024_07_10_*.csv
PART_I/data/Lello/Lello_2023_07_10_WholeDay/   # For exp3
PART_II/data/Alfredo/M_{i}_{j}_{k}_*/subset_signal.txt
PART_chimney/data/chimney/1.txt
PART_chimney/data/chimney/2.txt
```

Contact the repository author or refer to the internship report (`NCREP_report.pdf`) for data access.

---

## Usage

Each experiment script is self-contained. Configuration is done in the `### USER ###` block at the top of each file. Example:

```bash
# Single time-window modal analysis (Part I)
cd PART_I
python exp1.py

# Whole-day frequency tracking (Part I)
python exp3.py

# Chimney modal analysis
cd PART_chimney
python main.py
```

Results are saved in the respective `results/` subdirectory.

---

## Key outputs

- **PSD singular value curves** with identified modal frequencies marked
- **PP index plots** (P1, P2, P3) — coherence-based indicators
- **MAC matrices** — assessing mode shape orthogonality and tracking identity across time
- **Frequency evolution plots** — modal frequencies tracked over a full day or across damage states
- **Animated GIFs** — MAC matrix and PSD evolution over time

---

## Report

The full scientific report is available as `NCREP_report.pdf` in the root of this repository. It covers the theoretical background (FDD, SVD, MAC, novelty index), experimental setups, and results for all three parts.

---

## Author

**Colin** — Applied Mathematics / Actuarial Engineering  
Internship at NCREP  
