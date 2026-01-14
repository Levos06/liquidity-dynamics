# FlowMarket: Physics-based Order Book Modeling

**Repository Name**: `liquidity-dynamics`
**Description**: Modeling Limit Order Book dynamics using Physics-based equations (Navier-Stokes, Reaction-Diffusion, KdV).

## 1. Project Overview

This project explores the hypothesis that **Market Microstructure** (Limit Order Book dynamics) can be modeled using **Physical Laws** governing flow, diffusion, and pattern formation. We compare real market evolution against simulations of:
1.  **Fluid Dynamics** (Navier-Stokes, Burgers' Equation).
2.  **Reaction-Diffusion** (Gray-Scott Model).
3.  **Soliton Dynamics** (Korteweg-de Vries Equation).

## 2. Directory Structure

The project is organized into a modular structure:

```
.
├── data/                 # Input LOB data (Parquet)
├── results/              # Generated plots, heatmaps, and animations
│   ├── correlations/
│   ├── rd_correlations/
│   ├── advanced_1d_results/
│   └── animations/
├── src/
│   ├── solvers/          # Numerical Solvers
│   │   ├── fluid_solver.py      # 2D Navier-Stokes
│   │   ├── rd_solver.py         # 2D Gray-Scott
│   │   ├── kdv_solver.py        # 1D KdV / KdVB
│   │   └── ...
│   ├── experiments/      # Study Scripts
│   │   ├── correlation_study.py
│   │   ├── advanced_1d_study.py
│   │   └── ...
│   ├── utils/            # Helpers
│   │   └── visualizer.py
│   └── animations/       # Animation Generators
└── README.md
```

## 3. Experiments & Results

We run several comparative studies to validate the physical models.

### A. Fluid Dynamics (2D)
-   **Model**: Incompressible Navier-Stokes.
-   **Hypothesis**: Order book flows like a fluid with inertia.
-   **Run**: `python -m src.experiments.correlation_study`

### B. Reaction-Diffusion (2D)
-   **Model**: Gray-Scott.
-   **Hypothesis**: Orders interact locally (reaction) and spread (diffusion), forming Turing patterns.
-   **Run**: `python -m src.experiments.rd_correlation_study`

### C. Advanced 1D Models
-   **Models**: 
    -   **Viscous Burgers**: Inertial shock formation.
    -   **Korteweg-de Vries (KdV)**: Soliton formation (structural persistence).
    -   **Burgers + Forcing**: Mean-reverting flow (Elasticity).
-   **Run**: `python -m src.experiments.advanced_1d_study`

## 4. How to Run

1.  **Install Dependencies**:
    ```bash
    pip install numpy matplotlib scipy pyarrow fastparquet tqdm
    ```

2.  **Run Experiments** (from root directory):
    ```bash
    # 2D Fluid Correlation
    python -m src.experiments.correlation_study
    
    # 2D Reaction-Diffusion
    python -m src.experiments.rd_correlation_study
    
    # 1D Advanced Models (KdV, Burgers)
    python -m src.experiments.advanced_1d_study
    ```

3.  **Generate Animations**:
    ```bash
    python -m src.animations.animate_comparison
    python -m src.animations.rd_comparison_animation
    ```
