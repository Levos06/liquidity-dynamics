import numpy as np
import matplotlib.pyplot as plt
import os
from src.solvers.kdv_solver import KdVSolver
from src.solvers.burgers_forcing_solver import BurgersForcingSolver
from src.utils.visualizer import load_data, detrend_column_linear
from src.experiments.weighted_correlation_study import minmax_scale
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

def run_advanced_1d_study():
    input_dir = 'data/processed_data'
    plot_dir = 'results/advanced_1d_results'
    
    if os.path.exists(plot_dir):
        import shutil
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Loading data...")
    prices, volumes = load_data(input_dir)
    
    # We focus on Price-Col-Detrend (usually structural) and Volume-Raw
    cases = [
        ('price', prices, 'col_detrend', detrend_column_linear),
        ('price', prices, 'raw', lambda x: x)
    ]
    
    for m_name, m_data, t_name, t_func in cases:
        case_name = f"{m_name}-{t_name}"
        print(f"Running Advanced Study for {case_name}...")
        
        # Data Prep
        real_block = m_data[:, 0:300]
        if t_name == 'col_detrend':
             real_block_trans = t_func(real_block)
        else:
             real_block_trans = t_func(real_block)
             
        # Normalize to velocity range [-0.5, 0.5]
        g_min = np.min(real_block_trans)
        g_max = np.max(real_block_trans)
        
        velocity_scale = 0.5
        if g_max - g_min < 1e-9:
             real_block_norm = np.zeros_like(real_block_trans)
        else:
             real_block_norm = ((real_block_trans - g_min) / (g_max - g_min) - 0.5) * 2 * velocity_scale
             
        init_profile = real_block_norm[:, 0]
        
        # --- KdV Simulation ---
        # KdV requires smooth input because 3rd derivative explodes on noise.
        # Apply slight smoothing to initial profile.
        init_profile_smooth = gaussian_filter1d(init_profile, sigma=2.0)
        # KdVB: dt=0.001, nu=0.1
        kdv = KdVSolver(N=80, dt=0.001, dx=1.0, nu=0.1) 
        kdv.u = init_profile_smooth.copy()
        
        kdv_frames = []
        # We need 300 visual steps (total time ~150 with dt=0.5).
        # Here dt=0.001. To advance 0.5 per frame, we need 500 steps.
        steps_per_frame = 500 
        
        kdv_frames.append(kdv.u.copy())
        for _ in range(299):
            for _ in range(steps_per_frame):
                kdv.step()
            kdv_frames.append(kdv.u.copy())
            
        kdv_matrix = np.array(kdv_frames).T # (80, 300)
        
        # --- Burgers +  Forcing Simulation ---
        # u_t + u u_x = nu u_xx - k(u - u_eq)
        # Equilibrium = Initial Profile? Or 0 (Mean)?
        # If we assume "Mean Reversion", u_eq should be the long-term mean (0 for normalized data).
        u_eq = np.zeros(80) 
        
        bf = BurgersForcingSolver(N=80, dt=0.5, nu=0.1, dx=1.0, k=0.05, u_eq=u_eq)
        bf.u = init_profile.copy()
        
        bf_frames = []
        bf_frames.append(bf.u.copy())
        for _ in range(299):
            bf.step()
            bf_frames.append(bf.u.copy())
            
        bf_matrix = np.array(bf_frames).T
        
        # --- Visualization ---
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        vmin, vmax = -0.5, 0.5
        
        # Real
        ax[0].imshow(real_block_norm, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax[0].set_title(f"Real: {case_name}")
        
        # KdV
        ax[1].imshow(kdv_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax[1].set_title(f"KdV (Solitons)")
        
        # Forcing
        ax[2].imshow(bf_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax[2].set_title(f"Burgers + Mean Reversion")
        
        for a in ax:
            a.set_xlabel("Time")
            a.set_ylabel("Level")
            
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{case_name}_comparison.png"))
        plt.close()

    print("Done.")

if __name__ == "__main__":
    run_advanced_1d_study()
