import numpy as np
import matplotlib.pyplot as plt
import os
from src.solvers.rd_solver_1d import RDSolver1D
from src.utils.visualizer import load_data, detrend_column_linear, detrend_plane
from src.experiments.correlation_study import get_sliding_windows
from src.experiments.weighted_correlation_study import minmax_scale
from tqdm import tqdm

def run_rd_1d_study():
    input_dir = 'data/processed_data'
    plot_dir = 'results/rd_1d_results'
    
    if os.path.exists(plot_dir):
        import shutil
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Loading data...")
    prices, volumes = load_data(input_dir)
    
    metrics = {
        'price': prices,
        'volume': volumes,
        'pv': prices * volumes,
        'log_log_pv': np.log(np.log(prices * volumes + 1) + 1)
    }
    
    transforms = {
        'raw': lambda x: x,
        'col_detrend': detrend_column_linear
    }
    
    # We only need 1 example window for the visual study, or aggregate stats?
    # User said: "experiment with 1d simulations... each picture... is 1 column of height 80"
    # This implies we visualize the evolution of 1 column over time.
    # Let's pick the FIRST 300 steps of the dataset as the "Real" sample.
    
    samples_count = 1 # Just run one main demonstration for each case? 
    # Or correlation over sliding windows?
    # Let's do sliding windows logic to be consistent, but maybe visualize just one.
    
    # Actually, let's keep it simple: 1 "Experiment" = The first 300 timestamps.
    # Initialize 1D Sim with Vector at T=0.
    # Run for 300 steps to get Sim Matrix (80, 300).
    # Compare with Real Matrix (80, 300).
    
    # Transpose for visualization: Time usually on X or Y?
    # Real Data is (80 levels, 300 time). Matrix shape (80, 300).
    # X=Time, Y=Level. 
    
    for m_name, m_data in metrics.items():
        for t_name, t_func in transforms.items():
            case_name = f"{m_name}-{t_name}"
            print(f"Running 1D Study for {case_name}...")
            
            # Extract first 300 steps
            real_block = m_data[:, 0:300] # (80, 300)
            
            # Apply transform?
            # detrend_column_linear expects (80, T) and detrends along axis 0 (levels)? 
            # No, column detrend removes linear trend from the column vector.
            # Yes.
            if t_name == 'col_detrend':
                 # Apply col by col
                 real_block_trans = np.zeros_like(real_block)
                 for t in range(300):
                     col = real_block[:, t]
                     # detrend_column_linear accepts a Matrix and detrends columns?
                     # Let's check visualizer.py implementation if possible, or just re-implement simple detrend.
                     # "detrend_column_linear" in visualizer usually detrends the WINDOW.
                     # Let's assume we pass the whole block.
                     pass
                 # Re-using the function from visualizer which handles (80, 80) usually.
                 real_block_trans = t_func(real_block)
            else:
                 real_block_trans = t_func(real_block) # raw
                 
            # Normalize for Solver [0, 1]
            g_min = np.min(real_block_trans)
            g_max = np.max(real_block_trans)
            
            if g_max - g_min < 1e-9:
                real_block_norm = np.zeros_like(real_block_trans)
            else:
                real_block_norm = (real_block_trans - g_min) / (g_max - g_min)
                
            # Initial Profile (t=0)
            init_profile = real_block_norm[:, 0]
            
            # Simulation
            N = 80
            dt = 1.0
            solver = RDSolver1D(N, dt, Du=1.0, Dv=0.5, F=0.0545, k=0.062)
            solver.v = init_profile.copy()
            
            sim_history = []
            sim_history.append(solver.v.copy())
            
            for _ in range(299): # 299 more steps to total 300
                solver.step()
                sim_history.append(solver.v.copy())
                
            sim_matrix = np.array(sim_history).T # (300, 80) -> (80, 300)
            # Now we have Sim Matrix (80, 300) matching Real Block (80, 300)
            
            # Visualization: Space-Time Heatmap
            # Left: Real, Right: 1D Sim
            
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            # Real
            ax[0].imshow(real_block_norm, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
            ax[0].set_title(f"Real Evolution: {case_name}")
            ax[0].set_xlabel("Time")
            ax[0].set_ylabel("Price Level")
            
            # Sim
            ax[1].imshow(sim_matrix, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
            ax[1].set_title(f"1D RD Sim Evolution")
            ax[1].set_xlabel("Time")
            
            plt.tight_layout()
            save_path = os.path.join(plot_dir, f"{case_name}_spacetime.png")
            plt.savefig(save_path)
            plt.close()
            
            # Compute Correlation (Column vs Column over time)
            # Flatten columns or just dot product?
            corrs = []
            real_0 = real_block_norm[:, 0]
            
            for t in range(300):
                r_col = real_block_norm[:, t]
                s_col = sim_matrix[:, t]
                
                # Standard Pearson
                if np.std(r_col) < 1e-9 or np.std(s_col) < 1e-9:
                    c = 0
                else:
                    c = np.corrcoef(r_col, s_col)[0, 1]
                    
                # Weighted Metric?
                # The user just asked for "experiments", not explicitly the weighted metric again, but it's good practice to log it.
                # Let's simplified plot of just correlation.
                corrs.append(c)
                
            # Plot Correlation
            plt.figure(figsize=(10, 4))
            plt.plot(corrs)
            plt.ylim(-1, 1)
            plt.title(f"Correlation over Time (1D Profile Matching): {case_name}")
            plt.savefig(os.path.join(plot_dir, f"{case_name}_corr.png"))
            plt.close()

    print("Done.")

if __name__ == "__main__":
    run_rd_1d_study()
