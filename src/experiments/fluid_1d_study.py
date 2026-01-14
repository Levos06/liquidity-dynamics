import numpy as np
import matplotlib.pyplot as plt
import os
from src.solvers.fluid_solver_1d import FluidSolver1D
from src.utils.visualizer import load_data, detrend_column_linear
from src.experiments.weighted_correlation_study import minmax_scale
from tqdm import tqdm

def run_fluid_1d_study():
    input_dir = 'data/processed_data'
    plot_dir = 'results/fluid_1d_results'
    
    if os.path.exists(plot_dir):
        import shutil
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Loading data...")
    prices, volumes = load_data(input_dir)
    
    metrics = {
        'price': prices,
        'volume': volumes
    }
    
    # Only creating a few key examples as requested ("calm sections")
    transforms = {
        'raw': lambda x: x,
        'col_detrend': detrend_column_linear
    }
    
    for m_name, m_data in metrics.items():
        for t_name, t_func in transforms.items():
            case_name = f"{m_name}-{t_name}"
            print(f"Running 1D Fluid Study for {case_name}...")
            
            # Extract first 300 steps (same window as before)
            real_block = m_data[:, 0:300]
            
            if t_name == 'col_detrend':
                 # Visualizer function usually expects (80, T) and fits T? 
                 # Wait, detrend_column_linear in 'visualizer.py' fits a line over the 80 points (COLUMN).
                 # So applying it to the block is applying it column by column.
                 real_block_trans = t_func(real_block)
            else:
                 real_block_trans = t_func(real_block)

            # Normalize data to a range suitable for velocity, e.g., [-1, 1] or [0, 1]
            # Burgers equation is sensitive to magnitude (CFL condition: u*dt/dx < 1).
            # If we map Price to Velocity directly, we must ensure stability.
            
            g_min = np.min(real_block_trans)
            g_max = np.max(real_block_trans)
            
            # Map to [-0.5, 0.5] (so max speed is 0.5)
            # This is safer for dt=1.0, dx=1.0. u*dt/dx = 0.5 < 1.
            velocity_scale = 0.5
            if g_max - g_min < 1e-9:
                real_block_norm = np.zeros_like(real_block_trans)
            else:
                real_block_norm = ((real_block_trans - g_min) / (g_max - g_min) - 0.5) * 2 * velocity_scale
                
            # Initial Profile (t=0)
            init_profile = real_block_norm[:, 0]
            
            # Simulation
            N = 80
            dt = 0.5 # Smaller timestep for fluid stability
            # nu = 0.1 (viscosity)
            solver = FluidSolver1D(N, dt=dt, nu=0.1, dx=1.0)
            
            solver.u = init_profile.copy()
            
            sim_history = []
            
            # Run enough steps to match physical time 300
            # If dt=0.5, we need 600 steps to reach T=300? 
            # Or do we map steps 1:1?
            # RD solver had dt=1.0. 
            # Real data is per timestamp.
            # Let's keep steps 1:1, so we simulate 300 ticks.
            # If dt=0.5, the "fluid time" will be 150. That's fine, it's abstract time.
            
            sim_history.append(solver.u.copy())
            for _ in range(299): 
                solver.step()
                sim_history.append(solver.u.copy())
                
            sim_matrix = np.array(sim_history).T # (80, 300)
            
            # Visualization
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            # Real
            # Show original normalized [0, 1] for visuals or the velocity map?
            # Let's show the velocity map we used [-0.5, 0.5]
            ax[0].imshow(real_block_norm, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            ax[0].set_title(f"Real Evolution: {case_name}")
            ax[0].set_xlabel("Time")
            ax[0].set_ylabel("Price Level")
            
            # Sim
            ax[1].imshow(sim_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            ax[1].set_title(f"1D Burgers' Fluid Evolution")
            ax[1].set_xlabel("Time")
            
            plt.tight_layout()
            save_path = os.path.join(plot_dir, f"{case_name}_spacetime.png")
            plt.savefig(save_path)
            plt.close()
            
            # Correlation Plot
            corrs = []
            for t in range(300):
                r_col = real_block_norm[:, t]
                s_col = sim_matrix[:, t]
                if np.std(r_col) < 1e-9 or np.std(s_col) < 1e-9:
                    c = 0
                else:
                    c = np.corrcoef(r_col, s_col)[0, 1]
                corrs.append(c)
                
            plt.figure(figsize=(10, 4))
            plt.plot(corrs)
            plt.ylim(-1, 1)
            plt.title(f"Correlation (Burgers'): {case_name}")
            plt.savefig(os.path.join(plot_dir, f"{case_name}_corr.png"))
            plt.close()

    print("Done.")

if __name__ == "__main__":
    run_fluid_1d_study()
