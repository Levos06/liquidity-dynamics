import numpy as np
import matplotlib.pyplot as plt
import os
from src.solvers.fluid_solver import Fluid2D
from src.utils.visualizer import load_data, detrend_column_linear, detrend_plane
from tqdm import tqdm

def normalize(data):
    """Normalize data to mean=0, std=1."""
    if data is None: return None
    m = np.mean(data)
    s = np.std(data)
    if s < 1e-9:
        return np.zeros_like(data)
    return (data - m) / s

def get_sliding_windows(data, num_windows, window_size=80):
    windows = []
    for t in range(num_windows):
        win = data[:, t:t+window_size]
        windows.append(win)
    return np.array(windows)

def init_fluid_from_stream_function(psi, N=80):
    gy, gx = np.gradient(psi)
    u = gy 
    v = -gx
    return u, v

def run_correlation_study():
    input_dir = 'data/processed_data'
    plot_dir = 'results/correlations'
    
    # Clean up
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
        'log_pv': np.log(prices * volumes + 1),
        'log_log_pv': np.log(np.log(prices * volumes + 1) + 1)
    }
    
    transforms = {
        'raw': lambda x: x,
        'col_detrend': detrend_column_linear,
        'plane_detrend': detrend_plane
    }
    
    all_results = {}
    
    total_experiments = len(metrics) * len(transforms)
    print(f"Running {total_experiments} experiments x 300 steps...")
    
    for m_name, m_data in metrics.items():
        for t_name, t_func in transforms.items():
            case_name = f"{m_name}-{t_name}"
            
            # --- Prepare Real Data ---
            raw_windows = get_sliding_windows(m_data, 300, window_size=80)
            real_series = []
            for w in raw_windows:
                w_trans = t_func(w)
                w_norm = normalize(w_trans)
                real_series.append(w_norm)
            real_series = np.array(real_series)
            
            # --- Fluid Simulation ---
            # Init from FIRST window
            w0 = real_series[0]
            
            N = 80
            dt = 0.1
            fluid = Fluid2D(N, dt, diff=0.0001, visc=0.001)
            
            # 1. Init Velocity (structure)
            u_init, v_init = init_fluid_from_stream_function(w0, N)
            fluid.u = u_init
            fluid.v = v_init
            
            # 2. Init Density (Pattern for Correlation)
            # We want Correlation(t=0) = 1.
            # So fluid state at t=0 must match real_series[0].
            fluid.dens = w0.copy()
            
            correlations = []
            
            # Run 300 steps
            for t in range(300):
                # Measure correlation BEFORE step for t=0?
                # "Compare Real[t] vs Sim[t]"
                # At t=0: Sim=w0, Real=w0. Corr=1.
                
                # Capture current state (Density)
                sim_frame = fluid.dens.copy()
                sim_norm = normalize(sim_frame)
                
                real_t = real_series[t]
                
                # Pearson Correlation
                r_flat = real_t.flatten()
                s_flat = sim_norm.flatten()
                
                c = np.dot(r_flat, s_flat) / len(r_flat)
                correlations.append(c)
                
                # Evolve
                fluid.step()
                
            all_results[case_name] = correlations
            
            # Save individual plot
            plt.figure(figsize=(10, 6))
            plt.plot(correlations, label='Correlation')
            plt.title(f"Correlation: {case_name}")
            plt.xlabel("Time Step")
            plt.ylabel("Pearson Correlation")
            plt.ylim(-1, 1.1) 
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir,f"{case_name}.png"))
            plt.close()
            
    # --- Combined Plot ---
    print("Generating combined plot...")
    plt.figure(figsize=(15, 10))
    for name, corrs in all_results.items():
        plt.plot(corrs, label=name, alpha=0.7)
        
    plt.title("Correlation Evolution - All Cases")
    plt.xlabel("Time Step")
    plt.ylabel("Correlation")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "combined_correlation.png"))
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    run_correlation_study()
