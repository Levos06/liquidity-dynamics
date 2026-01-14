import numpy as np
import matplotlib.pyplot as plt
import os
from src.solvers.fluid_solver import Fluid2D
from src.utils.visualizer import load_data, detrend_column_linear, detrend_plane
from src.experiments.correlation_study import get_sliding_windows, init_fluid_from_stream_function
from tqdm import tqdm

def minmax_scale(data, min_val, max_val):
    """Scale data to [-0.5, 0.5] given min and max."""
    rng = max_val - min_val
    if rng < 1e-9:
        return np.zeros_like(data)
    # Scale to [0, 1] then shift to [-0.5, 0.5]
    return (data - min_val) / rng - 0.5

def normalize_frame(data):
    """Scale frame to [-0.5, 0.5] based on its own min/max."""
    return minmax_scale(data, np.min(data), np.max(data))

def run_weighted_correlation_study():
    input_dir = 'data/processed_data'
    plot_dir = 'results/weighted_correlations'
    
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
    print(f"Running {total_experiments} weighted experiments...")
    
    for m_name, m_data in metrics.items():
        for t_name, t_func in transforms.items():
            case_name = f"{m_name}-{t_name}"
            
            # --- 1. Prepare Data ---
            raw_windows = get_sliding_windows(m_data, 300, window_size=80)
            
            # Apply transform
            transformed_windows = []
            for w in raw_windows:
                transformed_windows.append(t_func(w))
            transformed_windows = np.array(transformed_windows)
            
            # Global Min/Max for this sequence
            g_min = np.min(transformed_windows)
            g_max = np.max(transformed_windows)
            
            # Normalize to [-0.5, 0.5]
            real_series = minmax_scale(transformed_windows, g_min, g_max)
            
            # --- 2. Fluid Simulation ---
            # Init from first normalized frame
            w0 = real_series[0]
            
            N = 80
            dt = 0.1
            fluid = Fluid2D(N, dt, diff=0.0001, visc=0.001)
            
            # Init
            u_init, v_init = init_fluid_from_stream_function(w0, N)
            fluid.u = u_init
            fluid.v = v_init
            fluid.dens = w0.copy()
            
            weighted_corrs = []
            
            # Initial state for MSD calculation
            real_0 = real_series[0]
            
            for t in range(300):
                # Sim Frame Normalized
                # Normalize Sim frame-by-frame to keep contrast
                sim_norm = normalize_frame(fluid.dens)
                
                real_t = real_series[t]
                
                # 1. Pearson Correlation
                # Flatten
                r_flat = real_t.flatten()
                s_flat = sim_norm.flatten()
                
                # Manual Pearson to avoid dependency issues if constant
                # but np.corrcoef is easier
                if np.std(r_flat) < 1e-9 or np.std(s_flat) < 1e-9:
                    corr = 0.0
                else:
                    corr = np.corrcoef(r_flat, s_flat)[0, 1]
                
                # 2. Mean Squared Deviation from t=0
                # Using the normalized real Data
                msd = np.mean((real_t - real_0)**2)
                
                # 3. Weighted Metric
                # Multiply!
                metric = corr * msd
                
                weighted_corrs.append(metric)
                
                fluid.step()
                
            all_results[case_name] = weighted_corrs
            
            # Plot Individual
            plt.figure(figsize=(10, 6))
            plt.plot(weighted_corrs, label='Weighted Corr (Corr * MSD)')
            plt.title(f"Weighted Correlation: {case_name}")
            plt.xlabel("Time Step")
            plt.ylabel("Score")
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir,f"{case_name}.png"))
            plt.close()
            
    # --- Combined Plot ---
    print("Generating combined plot...")
    plt.figure(figsize=(15, 10))
    for name, vals in all_results.items():
        plt.plot(vals, label=name, alpha=0.7)
        
    plt.title("Weighted Correlation Evolution - All Cases")
    plt.xlabel("Time Step")
    plt.ylabel("Weighted Score (Corr * MSD)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "combined_weighted.png"))
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    run_weighted_correlation_study()
