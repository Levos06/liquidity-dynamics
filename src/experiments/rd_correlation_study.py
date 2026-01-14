import numpy as np
import matplotlib.pyplot as plt
import os
from src.solvers.rd_solver import RDSolver
from src.utils.visualizer import load_data, detrend_column_linear, detrend_plane
from src.experiments.correlation_study import get_sliding_windows
from src.experiments.weighted_correlation_study import minmax_scale, normalize_frame
from tqdm import tqdm

def run_rd_correlation_study():
    input_dir = 'data/processed_data'
    plot_dir = 'results/rd_correlations'
    
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
    print(f"Running {total_experiments} RD experiments...")
    
    for m_name, m_data in metrics.items():
        for t_name, t_func in transforms.items():
            case_name = f"{m_name}-{t_name}"
            
            # --- 1. Prepare Data ---
            raw_windows = get_sliding_windows(m_data, 300, window_size=80)
            
            transformed_windows = []
            for w in raw_windows:
                transformed_windows.append(t_func(w))
            transformed_windows = np.array(transformed_windows)
            
            g_min = np.min(transformed_windows)
            g_max = np.max(transformed_windows)
            
            # Normalize to [-0.5, 0.5] for metric calculation
            real_series_metric = minmax_scale(transformed_windows, g_min, g_max)
            
            # For RD solver initialization, we need [0, 1] range (Concentration)
            # Map global min/max to [0, 1]
            if g_max - g_min < 1e-9:
                real_series_solver = np.zeros_like(transformed_windows)
            else:
                real_series_solver = (transformed_windows - g_min) / (g_max - g_min)
            
            # --- 2. Reaction-Diffusion Sim ---
            N = 80
            dt = 1.0 # Larger dt is okay for RD if D is small enough, but let's be careful.
            # Using parameters for patterns
            solver = RDSolver(N, dt=1.0, Du=1.0, Dv=0.5, F=0.0545, k=0.062)
            
            # Init V from first frame
            w0 = real_series_solver[0]
            solver.v = w0.copy()
            # U is already ones()
            # Perturb U slightly with V? Usually U=1, V=init is fine.
            # To maintain mass balance initially, maybe U = 1 - V?
            # solver.u = 1.0 - solver.v 
            # Let's keep U=1 (abundance) as per standard Gray-Scott demo
            
            weighted_corrs = []
            real_0 = real_series_metric[0]
            
            for t in range(300):
                # Sim Frame (V concentration)
                # Normalize detailed structure for correlation
                # V is in [0, 1]. Normalize to [-0.5, 0.5] based on frame min/max
                sim_norm = normalize_frame(solver.v)
                
                real_t = real_series_metric[t]
                
                # Metric
                r_flat = real_t.flatten()
                s_flat = sim_norm.flatten()
                
                if np.std(r_flat) < 1e-9 or np.std(s_flat) < 1e-9:
                    corr = 0.0
                else:
                    corr = np.corrcoef(r_flat, s_flat)[0, 1]
                
                msd = np.mean((real_t - real_0)**2)
                metric = corr * msd
                
                weighted_corrs.append(metric)
                
                solver.step()
                
            all_results[case_name] = weighted_corrs
            
            # Plot Individual
            plt.figure(figsize=(10, 6))
            plt.plot(weighted_corrs, label='RD Weighted Corr')
            plt.title(f"RD Weighted Correlation: {case_name}")
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
        
    plt.title("Reaction-Diffusion Correlation Evolution")
    plt.xlabel("Time Step")
    plt.ylabel("Weighted Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "combined_rd_weighted.png"))
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    run_rd_correlation_study()
