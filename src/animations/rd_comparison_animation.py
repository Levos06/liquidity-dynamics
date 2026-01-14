import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from src.solvers.rd_solver import RDSolver
from src.utils.visualizer import load_data, detrend_column_linear
from src.experiments.weighted_correlation_study import minmax_scale, get_sliding_windows, normalize_frame

def run_rd_animation_generation():
    input_dir = 'data/processed_data'
    output_dir = 'results/rd_animations'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    prices, volumes = load_data(input_dir)
    
    # Cases to animate
    # (Metric Name, Metric Data, Transform Name, Transform Func)
    cases = [
        ('volume', volumes, 'raw', lambda x: x),
        ('price', prices, 'raw', lambda x: x),
        ('price', prices, 'col_detrend', detrend_column_linear)
    ]
    
    for m_name, m_data, t_name, t_func in cases:
        case_name = f"{m_name}-{t_name}"
        print(f"Animating {case_name} (RD)...")
        
        # 1. Prepare Real Data Windows
        raw_windows = get_sliding_windows(m_data, 300, window_size=80)
        
        transformed_windows = []
        for w in raw_windows:
            transformed_windows.append(t_func(w))
        transformed_windows = np.array(transformed_windows)
        
        g_min = np.min(transformed_windows)
        g_max = np.max(transformed_windows)
        
        # Normalize to [-0.5, 0.5] for display (Real Series)
        real_series_display = minmax_scale(transformed_windows, g_min, g_max)
        
        # Prepare for Solver (0 to 1 range)
        if g_max - g_min < 1e-9:
            real_series_solver = np.zeros_like(transformed_windows)
        else:
            real_series_solver = (transformed_windows - g_min) / (g_max - g_min)
        
        # 2. RD Sim
        w0 = real_series_solver[0]
        N = 80
        dt = 1.0
        # Gray-Scott params
        solver = RDSolver(N, dt, Du=1.0, Dv=0.5, F=0.0545, k=0.062)
        
        solver.v = w0.copy()
        
        # Store frames
        sim_frames = []
        
        # Pre-calculate sim frames
        for _ in range(300):
            # Normalize frame to [-0.5, 0.5] for consistent visualization comparison
            # Gray-Scott V is naturally [0, 1] usually, but can vary.
            # We want to see the structure.
            sim_frames.append(normalize_frame(solver.v))
            solver.step()
            
        # 3. Animate
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Set titles
        ax[0].set_title(f"Real {case_name}")
        ax[1].set_title(f"Reaction-Diffusion (Gray-Scott)")
        
        # Init images
        vmin = -0.5
        vmax = 0.5
        
        im_real = ax[0].imshow(real_series_display[0], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        im_sim = ax[1].imshow(sim_frames[0], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        
        frame_text = fig.text(0.5, 0.02, '', ha='center')
        
        def update(frame):
            # Real
            im_real.set_data(real_series_display[frame])
            # Sim
            im_sim.set_data(sim_frames[frame])
            
            frame_text.set_text(f"Frame {frame}")
            return im_real, im_sim, frame_text
            
        ani = animation.FuncAnimation(fig, update, frames=300, interval=50, blit=True)
        
        save_path = os.path.join(output_dir, f"{case_name}.mp4")
        print(f"Saving {save_path}...")
        ani.save(save_path, writer='ffmpeg', fps=30)
        plt.close()
        
    print("Done.")

if __name__ == "__main__":
    run_rd_animation_generation()
