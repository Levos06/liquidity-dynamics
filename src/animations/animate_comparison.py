import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from src.solvers.fluid_solver import Fluid2D
from src.utils.visualizer import load_data, detrend_column_linear, detrend_plane
from src.experiments.correlation_study import normalize, get_sliding_windows, init_fluid_from_stream_function

def run_animation_generation():
    input_dir = 'data/processed_data'
    output_dir = 'results/animations'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    prices, volumes = load_data(input_dir)
    
    # Cases to animate
    # (Metric Name, Metric Data, Transform Name, Transform Func)
    cases = [
        ('price', prices, 'raw', lambda x: x),
        ('price', prices, 'col_detrend', detrend_column_linear),
        ('price', prices, 'plane_detrend', detrend_plane)
    ]
    
    for m_name, m_data, t_name, t_func in cases:
        case_name = f"{m_name}-{t_name}"
        print(f"Animating {case_name}...")
        
        # 1. Prepare Real Data Windows
        raw_windows = get_sliding_windows(m_data, 300, window_size=80)
        
        real_series = []
        for w in raw_windows:
            w_trans = t_func(w)
            w_norm = normalize(w_trans)
            real_series.append(w_norm)
        real_series = np.array(real_series)
        
        # 2. Fluid Sim
        w0 = real_series[0]
        N = 80
        dt = 0.1
        fluid = Fluid2D(N, dt, diff=0.0001, visc=0.001)
        
        u_init, v_init = init_fluid_from_stream_function(w0, N)
        fluid.u = u_init
        fluid.v = v_init
        fluid.dens = w0.copy()
        
        # Store frames
        sim_frames = []
        # Pre-calculate sim frames
        for _ in range(300):
            sim_frames.append(fluid.dens.copy())
            fluid.step()
            
        # 3. Animate
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Set titles
        ax[0].set_title(f"Real {case_name}")
        ax[1].set_title(f"Fluid Simulation (Density)")
        
        # Init images
        # Use common normalization for visualization?
        # Or individual. Normalized data is roughly -3 to 3.
        vmin = -3
        vmax = 3
        
        im_real = ax[0].imshow(real_series[0], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        im_sim = ax[1].imshow(normalize(sim_frames[0]), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        
        frame_text = fig.text(0.5, 0.02, '', ha='center')
        
        def update(frame):
            # Real
            im_real.set_data(real_series[frame])
            # Sim
            im_sim.set_data(normalize(sim_frames[frame]))
            
            frame_text.set_text(f"Frame {frame}")
            return im_real, im_sim, frame_text
            
        ani = animation.FuncAnimation(fig, update, frames=300, interval=50, blit=True)
        
        save_path = os.path.join(output_dir, f"{case_name}.mp4")
        print(f"Saving {save_path}...")
        ani.save(save_path, writer='ffmpeg', fps=30)
        plt.close()
        
    print("Done.")

if __name__ == "__main__":
    run_animation_generation()
