import numpy as np
import matplotlib.pyplot as plt
import os
import random

def load_data(input_dir):
    prices = np.load(os.path.join(input_dir, 'prices.npy'))
    volumes = np.load(os.path.join(input_dir, 'volumes.npy'))
    return prices, volumes

def get_random_windows(num_windows, total_timesteps, window_size=80):
    # Ensure window fits
    max_start = total_timesteps - window_size
    starts = sorted(random.sample(range(max_start), num_windows))
    return starts

def detrend_column_linear(window):
    """
    For each column (timestep), fit y = ax + b and subtract.
    Window shape: (80, 80) = (Levels, Time)
    But wait, user says "for each column... appoximating line...".
    Usually in LOB heatmaps, Y-axis is Price Level (0-79) and X-axis is Time.
    So "column" means a specific time step. 
    The "line" would be along the *levels* (vertical slice).
    y = value at level i
    x = level index i
    """
    rows, cols = window.shape
    x = np.arange(rows)
    detrended = np.zeros_like(window)
    
    for j in range(cols):
        y = window[:, j]
        # Fit y = ax + b
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        line = m * x + c
        detrended[:, j] = y - line
        
    return detrended

def detrend_plane(window):
    """
    Fit z = ax + by + c to the 2D window and subtract.
    x = row index
    y = col index
    z = value
    """
    rows, cols = window.shape
    x_indices, y_indices = np.indices((rows, cols))
    
    # Flatten
    x_flat = x_indices.flatten()
    y_flat = y_indices.flatten()
    z_flat = window.flatten()
    
    # Design matrix: [x, y, 1]
    A = np.vstack([x_flat, y_flat, np.ones(len(x_flat))]).T
    
    # Least squares
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    a, b, c = coeffs
    
    # Compute plane
    plane = a * x_indices + b * y_indices + c
    
    return window - plane

def save_heatmap(data, path, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.savefig(path)
    plt.close()

def main():
    input_dir = 'processed_data'
    images_dir = 'images'
    
    # Cleanup previous images
    if os.path.exists(images_dir):
        import shutil
        shutil.rmtree(images_dir)
        print(f"Cleaned up {images_dir}")
    
    print("Loading data...")
    prices, volumes = load_data(input_dir)
    
    # 3. Derive arrays
    pv = prices * volumes
    log_pv = np.log(pv + 1)
    log_log_pv = np.log(log_pv + 1)
    
    arrays = {
        'price': prices,
        'volume': volumes,
        'pv': pv,
        'log_pv': log_pv,
        'log_log_pv': log_log_pv
    }
    
    # Setup Experiments
    experiments = [
        ('1_raw', lambda w: w),
        ('2_col_detrend', detrend_column_linear),
        ('3_plane_detrend', detrend_plane)
    ]
    
    # Select 10 random window starts
    # Seed for reproducibility if needed, but user asked for random.
    # random.seed(42) 
    starts = get_random_windows(10, prices.shape[1], window_size=80)
    print(f"Selected window starts: {starts}")
    
    for exp_name, transform_func in experiments:
        print(f"Running Experiment: {exp_name}")
        exp_dir = os.path.join(images_dir, exp_name)
        
        for name, data in arrays.items():
            arr_dir = os.path.join(exp_dir, name)
            os.makedirs(arr_dir, exist_ok=True)
            
            for i, start in enumerate(starts):
                end = start + 80
                window = data[:, start:end]
                
                # Transform
                processed_window = transform_func(window)
                
                # Save
                fname = f"window_{i}_start_{start}.png"
                save_path = os.path.join(arr_dir, fname)
                save_heatmap(processed_window, save_path, f"{name} {exp_name} (t={start})")

    print("Done generating images.")

if __name__ == "__main__":
    main()
