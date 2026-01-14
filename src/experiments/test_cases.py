import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fluid_solver import Fluid2D
import os

def run_lid_driven_cavity(N=64, duration=4.0, dt=0.1):
    print("Running Lid Driven Cavity Simulation...")
    
    # Setup
    fluid = Fluid2D(N, dt, diff=0.0001, visc=0.0001)
    fluid.lid_velocity = 1.0
    
    # Setup plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # We will simulate for 'frames' steps
    frames = int(duration / dt)
    
    def update(frame):
        # Enforce lid boundary condition: handled by fluid.lid_velocity in solver
        
        fluid.step()
        
        # Visualization
        ax[0].clear()
        ax[1].clear()
        
        # Pressure field
        # Note: In our solver, density is stored in fluid.dens, 
        # but fluid.p is used for projection. We can visualize p or curl.
        # Let's visualize velocity magnitude/curl or just u/v
        
        # Visualizing Curl (Vorticity) is often nicer for fluids
        u = fluid.u
        v = fluid.v
        curl = (v[2:, 1:-1] - v[0:-2, 1:-1]) - (u[1:-1, 2:] - u[1:-1, 0:-2])
        
        # Plot vorticity
        im = ax[0].imshow(curl, cmap='RdBu', origin='lower', animated=True)
        ax[0].set_title(f"Vorticity (Frame {frame})")
        
        # Quiver plot for velocity
        # Subsample for cleaner plot
        steps = 2
        Y, X = np.mgrid[0:N:steps, 0:N:steps]
        # We need to slice u and v. 
        # Note: imshow origin='lower' means index 0 is bottom.
        # But matrices are (x, y) or (row, col)? 
        # In our solver: u[i, j]. Usually i is x, j is y? 
        # Let's assume standard matrix indexing: row=y, col=x?
        # Actually in the solver:
        # x = i - dt0 * u[i, j] -> u[i, j] is x-component.
        # set_bnd: x[0, :] = ... (Left wall). So index 0 is First dimension.
        # So First Dimension is X, Second is Y.
        # Thus u[x, y].
        # Matplotlib imshow expects (row, col) = (y, x).
        # So we should transpose for imshow.
        
        ax[1].quiver(X, Y, u.T[::steps, ::steps], v.T[::steps, ::steps], scale=5)
        ax[1].set_title("Velocity Field")
        
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    
    save_path = "lid_driven_cavity.mp4"
    print(f"Saving {save_path}...")
    ani.save(save_path, writer='ffmpeg', fps=30)
    plt.close()
    print("Done.")

def run_jet_simulation(N=64, duration=4.0, dt=0.1):
    print("Running Jet Simulation...")
    
    fluid = Fluid2D(N, dt, diff=0.000, visc=0.0001)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    frames = int(duration / dt)
    
    # Source position
    cx, cy = N//2, N//5
    
    def update(frame):
        # Add source
        # Add density source
        fluid.dens[cx-2:cx+2, cy-2:cy+2] += 10.0 * dt
        # Add velocity source (upwards jet)
        fluid.v[cx-2:cx+2, cy-2:cy+2] += 5.0 * dt
        
        # Add some randomness to make it interesting (turbulence)
        fluid.u[cx-2:cx+2, cy-2:cy+2] += np.random.uniform(-2, 2) * dt
        
        fluid.step()
        
        ax[0].clear()
        ax[1].clear()
        
        # Plot Density
        ax[0].imshow(fluid.dens.T, cmap='inferno', origin='lower', vmin=0, vmax=5)
        ax[0].set_title(f"Density (Frame {frame})")
        
        # Velocity
        steps = 2
        Y, X = np.mgrid[0:N:steps, 0:N:steps]
        ax[1].quiver(X, Y, fluid.u.T[::steps, ::steps], fluid.v.T[::steps, ::steps], scale=10)
        ax[1].set_title("Velocity")
        
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    
    save_path = "jet_simulation.mp4"
    print(f"Saving {save_path}...")
    ani.save(save_path, writer='ffmpeg', fps=30)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    run_lid_driven_cavity(N=64, duration=20.0, dt=0.1) 
    run_jet_simulation(N=64, duration=20.0, dt=0.1) # Increased duration
