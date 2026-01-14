import numpy as np
from tqdm import tqdm

class Fluid2D:
    def __init__(self, N, dt, diff, visc):
        self.N = N  # Grid size (NxN)
        self.dt = dt  # Time step
        self.diff = diff  # Diffusion rate
        self.visc = visc  # Viscosity

        # Arrays for density (optional scalar), u (x-velocity), v (y-velocity)
        # We also need previous states for the solver steps
        self.u = np.zeros((N, N))
        self.v = np.zeros((N, N))
        self.u_prev = np.zeros((N, N))
        self.v_prev = np.zeros((N, N))
        
        self.dens = np.zeros((N, N))
        self.dens_prev = np.zeros((N, N))
        
        self.lid_velocity = 0.0 # Velocity of the top wall (u-component)

    def add_source(self, x, s, dt):
        """Adds source s to field x."""
        x += dt * s

    def diffuse(self, b, x, x0, diff, dt):
        """
        Solves the diffusion equation using implicit method (Jacobi iteration).
        b: Boundary condition index (0: density/pressure, 1: u-velocity, 2: v-velocity)
        """
        a = dt * diff * (self.N - 2) * (self.N - 2)
        
        # Jacobi iteration
        # In a real high-performance code, we'd use Numba or C++ here or a better solver like Conjugate Gradient.
        # But for this demo, 20 iterations of Jacobi is standard in graphics papers (Stam 1999).
        for _ in range(20): 
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[0:-2, 1:-1] + x[2:, 1:-1] + 
                                                   x[1:-1, 0:-2] + x[1:-1, 2:])) / (1 + 4 * a)
            self.set_bnd(b, x)

    def advect(self, b, d, d0, u, v, dt):
        """
        Advects field d along velocity field (u, v) using Semi-Lagrangian backtracing.
        """
        dt0 = dt * (self.N - 2)
        N = self.N
        
        # Grid coordinates
        # We iterate over the inner grid
        # Vectorized implementation for speed in Python
        
        # Create a meshgrid of indices [1, N-2]
        j, i = np.meshgrid(np.arange(1, N-1), np.arange(1, N-1))
        
        # Backtrace
        x = i - dt0 * u[i, j]
        y = j - dt0 * v[i, j]
        
        # Clamp coordinates
        x = np.clip(x, 0.5, N - 1.5)
        y = np.clip(y, 0.5, N - 1.5)
        
        # Indices of the 4 neighbors
        i0 = x.astype(int)
        i1 = i0 + 1
        j0 = y.astype(int)
        j1 = j0 + 1
        
        # Interpolation weights
        s1 = x - i0
        s0 = 1 - s1
        t1 = y - j0
        t0 = 1 - t1
        
        # Bilinear interpolation
        d[i, j] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                   s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
        
        self.set_bnd(b, d)

    def project(self, u, v, p, div):
        """
        Projection step: enforces mass conservation (div(u) = 0).
        Solves the Poisson equation for pressure and subtracts gradient of pressure from velocity field.
        """
        N = self.N
        h = 1.0 / N
        
        # Calculate divergence
        # div[i, j] = -0.5 * h * (u[i+1, j] - u[i-1, j] + v[i, j+1] - v[i, j-1])
        div[1:-1, 1:-1] = -0.5 * h * (u[2:, 1:-1] - u[0:-2, 1:-1] + 
                                      v[1:-1, 2:] - v[1:-1, 0:-2])
        self.set_bnd(0, div)
        
        p.fill(0)
        self.set_bnd(0, p)
        
        # Solve Poisson equation (div(grad(p)) = div(u)) using Jacobi
        # In Stam's code: x[i, j] = (div[i, j] + x[i-1...]) / 4
        # Note: In standard form Laplace(p) = div(u), here we solve similar structure.
        
        for _ in range(20):
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] + p[0:-2, 1:-1] + p[2:, 1:-1] + 
                             p[1:-1, 0:-2] + p[1:-1, 2:]) / 4
            self.set_bnd(0, p)
            
        # Subtract gradient
        # u[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j]) / h
        # v[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1]) / h
        u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[0:-2, 1:-1]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, 0:-2]) / h
        
        self.set_bnd(1, u)
        self.set_bnd(2, v)

    def set_bnd(self, b, x):
        """
        Set boundary conditions.
        b=1: u-velocity (reflection on left/right walls)
        b=2: v-velocity (reflection on top/bottom walls)
        b=0: scalar (continuity)
        """
        N = self.N
        
        # Walls
        # Left and Right
        x[0, :] = -x[1, :] if b == 1 else x[1, :]
        x[-1, :] = -x[-2, :] if b == 1 else x[-2, :]
        
        # Top and Bottom
        x[:, 0] = -x[:, 1] if b == 2 else x[:, 1]
        
        if b == 1: # u-velocity on top wall
             # Moving lid: u_boundary = lid_vel. 
             # (u_ghost + u_fluid)/2 = u_boundary => u_ghost = 2*u_boundary - u_fluid
             x[:, -1] = 2 * self.lid_velocity - x[:, -2]
        else:
             x[:, -1] = -x[:, -2] if b == 2 else x[:, -2]
        
        # Corners (average of neighbors)
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

    def step(self):
        """
        Main simulation step.
        """
        N = self.N
        dt = self.dt
        diff = self.diff
        visc = self.visc
        
        # Velocity step
        
        # Add sources (if any) - omitted for now, usually done outside or via simple addition
        
        # Diffuse velocity
        self.diffuse(1, self.u_prev, self.u, visc, dt)
        self.diffuse(2, self.v_prev, self.v, visc, dt)
        
        # Project (clean up divergence from diffusion/source addition if any)
        self.project(self.u_prev, self.v_prev, self.u, self.v)
        
        # Advect velocity
        self.advect(1, self.u, self.u_prev, self.u_prev, self.v_prev, dt)
        self.advect(2, self.v, self.v_prev, self.u_prev, self.v_prev, dt)
        
        # Project (clean up divergence from advection)
        self.project(self.u, self.v, self.u_prev, self.v_prev)
        
        # Density step (scalar transport)
        self.diffuse(0, self.dens_prev, self.dens, diff, dt)
        self.advect(0, self.dens, self.dens_prev, self.u, self.v, dt)
