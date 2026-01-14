import numpy as np

class KdVSolver:
    def __init__(self, N, dt, dx=1.0, nu=0.01):
        self.N = N
        self.dt = dt
        self.dx = dx
        self.nu = nu 
        self.u_now = np.zeros(N)
        self.first_step = True

    @property
    def u(self):
        return self.u_now

    @u.setter
    def u(self, value):
        self.u_now = value.copy()
        self.first_step = True

    def step(self):
        """
        Step KdVB equation with Edge (Replicate) Boundaries.
        """
        dt = self.dt
        dx = self.dx
        nu = self.nu
        u = self.u_now
        
        # Pad array to handle boundaries without wrapping
        # Pad with 2 values on each side using 'edge' (repeat last value)
        # This implies u_x = 0, u_xx = 0, u_xxx = 0 at the virtual infinity.
        u_pad = np.pad(u, (2, 2), mode='edge')
        
        # Indices for the inner N points
        # u[i] corresponds to u_pad[i+2]
        # We process range [2, N+2) from u_pad
        
        # Neighbors slices
        # u is u_pad[2:-2]
        # ip1 is u_pad[3:-1] (i+1)
        # ip2 is u_pad[4:]   (i+2)
        # im1 is u_pad[1:-3] (i-1)
        # im2 is u_pad[0:-4] (i-2)
        
        c = u # Center
        ip1 = u_pad[3:-1]
        im1 = u_pad[1:-3]
        ip2 = u_pad[4:]
        im2 = u_pad[0:-4]
        
        # 1. Nonlinear: u u_x (Reduced from 6 u u_x for stability)
        # Central diff: (ip1 - im1) / 2dx
        nonlinear = 1.0 * c * (ip1 - im1) / (2 * dx)
        
        # 2. Dispersion: u_xxx
        # (ip2 - 2ip1 + 2im1 - im2) / 2dx^3
        dispersion = (ip2 - 2*ip1 + 2*im1 - im2) / (2 * dx**3)
        
        # 3. Viscosity: nu * u_xx
        # (ip1 - 2c + im1) / dx^2
        diffusion = nu * (ip1 - 2*c + im1) / (dx**2)
        
        rhs = -(nonlinear + dispersion) + diffusion
        
        # Euler Step
        u_next = c + dt * rhs
        
        # Clamp to prevent runaway explosion
        u_next = np.clip(u_next, -5.0, 5.0)
        
        self.u_now = u_next
