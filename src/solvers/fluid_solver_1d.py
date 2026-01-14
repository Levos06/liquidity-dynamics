import numpy as np

class FluidSolver1D:
    def __init__(self, N, dt, nu=0.1, dx=1.0):
        self.N = N
        self.dt = dt
        self.nu = nu  # Viscosity
        self.dx = dx
        
        # Velocity field
        self.u = np.zeros(N)

    def step(self):
        """
        Perform one update step using Upwind for Advection and Central for Diffusion.
        Equation: du/dt = -u * du/dx + nu * d2u/dx2
        """
        u = self.u
        un = u.copy()
        
        dt = self.dt
        dx = self.dx
        nu = self.nu
        
        # We need to handle indices carefully.
        # Vectorized implementation of Upwind Scheme:
        # If u[i] > 0: du/dx ~ (u[i] - u[i-1]) / dx
        # If u[i] < 0: du/dx ~ (u[i+1] - u[i]) / dx
        
        # 1. Diffusion Term: nu * (u[i+1] - 2u[i] + u[i-1]) / dx^2
        # Use np.roll for neighbors
        # Periodic boundaries or Zero-gradient? 
        # Standard approach for market Profile might assume isolation or reflection.
        # Let's use simple Fixed/Zero-Gradient at edges or Periodicity to avoid crashing.
        # Let's try Zero-Gradient (Neumann) by repeating edge values manually after calculation?
        # Or just use np.gradient for simplicity in a quick study?
        # Let's stick to manual finite difference with periodicity for stability in this test.
        
        ip1 = np.roll(un, -1) # i+1
        im1 = np.roll(un, 1)  # i-1
        
        diffusion = nu * (ip1 - 2*un + im1) / (dx**2)
        
        # 2. Advection Term: -u * du/dx
        # Upwind logic
        du_dx = np.zeros_like(un)
        
        # Mask for positive and negative velocities
        pos_mask = un > 0
        neg_mask = un <= 0
        
        # Backward difference for u > 0
        du_dx[pos_mask] = (un[pos_mask] - im1[pos_mask]) / dx
        
        # Forward difference for u < 0
        du_dx[neg_mask] = (ip1[neg_mask] - un[neg_mask]) / dx
        
        advection = -un * du_dx
        
        # Update
        self.u = un + dt * (advection + diffusion)
        
        # Boundary Conditions
        # Enforce simple boundary conditions to prevent explosion at edges due to 'roll' wrapping
        # if the profile is not periodic.
        # Let's clamp edges to 0 flow or copy neighbors?
        # Let's set velocity at walls to 0 (No slip/No flow through walls)
        self.u[0] = 0
        self.u[-1] = 0
