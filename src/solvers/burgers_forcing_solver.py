import numpy as np
from .fluid_solver_1d import FluidSolver1D

class BurgersForcingSolver(FluidSolver1D):
    def __init__(self, N, dt, nu=0.1, dx=1.0, k=0.01, u_eq=None):
        super().__init__(N, dt, nu, dx)
        self.k = k # Spring constant
        
        # Equilibrium state (attractor)
        if u_eq is None:
            self.u_eq = np.zeros(N)
        else:
            self.u_eq = u_eq.copy()

    def step(self):
        """
        Burgers step + Forcing.
        """
        # Save current u
        u_old = self.u.copy()
        
        # Run standard Burgers step (updates self.u)
        super().step()
        u_burgers = self.u.copy()
        
        # Apply Forcing term explicitly or semi-implicitly
        # du/dt = ... - k(u - u_eq)
        # Euler: u_new = u_burgers - k * dt * (u_old - u_eq)
        
        forcing = -self.k * (u_old - self.u_eq)
        
        self.u = u_burgers + self.dt * forcing
        
        # Re-enforce boundary conditions (0 velocity at walls) if needed
        self.u[0] = 0
        self.u[-1] = 0
