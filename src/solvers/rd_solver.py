import numpy as np
from scipy.signal import convolve2d

class RDSolver:
    def __init__(self, N, dt, Du=1.0, Dv=0.5, F=0.0545, k=0.062):
        self.N = N
        self.dt = dt
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        
        # U and V concentrations
        self.u = np.ones((N, N))
        self.v = np.zeros((N, N))
        
        # Laplacian kernel (5-point stencil)
        self.laplacian_kernel = np.array([[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]])

    def step(self):
        """
        Perform one update step using Forward Euler.
        """
        # Compute Laplacians
        # mode='same': returns output of same size as input
        # boundary='wrap': Periodic boundary conditions (good for removing edge artifacts)
        # or 'symm'/'fill'? Market data isn't periodic.
        # 'symm' (reflection) is probably better for bounded LOB.
        lu = convolve2d(self.u, self.laplacian_kernel, mode='same', boundary='symm')
        lv = convolve2d(self.v, self.laplacian_kernel, mode='same', boundary='symm')
        
        uvv = self.u * self.v * self.v
        
        # Reaction-Diffusion Equations
        # du/dt = Du*Lap(u) - uv^2 + F(1-u)
        du = self.Du * lu - uvv + self.F * (1 - self.u)
        
        # dv/dt = Dv*Lap(v) + uv^2 - (F+k)v
        dv = self.Dv * lv + uvv - (self.F + self.k) * self.v
        
        # Update
        self.u += du * self.dt
        self.v += dv * self.dt
        
        # Clamp to avoid numerical instability
        self.u = np.clip(self.u, 0, 1)
        self.v = np.clip(self.v, 0, 1)
