import numpy as np

class RDSolver1D:
    def __init__(self, N, dt, Du=1.0, Dv=0.5, F=0.0545, k=0.062):
        self.N = N
        self.dt = dt
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        
        # U and V concentrations (1D arrays)
        self.u = np.ones(N)
        self.v = np.zeros(N)
        
        # 1D Laplacian kernel
        self.laplacian_kernel = np.array([1, -2, 1])

    def step(self):
        """
        Perform one update step using Forward Euler.
        """
        # Compute Laplacians using convolution
        # mode='same': returns output of same size as input
        # boundary='symm': Reflection at boundaries (for LOB edges)
        
        # np.convolve is for 1D.
        # But we need to handle boundary conditions. np.convolve generally zero-pads or requires manual padding.
        # scipy.ndimage.convolve1d or np.convolve with 'same' and careful edge handling.
        
        # Let's use np.convolve valid mode and handle edges, OR pad manually.
        
        # Manual symmetric padding
        u_padded = np.pad(self.u, (1, 1), mode='symmetric')
        v_padded = np.pad(self.v, (1, 1), mode='symmetric')
        
        # Convolve (valid mode returns size N if input is N+2 and kernel is 3)
        lu = np.convolve(u_padded, self.laplacian_kernel, mode='valid')
        lv = np.convolve(v_padded, self.laplacian_kernel, mode='valid')
        
        uvv = self.u * self.v * self.v
        
        # Reaction-Diffusion Equations
        du = self.Du * lu - uvv + self.F * (1 - self.u)
        dv = self.Dv * lv + uvv - (self.F + self.k) * self.v
        
        # Update
        self.u += du * self.dt
        self.v += dv * self.dt
        
        # Clamp to avoid numerical instability
        self.u = np.clip(self.u, 0, 1)
        self.v = np.clip(self.v, 0, 1)
