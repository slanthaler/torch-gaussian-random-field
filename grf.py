import torch
import math

class GRF1d:
    '''
    Implements a Gaussian random field sampler in one spatial dimension.
    The fields are produced on a periodic domain, with decay of Fourier spectrum as

    ~ exp( -(length_scale*(tau + K) )**(-alpha) )

    Args:
        length_scale: controls length scale across which there is variation in the samples
        alpha: controls the decay of Fourier coefficients 
        tau: additional offset in wavenumbers to control "active" wavenumbers (|k| < tau)
    '''
    def __init__(self, length_scale=0.5, alpha=1., tau=2.):
        self.length_scale = length_scale
        self.alpha = alpha
        self.tau = tau
        
    def sample(self, Nsamp, Ngrid):
        # define wavenumber and coefficients
        wavenumbers = torch.fft.rfftfreq(Ngrid,1/Ngrid)
        decay_coeff = torch.exp( -(self.length_scale * (torch.abs(wavenumbers)+self.tau))**self.alpha )
        decay_coeff /= torch.norm(decay_coeff)
        rand_coeff = torch.randn(Nsamp,len(wavenumbers), dtype=torch.cfloat)

        # Fourier coefficients of u
        uhat = decay_coeff.view(1,-1) * rand_coeff
        
        # transform back via (real) inverse FT
        u = torch.fft.irfft(uhat, norm='forward')
        
        return u
    
    @staticmethod
    def grid(Ngrid, periodic=True):
        if periodic:
            return torch.linspace(0,1,Ngrid+1)[:-1] # periodic grid
        else:
            return torch.linspace(0,1,Ngrid) # non-periodic grid


class GRF2d:
    '''
    Implements a Gaussian random field sampler in two spatial dimensions.
    The fields are produced on a periodic domain, with decay of Fourier spectrum as

    ~ exp( -(length_scale*(tau + K) )**(-alpha) )

    Args:
        length_scale: controls length scale across which there is variation in the samples
        alpha: controls the decay of Fourier coefficients 
        tau: additional offset in wavenumbers to control "active" wavenumbers (|k| < tau)
    '''
    def __init__(self, length_scale=0.5, alpha=1., tau=2.):
        self.length_scale = length_scale
        self.alpha = alpha
        self.tau = tau
        
    def sample(self, Nsamp, Ngrid):
        # define wavenumber and coefficients
        kx = torch.fft.fftfreq(Ngrid,1/Ngrid)  # wavenumbers in x-direction
        ky = torch.fft.rfftfreq(Ngrid,1/Ngrid) # wavenumbers in y-direction
        Kx, Ky = torch.meshgrid(kx,ky,indexing='ij')
        K = torch.sqrt(Kx**2 + Ky**2)
        
        #
        decay_coeff = torch.exp( -(self.length_scale * (torch.abs(K)+self.tau))**self.alpha )
        decay_coeff /= torch.norm(decay_coeff)
        rand_coeff = torch.randn(Nsamp,*K.shape, dtype=torch.cfloat)
        
        # Fourier coefficients of u
        uhat = decay_coeff.unsqueeze(0) * rand_coeff
        
        # transform back via (real) inverse FT
        u = torch.fft.irfft2(uhat, norm='forward')
        return u
   
    @staticmethod
    def grid(Ngrid, periodic=True):
        if periodic:
            grid_1d = torch.linspace(0,1,Ngrid+1)[:-1] # periodic grid (remove last grid point, since 1==0 on torus)
        else:
            grid_1d = torch.linspace(0,1,Ngrid) # non-periodic grid (keep last grid point, since 1!=0)
        X,Y = torch.meshgrid(grid_1d,grid_1d,indexing='ij')
        return X,Y
