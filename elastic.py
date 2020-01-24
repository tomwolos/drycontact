import numpy as np
import pyfftw as pf


class Elastic:
    """Elastic deformation due to normal stress.

    Perform FFTW planning and calculate the elastic deformation due to given
    non-periodic normal stress distribution using half-space approximation. The
    influence coefficients and their FFT are saved to and loaded from
    coefficients{nx}x{ny}.npz (see below for the description of nx and ny).
    Dependencies: numpy, pyfftw

    Attributes
    ----------
    nx, ny : int, int
        numbers of points used for the discretisation of stress/deformation
    scale : float
        scaling factor for domains other than the unit square (default 1.0)
    modulus : float
        elastic modulus (default 1.0)
    path : string
        path to the coefficients file (default '');
        set path=None if you do not want to save or load the coefficients
    verbose : bool
        print when generating influence coefficients (default True)

    Methods
    -------
    update(stress : numpy.ndarray) -> numpy.ndarray
        calculates and returns elastic deformation for the given stress
    """

    def __init__(self, nx: int,
                 ny: int,
                 scale: float=1.0,
                 modulus: float=1.0,
                 path: str='',
                 verbose: bool=True):
        self.nx = nx
        self.ny = ny
        self.scale = scale
        self.modulus = modulus
        self.path = path
        self.verbose = verbose

        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, ny/nx, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # FFTW planning
        self.real = pf.empty_aligned((2*nx, 2*ny), 'float64')
        self.complex = pf.empty_aligned((2*nx, ny+1), 'complex128')
        self.fft = pf.FFTW(self.real, self.complex, axes=(0, 1))
        self.ifft = pf.FFTW(self.complex, self.real, axes=(0, 1),
                            direction='FFTW_BACKWARD')

        # load or generate and save the influence coefficients and their FFT
        self.fftcoefficients = pf.empty_aligned((2*nx, ny+1), 'complex128')
        self.coefficients = np.zeros((nx, ny))
        if path is not None:
            filename = f'{path}coefficients{nx}x{ny}.npz'
            try:
                temp = np.load(filename)
                self.fftcoefficients[:] = temp['fftcoefficients'][:]
                self.coefficients[:] = temp['coefficients'][:]
            except IOError:
                self.__generate()
                np.savez(filename, fftcoefficients=self.fftcoefficients,
                         coefficients=self.coefficients)
        else:
            self.__generate()
        self.fftcoefficients[:] *= self.scale/np.pi/self.modulus
        self.coefficients[:] *= self.scale/np.pi/self.modulus

    def __generate(self):
        if self.verbose:
            print('Generating influence coefficients', flush=True)
        for jx in range(self.nx):
            for jy in range(self.ny):
                x1 = self.x[jx] - self.x[-1] + self.dx/2
                x2 = self.x[jx] - self.x[-1] - self.dx/2
                y1 = self.y[jy] - self.y[-1] + self.dy/2
                y2 = self.y[jy] - self.y[-1] - self.dy/2
                self.real[jx][jy] = (
                    np.abs(x1)*(np.arcsinh(y1/x1) - np.arcsinh(y2/x1))
                    + np.abs(x2)*(np.arcsinh(y2/x2) - np.arcsinh(y1/x2))
                    + np.abs(y1)*(np.arcsinh(x1/y1) - np.arcsinh(x2/y1))
                    + np.abs(y2)*(np.arcsinh(x2/y2) - np.arcsinh(x1/y2)))
                self.real[-2-jx][jy] = self.real[jx][jy]
                self.real[jx][-2-jy] = self.real[jx][jy]
                self.real[-2-jx][-2-jy] = self.real[jx][jy]
        self.coefficients[:] = self.real[:self.nx, :self.ny]
        self.fft.execute()
        self.fftcoefficients[:] = self.complex[:]
        if self.verbose:
            print('done.\n', flush=True)

    def update(self, stress: np.ndarray):
        self.real[:] = 0.0
        self.real[:self.nx, :self.ny] = stress
        self.fft.execute()
        self.complex[:] *= self.fftcoefficients
        self.ifft.execute()
        self.real[-1-self.nx:-1, -1-self.ny:-1] /= 2*self.nx*2*self.ny
        return self.real[-1-self.nx:-1, -1-self.ny:-1]
