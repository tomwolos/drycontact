from elastic import Elastic
import numpy as np
import pyfftw as pf


class Adhesion:
    """Adhesion model.

    Calculates adhesion force based on the contact gap.
    Dependencies: numpy

    Attributes
    ----------
    model : str
        name of the adhesion model (default '' = no adhesion)
        options:
        ''
        'exp'
        'MD'
    kwargs : model parameters given as keyword=value
        model:      keywords:
        'exp'       rho, gamma
        'MD'        rho, gamma

    Methods
    -------

    """
    def __init__(self, model: str='',
                       **kwargs):
        self.model = model
        self.params = kwargs

    def update(self, gap: np.ndarray):
        temp = np.zeros(gap.shape)
        if self.model == 'exp':
            temp[:] = (self.params['gamma']/self.params['rho']
                    * np.exp(-gap/self.params['rho']))
        if self.model == 'MD':
            temp[gap <= self.params['rho']] = (
                    self.params['gamma']/self.params['rho'])
        return temp


class DryContact:
    """Elastic-adhesive dry contact.

    Calculate the gap and stress in a dry contact considering elastic
    deformation and adhesion.
    Dependencies: numpy, pyfftw, elastic

    Attributes
    ----------
    geometry : np.ndarray
        undeformed contact geometry
    scale : float
        scaling factor for domains other than the unit square (default 1.0)
    modulus : float
        elastic modulus (default 1.0)
    target : tuple(str, float)
        convergence criterion (default ('meanstress', 2*np.pi/3));
        options:
        ('meangap', value)
        ('meanstress', value)
    adhesion : None or Adhesion
        See Adhesion for more information
    path : string
        path to the file with influence coefficients (default '');
        set path=None if you do not want to save or load the coefficients
        see Elastic class for more information
    verbose : bool
        print information while solving the problem

    Methods
    -----
    """
    def __init__(self, geometry: np.ndarray,
                 scale: float=1.0,
                 modulus: float=1.0,
                 target: tuple=('meanstress', 2*np.pi/3),
                 adhesion: object=Adhesion(''),
                 path: str=''):
        self.geometry = np.asarray(geometry)
        self.scale = scale
        self.modulus = modulus
        self.target = target
        self.adhesion = adhesion
        self.path = path

        self.separation = 0
        self.gap = np.zeros(self.geometry.shape)
        self.adhesion_force = np.zeros(self.geometry.shape)
        self.stress = np.zeros(self.geometry.shape)
        self.deformation = np.zeros(self.geometry.shape)

        (self.nx, self.ny) = self.geometry.shape
        self.elastic = Elastic(self.nx, self.ny, scale, modulus, path)
        self.x = np.linspace(0, scale, self.nx)
        self.y = np.linspace(0, scale*self.ny/self.nx, self.ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.separation_change = 0
        self.unbalanced_target = 0

        self.__tolerance = 1e-8
        self.__inner = 20
        self.__outer = 20

    def __separation(self):
        if self.target[0] == 'meanstress':
            self.unbalanced_target = self.target[1] - np.mean(self.stress)
            self.separation_change = -self.unbalanced_target
        elif self.target[0] == 'meangap':
            self.unbalanced_target = self.target[1] - np.mean(self.gap)
            self.separation_change = self.unbalanced_target

    def solve(self):
        self.separation = -np.min(self.geometry)
        self.stress[:] = 0.0
        for outer in range(self.__outer):
            for _ in range(self.__inner):
                self.deformation[:] = self.elastic.update(self.stress)
                self.gap[:] = (self.separation + self.geometry
                               + self.deformation)
                self.adhesion_force[:] = self.adhesion.update(self.gap)

                A = 0.1
                C = self.elastic.coefficients[-1, -1]
                temp = np.sqrt((self.stress + self.adhesion_force)**2
                               + self.gap**2 + np.finfo(float).eps)
                self.stress[:] = self.stress - ((self.stress +
                                 self.adhesion_force + self.gap - temp)
                                 /(1 + C - (self.stress + self.adhesion_force
                                 + C*self.gap)/temp + A))
                self.stress[np.isnan(self.stress)] = 0.0
            
            self.deformation[:] = self.elastic.update(self.stress)
            self.gap[:] = np.maximum(self.separation + self.geometry
                                     + self.deformation, 0)
            self.adhesion_force[:] = self.adhesion.update(self.gap)
            self.__separation()

            unbalanced_complementarity = (np.sum(self.stress*self.gap)
                + np.sum(self.adhesion_force*self.gap))
            if outer < self.__outer:
                self.separation = self.separation + self.separation_change
            convergence = (np.abs(self.unbalanced_target) < self.__tolerance) and (
                unbalanced_complementarity < self.__tolerance)
            if np.mod(outer, 1) == 0 or convergence or outer == self.__outer:
                print('outer: {:3d}, p0 = {: .4e} g0 = {: .4e}, hsep = {: .4e}, |dtarget| = {:.4e}, |p*g| = {:.4e}'.format(
                    outer, np.mean(self.stress), np.mean(self.gap),
                    self.separation, np.abs(self.unbalanced_target), unbalanced_complementarity))
            if convergence:
                return None
