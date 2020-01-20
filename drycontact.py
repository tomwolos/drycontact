from elastic import Elastic
import numpy as np
import pyfftw as pf


class DryContact:
    """Elastic-adhesive dry contact.

    Parameters
    ----------
    geometry : np.ndarray
        undeformed contact geometry
    scale : float
        scaling factor for domains other than the unit square (default 1.0)
    modulus : float
        elastic modulus (default 1.0)
    target : tuple(str, float)
        convergence criterion (default ("meanstress", 2*np.pi/3)), options are
            ("meangap", value)
            ("meanstress", value)
    adhesion : tuple(str, float, float)
        adhesion model (default (None,)), options are
            (None,)
            ("MD", rho, gamma)
            ("exp", rho, gamma)

    Usage
    -----
    """
    def __init__(self, geometry: np.ndarray,
                 scale: float=1.0,
                 modulus: float=1.0,
                 target: tuple=("meanstress", 2*np.pi/3),
                 adhesion: tuple=(None,),
                 path: str=""):
        self.geometry = np.asarray(geometry)
        self.scale = scale
        self.modulus = modulus
        self.target = target
        self.adhesion = adhesion
        (self.nx, self.ny) = self.geometry.shape

        self.elastic = Elastic(self.nx, self.ny, scale, modulus, path)
        self.separation = 0
        self.deformation = np.zeros(self.shape)
        self.gap = np.zeros(self.shape)
        self.pressure = np.zeros(self.shape)
        self.adhesion_force = np.zeros(self.shape)
        self.coefficient0 = 0
        self.x = np.linspace(0, 1, self.shape[0])
        self.y = np.linspace(0, self.shape[1]/self.shape[0], self.shape[1])
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.__step = 1
        self.__tolerance = 1e-8
        self.__inner = 20
        self.__outer = 20

        self.x *= self.domain
        self.y *= self.domain
        self.dx *= self.domain
        self.dy *= self.domain

    def __gap(self, maximum=False) -> None:
        self.__deformation()
        self.gap = self.separation + self.geometry + self.deformation
        if maximum:
            self.gap = np.maximum(self.gap, 0)

    def __adhesion(self) -> None:
        if self.adhesion[0] == 'MD':
            self.adhesion_force[:] = 0
            self.adhesion_force[self.gap <= self.adhesion[1]] = self.adhesion[2]/self.adhesion[1]
        elif self.adhesion[0] == 'exp':
            self.adhesion_force[:] = self.adhesion[2]/self.adhesion[1] * \
                np.exp(-self.gap[:]/self.adhesion[1])

    def solve(self) -> None:
        self.pressure[:] = 0
        self.adhesion_force[:] = 0
        self.deformation[:] = 0
        self.separation = -np.min(self.geometry)
        #self.pressure[:] = 1e-10
        for outer in np.arange(self.__outer):
            for _ in np.arange(self.__inner):
                self.__gap()
                self.__adhesion()
                pressure = self.pressure + self.adhesion_force
                #C = *self.shape[0]*self.coefficient0#np.sum(self.gap)/np.sum(pressure)
                #E = np.maximum(np.abs(np.sum(self.gap*pressure)), 1e-10)
                #E = np.maximum(np.sum(np.abs(self.gap*pressure)), 1)
                C = np.sum(pressure**2)
                D = np.sum(self.gap**2)
                E = np.sum((self.gap + pressure)**2)
                #E = np.sum(self.gap**2 + pressure**2)
                #print(C, D)
                #Cp = np.sqrt(D/E)
                Cg = np.sqrt(1/E + .01)
                Cp = 1/np.maximum(np.sum(np.abs(pressure)), 1)
                #Cg = 1/np.maximum(np.sum(np.abs(self.gap)), 1)
                
                Cp = 1
                #Cg = 1/self.coefficient1
                Cg = 1
                #Cg = 1/np.sqrt(D)
                #Cp = 1/np.sqrt(C)
                #Cp = 1/np.max(np.abs(pressure))
                #Cg = 1/np.sum(np.abs(self.gap))
                #print(np.min(Cp), np.min(Cg))
                #print(Cp, Cg)
                #Cg = 1
                #E = C+D
                #E = 1
                #C = 1
                #C = 1/np.sum(np.abs(pressure))/E
                #D = 1/np.sum(np.abs(self.gap))/E
                #C = 1/np.sqrt(np.sum(pressure**2))/E
                #D = 1/np.sqrt(np.sum(self.gap**2))/E
                #C = np.mean(np.abs(self.gap))/np.mean(np.abs(pressure))
                #D = np.sum(np.abs(pressure))/np.sum(np.abs(self.gap))
                #print(np.std(self.gap), np.std(pressure))
                #print(C, D)

                #Cg = 1/np.mean(np.abs(self.gap))

                pressure0 = pressure*Cp
                gap0 = self.gap*Cg

                EE = 0.1

                #gap0 = (self.gap - np.mean(self.gap))/np.std(self.gap)*np.std(pressure) + np.mean(pressure)
                #gap0 = self.gap*np.sum(pressure**2)/np.sum(self.gap**2)
                #gap0 = self.gap*np.sum(np.abs(pressure))/np.sum(np.abs(self.gap))
                #gap0 = self.gap*np.mean(pressure)/np.mean(self.gap)
                R = pressure0 + gap0 - np.sqrt(pressure0**2 + gap0**2)
                #J1 = 1 + self.coefficient0*Cg - (pressure + gap0*self.coefficient0*Cg)/np.sqrt(pressure0**2 + gap0**2 + np.finfo(float).eps)
                #J1 = Cp + self.coefficient0*Cg - (Cp*pressure0 + gap0*self.coefficient0*Cg)/np.sqrt(pressure0**2 + gap0**2 + np.finfo(float).eps)
                J1 = Cp + self.coefficient0*Cg - (Cp*pressure0 + gap0*self.coefficient0*Cg)/np.sqrt(pressure0**2 + gap0**2 + np.finfo(float).eps)
                #J1 = 1 - pressure0/np.sqrt(pressure0**2 + gap0**2 + np.finfo(float).eps)
                #self.pressure = self.pressure - R/J1
                self.pressure = self.pressure - R/(J1 + EE)
                #self.pressure = self.pressure - R/(np.abs(J1) + 0.00001)
                
            
            self.__gap(True)
            self.__adhesion()
            pressure = self.pressure + self.adhesion_force
            unbalanced_complementarity = np.sum(pressure*self.gap)
            #e = e*0.9
            if self.target[0] == 'pressure':
                unbalanced_target = self.target[1] - np.mean(self.pressure)
                separation_change = -self.__step*unbalanced_target
            elif self.target[0] == 'load':
                unbalanced_target = self.target[1] - \
                    np.sum(self.pressure)*self.dx*self.dy
                separation_change = -self.__step * \
                    3**(1/3)/2/np.pi*unbalanced_target
            elif self.target[0] == 'gap':
                unbalanced_target = self.target[1] - np.mean(self.gap)
                separation_change = self.__step*unbalanced_target
            if outer < self.__outer:
                self.separation = self.separation + separation_change
            convergence = (np.abs(unbalanced_target) < self.__tolerance) and (
                unbalanced_complementarity < self.__tolerance)
            if np.mod(outer, 1) == 0 or convergence or outer == self.__outer:
                print('outer: {:3d}, p0 = {: .4e} g0 = {: .4e}, hsep = {: .4e}, |dtarget| = {:.4e}, |p*g| = {:.4e}'.format(
                    outer, np.mean(self.pressure), np.mean(self.gap), self.separation, np.abs(unbalanced_target), unbalanced_complementarity))
            if convergence:
                return None
