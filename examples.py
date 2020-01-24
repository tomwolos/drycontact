#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from drycontact import Adhesion, DryContact
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

N = 256
L = 1
R = 2*L
E = 1

gap0 = 0.0429*L
p0 = 2e-4*E

rho = 3e-4*L
gamma = 1e-5*L*E
T = gamma**(2/3)*R**(1/3)/rho/E**(2/3)


# rho = gamma**(2/3)*R**(1/3)/T/E**(2/3)
# print('Tabor parameter: {}'.format(T))

print('\nElastic (adhesive) contact for grid {} x {}\n'.format(N, N))

tx = np.linspace(-L/2, L/2, N)
ty = np.linspace(-L/2, L/2, N)
(x, y) = np.meshgrid(tx, ty)
geometry = x**2/2/R + y**2/2/R


adhesion3 = Adhesion('exp', rho=2e-3*L, gamma=1e-4*L*E)
contact3 = DryContact(geometry, L, E, ('meangap', 0.0429),
                      adhesion3,
                      '/home/tom/work/coefficients/')

adhesion2 = Adhesion('MD', rho=3e-4*L, gamma=1e-5*L*E)
contact2 = DryContact(geometry, L, E, ('meangap', 0.0405),
                      adhesion2,
                      '/home/tom/work/coefficients/')

adhesion1 = Adhesion('')
contact1 = DryContact(geometry, L, E, ('meangap', 0.0398),
                      adhesion1, '/home/tom/work/coefficients/')
contact = contact3
print(contact.adhesion)
print(contact.target)
contact.solve()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_wireframe(x, y, contact.stress)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, contact.stress, edgecolor='None')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, contact.gap, edgecolor='none')

yc = N//2

temp = contact.gap - contact.geometry
fig = plt.figure()
plt.plot(tx, 1e3*temp[:, yc])
plt.minorticks_on()
plt.grid(True, 'both')
plt.xlabel('x')
plt.ylabel('displacement')

temp = 1e3*(contact.separation + contact.geometry)
temp[temp > 20] = np.nan
fig = plt.figure()
plt.plot(tx, temp[:, yc], 'r')
temp = -1e3*contact.deformation
plt.plot(tx, temp[:, yc], 'b')
plt.minorticks_on()
plt.grid(True, 'both')
plt.xlabel('x')
plt.ylabel('geometry')

fig = plt.figure()
plt.plot(tx, contact.stress[:, yc])
plt.minorticks_on()
plt.grid(True, 'both')
plt.xlabel('x')
plt.ylabel('stress')

plt.show(block=True)
