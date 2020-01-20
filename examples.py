#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from drycontact import DryContact
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


#rho = gamma**(2/3)*R**(1/3)/T/E**(2/3)
#print('Tabor parameter: {}'.format(T))

print('\nElastic (adhesive) contact for grid {} x {}\n'.format(N, N))

tx = np.linspace(-L/2, L/2, N)
ty = np.linspace(-L/2, L/2, N)
(x, y) = np.meshgrid(tx, ty)
geometry = x**2/2/R + y**2/2/R

#contact = DryContact(geometry, ('gap', 0.0398), L, E)
contact = DryContact(geometry, ('gap', 0.0405), L, E, ('MD', 3e-4*L, 1e-5*L*E))
#contact = DryContact(geometry, ('gap', 0.0429), L, E, ('exp', 2e-3*L, 1e-4*L*E))



contact.solve()


#fig = plt.figure()
#ax = plt.axes(projection = '3d')
#ax.plot_wireframe(x, y, contact.pressure)

#fig = plt.figure()
#ax = plt.axes(projection = '3d')
#ax.plot_surface(x, y, contact.gap, edgecolor='None')

#fig = plt.figure()
#ax = plt.axes(projection = '3d')
#ax.plot_surface(x, y, contact.geometry, edgecolor='none')



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
plt.plot(tx, contact.pressure[:, yc])
plt.minorticks_on()
plt.grid(True, 'both')
plt.xlabel('x')
plt.ylabel('stress')

plt.show(block = True)
