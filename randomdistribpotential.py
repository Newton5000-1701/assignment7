#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:29:55 2024

@author: isaacthompson
"""

import numpy as np
from matplotlib import pyplot as pp
import creategrid  # Import creategrid to access qgrid function

# Parameters
dx = 0.05  # Adjust spacing to handle larger grid size effectively
dy = 0.05
x = np.arange(-10, 10, dx)
y = np.arange(-20, 20, dy)
X, Y = np.meshgrid(x, y)
e_0 = 8.8541 * 10**(-12)  # Vacuum permittivity in SI units

# Get 50 random charges from qgrid
data = creategrid.qgrid(50)
charges = data['charges']
positions = data['coordinates']

# Masking radius for each charge
mask_radius = 0.06  # Adjust as needed to prevent singularities

# Distance function with masking
def r(X, Y, xp, yp):
    r = np.sqrt((X - xp)**2 + (Y - yp)**2)
    return np.where(r <= mask_radius, np.nan, r)

# Calculate potential by summing contributions
def V_total(X, Y):
    V = np.zeros_like(X, dtype=float)
    for i in range(len(charges)):
        q = charges[i]
        xp, yp = positions[i]
        a = q / (4 * np.pi * e_0)
        V += a / r(X, Y, xp, yp)
    return V

# Total potential
V = V_total(X, Y)

# Electric field components using central differencing
def x_partial(f, X, Y, h):
    return (f(X + h, Y) - f(X - h, Y)) / (2 * h)

def y_partial(f, X, Y, h):
    return (f(X, Y + h) - f(X, Y - h)) / (2 * h)

# Calculate electric field
E_x = -x_partial(V_total, X, Y, dx)
E_y = -y_partial(V_total, X, Y, dy)

# Plot electric potential
pp.figure(figsize=(10, 8))
pp.contour(X, Y, V, cmap='inferno', levels=100)
pp.colorbar(label="Electric Potential (V)")
pp.xlabel("x (m)")
pp.ylabel("y (m)")
pp.title("Electric Potential of 50 Random Charges")
pp.savefig('potential.png')
pp.show()

# Normalize and mask the electric field vectors for visualization
E_magnitude = np.sqrt(E_x**2 + E_y**2)
E_x_norm = E_x / E_magnitude
E_y_norm = E_y / E_magnitude

# Reduce density for quiver plot
density_factor = 10
mask = np.zeros_like(E_magnitude, dtype=bool)
mask[::density_factor, ::density_factor] = True

# Apply mask to normalized field components for plotting field directions
E_x_reduced = np.where(mask, E_x_norm, np.nan)  # Use NaN to ignore in quiver plot
E_y_reduced = np.where(mask, E_y_norm, np.nan)

# Plot electric field direction and magnitude
pp.figure(figsize=(10, 8))
contour = pp.contour(X, Y, E_magnitude, cmap='Spectral', levels=200)
pp.colorbar(contour, label="Electric Field Magnitude (V/m)")
pp.quiver(X, Y, E_x_reduced, E_y_reduced, color='black', scale=35, width=0.003)
pp.xlabel("x (m)")
pp.ylabel("y (m)")
pp.title("Electric Field of 50 Random Charges (Density-Reduced)")
pp.savefig('efield.png')
pp.show()
