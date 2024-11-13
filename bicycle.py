#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:33:06 2024

@author: isaacthompson
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 70  # mass of the cyclist (kg)
P = 400  # power output (W)
v0 = 4  # initial velocity (m/s)
dt = 0.1  # time step (s)
t_max = 200  # maximum time (s)

# Time array
t = np.arange(0, t_max + dt, dt)

# Velocity array, initialized with the initial velocity
v = np.zeros_like(t)
v[0] = v0

# Euler method to solve for velocity at each time step
for i in range(1, len(t)):
    dvdt = P / (m * v[i-1])  # derivative of velocity
    v[i] = v[i-1] + dvdt * dt  # update velocity using Euler's method

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(t, v)
plt.title("Cyclist's Velocity Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.savefig("bicycle.png")
plt.show()