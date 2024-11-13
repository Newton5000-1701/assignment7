import numpy as np
import matplotlib.pyplot as plt

# Constants for the damped harmonic oscillator
m = 2
k = 3
y0 = 2
V0 = 0
t_max = 20
dt = 0.01
timesteps = int(t_max / dt)

# Natural frequency
omega = np.sqrt(k / m)


damping_cases = {
    "No Damping": 0,
    "Underdamping": 2,
    "Critical Damping": 2 * omega,
    "Overdamping": 3
}


def dydt(y, V):
    return V

def dVdt(y, V, gamma):
    return -omega**2 * y - gamma * V

# Euler Integrator
def euler_integrator(gamma):
    y = np.zeros(timesteps)
    V = np.zeros(timesteps)
    y[0], V[0] = y0, V0
    
    for n in range(1, timesteps):
        y[n] = y[n-1] + dydt(y[n-1], V[n-1]) * dt
        V[n] = V[n-1] + dVdt(y[n-1], V[n-1], gamma) * dt
    
    return y

# RK2 Integrator
def rk2_integrator(gamma):
    y = np.zeros(timesteps)
    V = np.zeros(timesteps)
    y[0], V[0] = y0, V0

    for n in range(1, timesteps):
        k1_y = dydt(y[n-1], V[n-1]) * dt
        k1_V = dVdt(y[n-1], V[n-1], gamma) * dt

        k2_y = dydt(y[n-1] + k1_y / 2, V[n-1] + k1_V / 2) * dt
        k2_V = dVdt(y[n-1] + k1_y / 2, V[n-1] + k1_V / 2, gamma) * dt

        y[n] = y[n-1] + k2_y
        V[n] = V[n-1] + k2_V

    return y

# Analytical solutions
def analytical_no_damping(t):
    return 2 * np.cos(omega * t)

def analytical_critical_damping(t, omega):
    return 2 * np.exp(-omega * t) * (t*omega + 1)

def analytical_overdamping(t, gamma):
    term1 = 2 * np.exp(-gamma * t / 2)
    term2 = np.cosh((t / 2) * np.sqrt(gamma**2 - 4 * omega**2))
    term3 = (gamma / np.sqrt(gamma**2 - 4 * omega**2)) * np.sinh((t / 2) * np.sqrt(gamma**2 - 4 * omega**2))
    return term1 * (term2 + term3)

def analytical_underdamping(t, gamma):
    term1 = 2 * np.exp(-gamma * t / 2)
    term2 = np.cos((t / 2) * np.sqrt(4 * omega**2 - gamma**2))
    term3 = (gamma / np.sqrt(4 * omega**2 - gamma**2)) * np.sin((t / 2) * np.sqrt(4 * omega**2 - gamma**2))
    return term1 * (term2 + term3)


time = np.arange(0, t_max, dt)


for label, gamma in damping_cases.items():
    # Numerical solutions
    y_euler = euler_integrator(gamma)
    y_rk2 = rk2_integrator(gamma)

  
    if label == "No Damping":
        y_analytical = analytical_no_damping(time)
        # Plot with sidebar for No Damping
        plt.figure(figsize=(10, 6))
        plt.plot(time, y_euler, label='Euler', color='b')
        plt.plot(time, y_rk2, label='RK2', color='purple')
        plt.plot(time, y_analytical, label='Analytical', color='g', linestyle='--')
        
       
        plt.title(f"Damped Harmonic Oscillator - {label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (y)")
        
       
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
       
        
    else:
        # Numerical solutions
        y_analytical = (analytical_critical_damping(time, omega) if label == "Critical Damping" else
                        analytical_overdamping(time, gamma) if label == "Overdamping" else
                        analytical_underdamping(time, gamma))

        # Plot without sidebar for other cases
        plt.figure(figsize=(10, 6))
        plt.plot(time, y_euler, label='Euler', color='b')
        plt.plot(time, y_rk2, label='RK2', color='purple')
        plt.plot(time, y_analytical, label='Analytical', color='g', linestyle='--')
        
        # Title and labels
        plt.title(f"Damped Harmonic Oscillator - {label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (y)")

        # Place the legend on the main plot
        plt.legend(loc="upper right")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{label.replace(' ', '_').lower()}_comparison.png")
    plt.show()