import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj, ellipk
import warnings

def calculate_leading_order_modulus(x, N, mag_A, theta):
    """
    Calculates the modulus of the leading order asymptotic term for SG.

    Parameters:
    x     : array-like or scalar, spatial coordinate
    N     : int, number of solitons (must be > 0)
    mag_A : float, magnitude of A (|A|)
    theta : float, argument of A (radians)
    """
    # Coerce inputs to numeric types / arrays
    x = np.asarray(x, dtype=float)
    if np.isscalar(x):
        # ensure array-like behavior for downstream ops
        x = np.array([x], dtype=float)

    if N <= 0:
        raise ValueError("N must be a positive integer")

    mag_A = float(mag_A)
    theta = float(theta)

    # 1. Calculate A components (kept for clarity though not used directly)
    # A = |A|e^{i\theta}
    # Re(A) = |A|cos(theta), Im(A) = |A|sin(theta)

    # 2. Calculate the Pre-factor Magnitude (same formula)
    prefactor_mag = mag_A * np.abs(np.sin(2.0 * theta))

    # 3. Setup Elliptic Parameters
    # Theorem uses modulus k = cos(theta).
    # SciPy uses parameter m = k**2 = cos^2(theta).
    k = np.cos(theta)
    m = float(k**2)
    # Clamp m to [0,1] to avoid tiny numerical excursion outside valid range
    m = np.clip(m, 0.0, 1.0)

    # Calculate Complete Elliptic Integral K(k) = K(m) in scipy
    K_val = ellipk(m)

    # 4. Calculate the Argument of sd(...)
    # arg = -2|A|x + (2 K(cos(theta))/pi) * ln(N)
    shift = (2.0 * K_val / np.pi) * np.log(float(N))
    u = -2.0 * mag_A * x + shift

    # 5. Calculate Jacobi Elliptic Function sd(u) = sn(u)/dn(u)
    sn, cn, dn, ph = ellipj(u, m)

    # Protect division by very small dn values which would produce inf/NaN
    eps = max(np.finfo(float).eps, 1e-16)
    # Preserve sign of dn when clamping
    dn_safe = np.where(np.abs(dn) < eps, np.copysign(eps, dn), dn)
    sd = sn / dn_safe

    # Warn if clamping occurred so user can be aware
    if np.any(np.abs(dn) < eps):
        warnings.warn("Small dn values encountered; applied small-floor to avoid division by zero.")

    # 6. Combine
    psi_modulus = prefactor_mag * np.abs(sd)

    # If original input was scalar, return scalar
    if psi_modulus.shape == (1,):
        return float(psi_modulus[0])
    return psi_modulus

def plot_asymptotics():
    # Define spatial domain. 
    # The solitons are located near x where the argument u ~ 0.
    # u = -2|A|x + shift => x_center = shift / (2|A|)
    # We will define a dynamic range for each plot or a broad fixed one.
    
    # Test Cases: (N, |A|, theta_label, theta_val)
    cases = [
        (10,  1.0, r"$\pi/4$", np.pi/4),
        (100, 1.0, r"$\pi/4$", np.pi/4),
        (100, 0.5, r"$\pi/4$", np.pi/4),
        (100, 1.0, r"$\pi/6$", np.pi/6),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(r"Modulus of Leading Order Asymptotic $|\psi_{SG}(x,t;N)|$", fontsize=16)
    
    for ax, (N, mag_A, theta_lbl, theta) in zip(axs.flat, cases):
        
        # Determine center to focus the plot
        m = np.cos(theta)**2
        K_val = ellipk(m)
        shift = (2 * K_val / np.pi) * np.log(N)
        center_x = shift / (2 * mag_A)
        
        # Create x range centered around the soliton train
        x = np.linspace(center_x - 10, center_x + 10, 1000)
        
        y = calculate_leading_order_modulus(x, N, mag_A, theta)
        
        ax.plot(x, y, label=f"N={N}, |A|={mag_A}, $\\theta$={theta_lbl}", color='teal')
        ax.set_title(f"N={N}, |A|={mag_A}, $\\theta$={theta_lbl}")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$|\psi_{SG}|$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_asymptotics()