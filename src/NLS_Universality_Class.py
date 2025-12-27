import numpy as np
import matplotlib.pyplot as plt
import csv
from mpmath import mp
from scipy.special import gamma

# --- CONFIGURATION ---
# Set high precision (50 digits) for ill-conditioned matrices (N > 15)
mp.dps = 50 

def generate_ks(N, dist_real, dist_imag):
    """
    Generates eigenvalues k_j = v_j + i*mu_j from the provided distribution functions.
    """
    ks = []
    for _ in range(N):
        re = dist_real()
        im = dist_imag()
        ks.append(mp.mpc(re, im))
    return ks

def compute_cs(ks):
    """
    Computes the norming constants c_j for the maximal solution (condensate).
    """
    N = len(ks)
    cs = []
    for j in range(N):
        k_j = ks[j]
        num = mp.mpc(1, 0)
        den = mp.mpc(1, 0)
        
        for n in range(N):
            # Numerator: Product over all n of (k_j - conj(k_n))
            num *= (k_j - mp.conj(ks[n]))
            # Denominator: Product over all n != j of (k_j - k_n)
            if n != j:
                den *= (k_j - ks[n])
            
        cs.append(num / den)
    return cs

def theta(k, x, t):
    return 2 * k * x + 2 * (k**2) * t

def get_indices(ks, cs, x, t):
    """
    Separates indices into Unstable (Abs > 1) and Stable (Abs <= 1) sets.
    """
    N = len(ks)
    Cexp = []
    
    # Calculate Cexp_j = c_j * exp(i * theta_j)
    for j in range(N):
        val = cs[j] * mp.exp(mp.j * theta(ks[j], x, t))
        Cexp.append(val)
        
    idx_is = []   # Unstable (Inverse Stable)
    idx_isc = []  # Stable
    
    for j in range(N):
        if abs(Cexp[j]) > 1:
            idx_is.append(j)
        else:
            idx_isc.append(j)
            
    return idx_is, idx_isc, Cexp

# --- Helper functions for diagonal entries ---
def q_func(k, ks, idx_is):
    res = mp.mpc(1, 0)
    for j in idx_is:
        res *= (k - ks[j]) / (k - mp.conj(ks[j]))
    return res

def quhp_func(k, ks, idx_is, l_idx):
    res = mp.mpc(1, 0)
    for j in idx_is:
        if l_idx != j:
            res *= (k - ks[j]) / (k - mp.conj(ks[j]))
        else:
            res *= 1 / (k - mp.conj(ks[j]))
    return res

def qlhp_func(k, ks, idx_is, l_idx):
    res = mp.mpc(1, 0)
    for j in idx_is:
        if l_idx != j:
            res *= (k - ks[j]) / (k - mp.conj(ks[j]))
        else:
            res *= (k - ks[j])
    return res

def solve_nls_point(ks, cs, x, t):
    """
    Solves the linear system for the NLS solution at a specific (x, t).
    """
    N = len(ks)
    idx_is, idx_isc, Cexp = get_indices(ks, cs, x, t)
    
    # 1. Build Diagonal D vector
    diag1 = [mp.mpc(0,0)] * N
    diag2 = [mp.mpc(0,0)] * N
    
    for i in idx_isc:
        q_val = q_func(ks[i], ks, idx_is)
        diag1[i] = (q_val**2) * Cexp[i]
        
    for i in idx_is:
        q_val = quhp_func(ks[i], ks, idx_is, i)
        diag1[i] = 1 / ((q_val**2) * Cexp[i])
        
    for i in idx_isc:
        k_val = mp.conj(ks[i])
        q_val = q_func(k_val, ks, idx_is)
        diag2[i] = - (q_val**(-2)) * mp.conj(Cexp[i])
        
    for i in idx_is:
        k_val = mp.conj(ks[i])
        q_val = qlhp_func(k_val, ks, idx_is, i)
        diag2[i] = - (q_val**2) / mp.conj(Cexp[i])
        
    D_vec = diag1 + diag2 

    # 2. Build H Matrices Blocks
    # Optimized construction to avoid redundant checks
    H_mat = mp.matrix(2*N, 2*N)
    
    for i in range(N):
        for j in range(N):
            cross_set = (i in idx_isc and j in idx_is) or (i in idx_is and j in idx_isc)
            same_set = (i in idx_isc and j in idx_isc) or (i in idx_is and j in idx_is)
            
            val11, val12, val21, val22 = 0, 0, 0, 0
            
            if cross_set:
                val11 = 1 / (ks[i] - ks[j])
                val22 = mp.conj(val11)
            
            if same_set:
                val12 = 1 / (ks[i] - mp.conj(ks[j]))
                val21 = 1 / (mp.conj(ks[i]) - ks[j])
            
            # Fill block matrix directly
            H_mat[i, j] = val11
            H_mat[i, j+N] = val12
            H_mat[i+N, j] = val21
            H_mat[i+N, j+N] = val22

    # 3. Construct System A*u = RHS
    b = mp.matrix(2*N, 1)
    for i in range(N):
        if i in idx_is: b[i] = 1
        if i in idx_isc: b[i+N] = 1

    A = mp.matrix(2*N, 2*N)
    RHS = mp.matrix(2*N, 1)
    
    for r in range(2*N):
        RHS[r] = D_vec[r] * b[r]
        for c in range(2*N):
            val = -D_vec[r] * H_mat[r, c]
            if r == c: val += 1
            A[r, c] = val

    try:
        u = mp.lu_solve(A, RHS)
    except:
        return mp.mpc(0, 0)

    # 4. Sum solution components
    u1 = u[0:N]
    u2 = u[N:2*N]
    
    total_sum = mp.mpc(0, 0)
    for i in idx_isc: total_sum += u2[i]
    for i in idx_is:  total_sum += u1[i]
        
    return 2 * mp.j * total_sum

def run_solver_loop(N, ks, label, mu_theoretical):
    """
    Solves the NLS for a range of x using the specified theoretical mean for scaling.
    """
    print(f"[{label}] Computing cs constants...")
    cs = compute_cs(ks)
    
    print(f"[{label}] Running NLS solver loop for N={N}...")
    
    # Scale factor uses the THEORETICAL mean (mu_theoretical)
    scale_factor = 2 / (N * mu_theoretical)
    
    # Define range for scaled position x
    plot_points = 200
    xs_scaled = np.linspace(-4, 4, plot_points)
    results = []
    
    for i, xx in enumerate(xs_scaled):
        # Physical x is rescaled: x_phys = (2 * x_scaled) / (N * mu)
        x_phys = xx * scale_factor
        
        # Solve at t=0
        sol = solve_nls_point(ks, cs, x_phys, 0)
        
        # Scaled amplitude
        amp = float(abs(sol) * scale_factor)
        results.append((xx, amp))
        
        if i % 50 == 0:
            print(f"  Progress: {i}/{plot_points}")
            
    return results

def random_PIII(N_soliton, dist_real, dist_imag, mu_imag, trial=0, plot=False):
    # 1. Generate Data
    ks = generate_ks(N_soliton, dist_real, dist_imag)
    
    # 2. Run Simulation (Pass theoretical mean)
    data = run_solver_loop(N_soliton, ks, "PIII", mu_imag)
    
    # 3. Export
    filename = f"{N_soliton}SolitonsPIII_random_%05d.csv" % trial
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x_scaled", "amplitude"])
        writer.writerows(data)
    print(f"Exported data to {filename}")
    
    # 4. Plot
    if plot:
        x_val, y_val = zip(*data)
        plt.figure(figsize=(10, 6))
        plt.plot(x_val, y_val)
        plt.title(rf"NLS Random PIII (N={N_soliton}, $\nu$=3)")
        plt.xlabel(r"$X$")
        plt.ylabel(r"$|\Psi_{N,III}(X,0)|$")
        plt.grid(True)
        plt.show()

def random_PV(N_soliton, dist_real, dist_imag, mu_imag):
    # 1. Generate Data
    ks = generate_ks(N_soliton, dist_real, dist_imag)
    
    # 2. Run Simulation (Pass theoretical mean)
    data = run_solver_loop(N_soliton, ks, "PV", mu_imag)
    
    # 3. Export
    filename = f"{N_soliton}SolitonsPV_random.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x_scaled", "amplitude"])
        writer.writerows(data)
    print(f"Exported data to {filename}")
    
    # 4. Plot
    x_val, y_val = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.plot(x_val, y_val)
    plt.title(rf"NLS Random PV (N={N_soliton}, $\zeta=-\frac{1}{2}$)")
    plt.xlabel(r"$X$")
    plt.ylabel(r"$|\Psi_{N,PV}(X,0)|$")
    plt.grid(True)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # ---------------- PIII CONFIGURATION ----------------
    # Parameter for Chi distribution degrees of freedom
    nu = 3
    
    def piii_real(): 
        return np.random.normal(0, 3)
    
    def piii_imag(): 
        # Using the parameter nu as requested
        return np.sqrt(np.random.chisquare(nu))

    # Exact Theoretical Mean for sqrt(ChiSq(nu)) = Chi(nu) distribution
    # Mean = sqrt(2) * Gamma((nu + 1) / 2) / Gamma(nu / 2)
    mu_piii_exact = np.sqrt(2) * gamma((nu + 1) / 2) / gamma(nu / 2)

    for k in range(2):  # Run a single simulation; increase range for more runs
        print(f"Starting {k} PIII Simulation (Exact Mean = {mu_piii_exact:.6f})...")
        random_PIII(N_soliton=100, dist_real=piii_real, dist_imag=piii_imag, mu_imag=mu_piii_exact, trial=k)

#     # ---------------- PV CONFIGURATION ------------------
#     # Example PV: Real Uniform(-2,2), Imag Uniform(1,3)
#     def pv_real(): 
#         return np.random.uniform(-2, 2)
    
#     def pv_imag(): 
#         return np.random.uniform(1, 3)
    
#     # Exact Theoretical Mean for Uniform(a, b) = (a + b) / 2
#     mu_pv_exact = (1.0 + 3.0) / 2.0
    
#     print(f"\nStarting PV Simulation (Exact Mean = {mu_pv_exact:.6f})...")
#     random_PV(N_soliton=20, dist_real=pv_real, dist_imag=pv_imag, mu_imag=mu_pv_exact)