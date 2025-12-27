# Julia translation of NLS_Universality_Class.py
# This file implements the core routines to generate eigenvalues (k_j), compute
# norming constants (c_j), build and solve the linear system and produce the
# scaled NLS solution at t=0 on a grid in X.  It closely follows the logic of
# the original Python script but written in idiomatic Julia.
#
# Important packages used (install before running):
#   julia> import Pkg
#   julia> Pkg.add(["SpecialFunctions", "DelimitedFiles", "Distributions", "LinearAlgebra", "Plots"]) 
#
# How to run:
#   1) Open a Julia REPL in the repository root (the directory that contains src/).
#   2) Make sure packages are installed (see commands above).
#   3) From the REPL, do:
#         julia> include("src/NLS_Universality_Class.jl")
#         julia> main()            # runs the example simulations
#
# The code is commented heavily to help users unfamiliar with Julia.  Types
# use BigFloat for extra precision similarly to mpmath's mp.  You can change
# the precision with `setprecision` below.

# -------------------- Imports and precision --------------------
using LinearAlgebra
using SpecialFunctions   # for gamma if needed in distributions
using DelimitedFiles     # for simple CSV output (writedlm)
using Distributions
using Random
using Plots
# Threads lives in Base.Threads (part of Julia stdlib). Do NOT Pkg.add("Threads").
# We will refer to it via Base.Threads (e.g. Base.Threads.@threads, Base.Threads.nthreads()).

# Increase BigFloat precision (bits). 256 bits ~ 77 decimal digits.
setprecision(256)

# Type alias for complex high-precision numbers
const CBig = Complex{BigFloat}

# -------------------- Utility: convert numeric -> BigFloat --------------------
# Helper that turns a real number to BigFloat and then to complex
to_big(x::Real) = BigFloat(x)
to_cbig(x::Real) = CBig(to_big(x), to_big(0))

# -------------------- Generate eigenvalues k_j --------------------
# We accept two functions dist_real() and dist_imag() that produce real numbers
# (Float64 or BigFloat); we convert to BigFloat and build complex eigenvalues.
function generate_ks(N::Int, dist_real::Function, dist_imag::Function)
    ks = Vector{CBig}(undef, N)
    for j in 1:N
        re = to_big(dist_real())
        im = to_big(dist_imag())
        ks[j] = CBig(re, im)
    end
    return ks
end

# -------------------- Compute norming constants c_j --------------------
# c_j = prod_{n}(k_j - conj(k_n)) / prod_{n != j}(k_j - k_n)
function compute_cs(ks::Vector{CBig})
    N = length(ks)
    cs = Vector{CBig}(undef, N)
    for j in 1:N
        k_j = ks[j]
        num = CBig(1,0)
        den = CBig(1,0)
        for n in 1:N
            num *= (k_j - conj(ks[n]))
            if n != j
                den *= (k_j - ks[n])
            end
        end
        cs[j] = num / den
    end
    return cs
end

# -------------------- Phase theta(k, x, t) --------------------
# In Python: theta = 2*k*x + 2*k^2*t
#
# NOTE: x and t should be real numbers. Accept Real inputs for x and t
# and convert them to BigFloat internally. This avoids method errors when
# get_indices passes BigFloat (not complex) values for x and t.
function theta(k::CBig, x::Real, t::Real)
    x_b = to_big(x)
    t_b = to_big(t)
    return 2 * k * x_b + 2 * (k^2) * t_b
end

# -------------------- Classify indices into unstable/stable --------------------
# Compute Cexp_j = c_j * exp(i * theta_j) and split indices where |Cexp|>1
function get_indices(ks::Vector{CBig}, cs::Vector{CBig}, x::Real, t::Real)
    N = length(ks)
    Cexp = Vector{CBig}(undef, N)
    x_b = to_big(x)
    t_b = to_big(t)
    for j in 1:N
        Cexp[j] = cs[j] * exp(im * theta(ks[j], x_b, t_b))
    end
    idx_is = Int[]   # unstable (inverse stable in original naming)
    idx_isc = Int[]  # stable
    for j in 1:N
        if abs(Cexp[j]) > one(BigFloat)
            push!(idx_is, j)
        else
            push!(idx_isc, j)
        end
    end
    return idx_is, idx_isc, Cexp
end

# -------------------- Helper q functions used for diagonal entries --------------------
function q_func(k::CBig, ks::Vector{CBig}, idx_is::Vector{Int})
    res = CBig(1,0)
    for j in idx_is
        res *= (k - ks[j]) / (k - conj(ks[j]))
    end
    return res
end

function quhp_func(k::CBig, ks::Vector{CBig}, idx_is::Vector{Int}, l_idx::Int)
    res = CBig(1,0)
    for j in idx_is
        if l_idx != j
            res *= (k - ks[j]) / (k - conj(ks[j]))
        else
            res *= inv(k - conj(ks[j]))
        end
    end
    return res
end

function qlhp_func(k::CBig, ks::Vector{CBig}, idx_is::Vector{Int}, l_idx::Int)
    res = CBig(1,0)
    for j in idx_is
        if l_idx != j
            res *= (k - ks[j]) / (k - conj(ks[j]))
        else
            res *= (k - ks[j])
        end
    end
    return res
end

# -------------------- Solve linear system for one (x,t) point --------------------
# Returns complex value of psi at (x,t)
function solve_nls_point(ks::Vector{CBig}, cs::Vector{CBig}, x::Real, t::Real)
    N = length(ks)
    idx_is, idx_isc, Cexp = get_indices(ks, cs, x, t)

    # Build diag1 and diag2 vectors (length N each)
    diag1 = [CBig(0,0) for _ in 1:N]
    diag2 = [CBig(0,0) for _ in 1:N]

    for i in idx_isc
        qv = q_func(ks[i], ks, idx_is)
        diag1[i] = (qv^2) * Cexp[i]
    end

    for i in idx_is
        qv = quhp_func(ks[i], ks, idx_is, i)
        diag1[i] = inv((qv^2) * Cexp[i])
    end

    for i in idx_isc
        kval = conj(ks[i])
        qv = q_func(kval, ks, idx_is)
        diag2[i] = - (qv^(-2)) * conj(Cexp[i])
    end

    for i in idx_is
        kval = conj(ks[i])
        qv = qlhp_func(kval, ks, idx_is, i)
        diag2[i] = - (qv^2) / conj(Cexp[i])
    end

    D_vec = vcat(diag1, diag2)  # length 2N

    # Build H matrix (2N x 2N)
    H = Matrix{CBig}(undef, 2N, 2N)
    # Fill with zeros to be safe
    for r in 1:2N, c in 1:2N
        H[r,c] = CBig(0,0)
    end

    for i in 1:N
        for j in 1:N
            cross_set = (i in idx_isc && j in idx_is) || (i in idx_is && j in idx_isc)
            same_set = (i in idx_isc && j in idx_isc) || (i in idx_is && j in idx_is)

            val11 = CBig(0,0)
            val12 = CBig(0,0)
            val21 = CBig(0,0)
            val22 = CBig(0,0)

            if cross_set
                val11 = inv(ks[i] - ks[j])
                val22 = conj(val11)
            end
            if same_set
                val12 = inv(ks[i] - conj(ks[j]))
                val21 = inv(conj(ks[i]) - ks[j])
            end

            H[i, j] = val11
            H[i, j+N] = val12
            H[i+N, j] = val21
            H[i+N, j+N] = val22
        end
    end

    # Build RHS and A
    b = zeros(CBig, 2N)
    for i in 1:N
        if i in idx_is
            b[i] = CBig(1,0)
        end
        if i in idx_isc
            b[i+N] = CBig(1,0)
        end
    end

    A = Matrix{CBig}(undef, 2N, 2N)
    RHS = zeros(CBig, 2N)

    for r in 1:2N
        RHS[r] = D_vec[r] * b[r]
        for c in 1:2N
            val = -D_vec[r] * H[r,c]
            if r == c
                val += CBig(1,0)
            end
            A[r,c] = val
        end
    end

    # Solve A * u = RHS
    # Use LU factorization for complex BigFloat matrices
    # Predeclare sol_vec so it's always defined in this function scope.
    sol_vec = zeros(CBig, 2N)
    try
        sol_vec = A \ RHS
    catch err
        # If solve fails, print diagnostics and return zero for this point
        println("  [solve_nls_point] Linear solve failed for N=$(N), matrix size=$(size(A)). Error: ", err)
        return CBig(0,0)
    end

    u1 = sol_vec[1:N]
    u2 = sol_vec[N+1:2N]

    total_sum = CBig(0,0)
    for i in idx_isc
        total_sum += u2[i]
    end
    for i in idx_is
        total_sum += u1[i]
    end

    return 2 * im * total_sum
end

# -------------------- Run solver loop across scaled X --------------------
# Returns array of tuples (x_scaled, amplitude)
function run_solver_loop(N::Int, ks::Vector{CBig}, label::String, mu_theoretical::Real; plot_points::Int=200)
    println("[$label] Computing cs constants...")
    cs = compute_cs(ks)

    println("[$label] Running NLS solver loop for N=$N using $(Base.Threads.nthreads()) threads...")

    scale_factor = 2 / (N * to_big(mu_theoretical))

    xs_scaled = collect(range(-4.0, stop=4.0, length=plot_points))
    results = Vector{Tuple{Float64,Float64}}(undef, length(xs_scaled))

    # Atomic counter for progress reporting (use Base.Threads)
    counter = Base.Threads.Atomic{Int}(0)

    Base.Threads.@threads for idx in eachindex(xs_scaled)
        xx = xs_scaled[idx]
        x_phys = xx * scale_factor
        sol = solve_nls_point(ks, cs, x_phys, 0.0)
        amp = Float64(abs(sol) * scale_factor)
        results[idx] = (Float64(xx), amp)

        # Update progress periodically (use atomic op from Base.Threads)
        v = Base.Threads.atomic_add!(counter, 1)
        if v % 50 == 0
            # Print from any thread (may interleave) — sufficient for coarse progress
            println("  Progress: $v/$(length(xs_scaled))")
        end
    end

    return results
end

# -------------------- Random experiments for PIII and PV --------------------
# The distributions are passed as functions returning Float64 values.
function random_PIII(N_soliton::Int; nu::Int=3, trial::Int=0, show_plot::Bool=false)
    # real part ~ Normal(0, 3)
    real_dist() = 3.0 * randn()
    # imaginary part ~ sqrt(ChiSq(nu))
    ch = Chisq(nu)
    imag_dist() = sqrt(rand(ch))

    ks = generate_ks(N_soliton, real_dist, imag_dist)

    # theoretical mean of Chi distribution for sqrt(ChiSq(nu))
    mu_piii_exact = sqrt(2) * gamma((nu + 1) / 2) / gamma(nu / 2)

    data = run_solver_loop(N_soliton, ks, "PIII", mu_piii_exact)

    filename = string(N_soliton, "SolitonsPIII_random_", lpad(string(trial), 5, '0'), ".csv")
    writedlm(filename, [("x_scaled", "amplitude")]) # header-like row
    open(filename, "a") do io
        for (xv, av) in data
            println(io, "$(xv),$(av)")
        end
    end
    println("Exported data to $(filename)")

    if show_plot
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        plot(xs, ys, title="NLS Random PIII (N=$(N_soliton), nu=$(nu))", xlabel="X", ylabel="|Psi|")
        display(current())
    end
end

function random_PV(N_soliton::Int; trial::Int=0, show_plot::Bool=false)
    # Example distributions: real Uniform(-2,2), imag Uniform(1,3)
    real_dist() = rand()*4.0 - 2.0
    imag_dist() = 1.0 + 2.0*rand()

    ks = generate_ks(N_soliton, real_dist, imag_dist)
    mu_pv_exact = (1.0 + 3.0) / 2.0

    data = run_solver_loop(N_soliton, ks, "PV", mu_pv_exact)

    filename = string(N_soliton, "SolitonsPV_random_", lpad(string(trial), 5, '0'), ".csv")
    writedlm(filename, [("x_scaled", "amplitude")])
    open(filename, "a") do io
        for (xv, av) in data
            println(io, "$(xv),$(av)")
        end
    end
    println("Exported data to $(filename)")

    if show_plot
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        plot(xs, ys, title="NLS Random PV (N=$(N_soliton))", xlabel="X", ylabel="|Psi|")
        display(current())
    end
end

# -------------------- Helpers for averaging --------------------
function _linear_interp(xs::Vector{Float64}, ys::Vector{Float64}, xt::Vector{Float64})
    # Simple linear interpolation (xs must be sorted)
    n = length(xt)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        x = xt[i]
        if x <= xs[1]
            out[i] = ys[1]
        elseif x >= xs[end]
            out[i] = ys[end]
        else
            j = searchsortedfirst(xs, x)
            # xs[j-1] < x <= xs[j]
            x0, x1 = xs[j-1], xs[j]
            y0, y1 = ys[j-1], ys[j]
            t = (x - x0) / (x1 - x0)
            out[i] = y0 + t * (y1 - y0)
        end
    end
    return out
end

function run_trials_and_average(dataset::String, K::Int, N_soliton::Int; nu::Int=3, out_filename::String="average.csv")
    # Runs K trials for `dataset` ("PIII" or "PV"), saves each trial file,
    # then reads them back, averages the amplitude column and writes out_filename.
    files = String[]
    for k in 0:(K-1)
        if dataset == "PIII"
            random_PIII(N_soliton; nu=nu, trial=k, show_plot=false)
            push!(files, string(N_soliton, "SolitonsPIII_random_", lpad(string(k),5,'0'), ".csv"))
        elseif dataset == "PV"
            random_PV(N_soliton; trial=k, show_plot=false)
            push!(files, string(N_soliton, "SolitonsPV_random_", lpad(string(k),5,'0'), ".csv"))
        else
            error("Unknown dataset: $dataset. Use \"PIII\" or \"PV\".")
        end
    end

    # Read and collect amplitudes. Allow different x-grids by interpolating.
    x_ref = nothing
    amp_matrix = Float64[]
    for f in files
        lines = readlines(f)
        data_x = Float64[]
        data_a = Float64[]
        for (i, line) in enumerate(lines)
            if i == 1
                continue
            end
            parts = split(chomp(line), ',')
            if length(parts) < 2
                continue
            end
            push!(data_x, parse(Float64, strip(parts[1])))
            push!(data_a, parse(Float64, strip(parts[2])))
        end
        if x_ref === nothing
            x_ref = copy(data_x)
            amp_matrix = reshape(collect(data_a), :, 1)
        else
            if length(data_x) != length(x_ref) || any(abs.(data_x .- x_ref) .> 1e-8)
                # interpolate to x_ref
                vals = _linear_interp(data_x, data_a, x_ref)
            else
                vals = data_a
            end
            amp_matrix = hcat(amp_matrix, collect(vals))
        end
    end

    # Compute mean across columns (trials)
    mean_amp = vec(mean(amp_matrix, dims=2))

    # Write averaged file
    open(out_filename, "w") do io
        println(io, "x_scaled,amplitude_mean")
        for i in 1:length(x_ref)
            println(io, "$(x_ref[i]),$(mean_amp[i])")
        end
    end
    println("Wrote averaged data to $(out_filename)")
    return out_filename
end

# -------------------- Entry point example --------------------
function main()
    # Example: run two small PIII simulations; adjust N_soliton, nu, and trials as you like.
    nu = 3
    for k in 0:1
        println("Starting trial $(k) for PIII (nu=$(nu))")
        random_PIII(100; nu=nu, trial=k, show_plot=false)
    end

    # Example PV run (commented out by default)
    # random_PV(20, show_plot=false)
end

# When included, `main()` will not run automatically; call main() from the REPL as described above.

# End of file
