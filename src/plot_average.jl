# Simple Julia script to plot an averaged CSV of the form:
# x_scaled,amplitude_mean
#
# Usage (from Julia REPL in project root):
#   julia> include("src/plot_average.jl")
#   julia> plot_average_csv("PIII_avg_5trials.csv")
#
# This uses Plots.jl (already used in the project). If Plots.jl is not
# installed, run in the REPL:
#   import Pkg; Pkg.add("Plots")

using Plots

function plot_average_csv(filename::AbstractString; out::AbstractString="")
    # Read CSV manually (skip header), parse two columns of floats
    xs = Float64[]
    ys = Float64[]

    open(filename, "r") do io
        # skip header line
        header = readline(io)
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            parts = split(s, ',')
            if length(parts) < 2
                continue
            end
            try
                x = parse(Float64, strip(parts[1]))
                y = parse(Float64, strip(parts[2]))
            catch err
                @warn "Skipping line (parse error)" line= line err=err
                continue
            end
            push!(xs, x)
            push!(ys, y)
        end
    end

    if isempty(xs)
        println("No data found in $filename")
        return
    end

    if isempty(out)
        out = replace(filename, r"\.csv$" => "") * ".png"
    end

    p = plot(xs, ys,
        linewidth=2,
        xlabel = "X",
        ylabel = "|Ψ| (mean)",
        title = "$(filename)",
        legend = false,
        grid = true,
        size = (900, 500)
    )

    savefig(p, out)
    println("Saved plot to $out")
    return out
end

function _read_two_col_csv(filename::AbstractString)
    lines = readlines(filename)
    xs = Float64[]
    ys = Float64[]
    if isempty(lines)
        return xs, ys
    end
    # Determine if first line is header: try parse first token
    first_parts = split(strip(lines[1]), ',')
    is_header = true
    if length(first_parts) >= 2
        if tryparse(Float64, strip(first_parts[1])) !== nothing && tryparse(Float64, strip(first_parts[2])) !== nothing
            is_header = false
        end
    end
    start_idx = is_header ? 2 : 1
    for i in start_idx:length(lines)
        line = strip(lines[i])
        isempty(line) && continue
        parts = split(line, ',')
        if length(parts) < 2
            continue
        end
        a = tryparse(Float64, strip(parts[1]))
        b = tryparse(Float64, strip(parts[2]))
        if a === nothing || b === nothing
            continue
        end
        push!(xs, a)
        push!(ys, b)
    end
    return xs, ys
end

function _linear_interp(xs::Vector{Float64}, ys::Vector{Float64}, xt::Vector{Float64})
    # Simple linear interpolation: xs may be unsorted; returns values at xt.
    if length(xs) != length(ys) || isempty(xs)
        return Float64[]
    end
    order = sortperm(xs)
    xs_s = xs[order]
    ys_s = ys[order]
    n = length(xt)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        x = xt[i]
        if x <= xs_s[1]
            out[i] = ys_s[1]
        elseif x >= xs_s[end]
            out[i] = ys_s[end]
        else
            j = searchsortedfirst(xs_s, x)
            # xs_s[j-1] < x <= xs_s[j]
            x0 = xs_s[j-1]; x1 = xs_s[j]
            y0 = ys_s[j-1]; y1 = ys_s[j]
            t = (x - x0) / (x1 - x0)
            out[i] = y0 + t * (y1 - y0)
        end
    end
    return out
end

"""
Plot two CSV files (two columns x, y) on the same axes and save image.
If grids differ the second series is interpolated onto the first grid.
Example:
  plot_pair_csv("PIII_avg_5trials.csv", "RHP_PIII.csv")
"""
function plot_pair_csv(file1::AbstractString, file2::AbstractString; label1::AbstractString="", label2::AbstractString="", out::AbstractString="")
    xs1, ys1 = _read_two_col_csv(file1)
    xs2, ys2 = _read_two_col_csv(file2)
    if isempty(xs1) || isempty(xs2)
        println("One of the files has no data: $file1 or $file2")
        return
    end
    # Ensure xs1 is sorted
    order1 = sortperm(xs1)
    xs1s = xs1[order1]
    ys1s = ys1[order1]

    # Interpolate second onto xs1 if needed
    if length(xs1s) == length(xs2) && all(abs.(xs1s .- xs2) .< 1e-8)
        ys2_on_1 = ys2
    else
        ys2_on_1 = _linear_interp(xs2, ys2, xs1s)
    end

    lbl1 = isempty(label1) ? split(basename(file1), '/') |> last : label1
    lbl2 = isempty(label2) ? split(basename(file2), '/') |> last : label2

    # Default plotting styles: if a filename contains 'RHP' plot dashed black
    style2 = if occursin("RHP", uppercase(file2))
        (:black, :dash)
    else
        (:blue, :solid)
    end

    if isempty(out)
        out = replace(file1, r"\.csv$" => "") * "_vs_" * replace(file2, r".*/" => "")
        out = replace(out, r"\.csv$" => ".png")
    end

    p = plot(xs1s, ys1s, label=lbl1, linewidth=2)
    plot!(xs1s, ys2_on_1, label=lbl2, color=style2[1], linestyle=style2[2], linewidth=2)
    xlabel!(p, "X")
    ylabel!(p, "|Ψ|")
    title!(p, "$(basename(file1)) vs $(basename(file2))")
    plot!(p, grid=true)
    savefig(p, out)
    println("Saved comparison plot to $out")
    return out
end
