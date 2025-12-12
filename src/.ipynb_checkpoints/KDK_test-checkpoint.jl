using Distributed
addprocs()
using RogueWaveInfiniteNLS
@everywhere Xlist = collect(-16:0.05:16)
@everywhere atest = 0.8im
@everywhere btest = 1
@everywhere compute_psi_slice = X->psi(X,0,atest,btest,1)
outpsiX = pmap(compute_psi_slice, Xlist)

using Plots
using LaTeXStrings
abs_outpsiX = abs.(outpsiX)
plot(Xlist, abs_outpsiX, label=L"|\Psi(X,0,a=0.8i,b=1,\beta=1)|", xlabel=L"X", ylabel=L"|\Psi|", legend=:topright, linewidth=2.5, linecolor="darkblue")