using DynamicalSystems

lo = Systems.lorenz([1.0, 1.0, 50.0])
tr = trajectory(lo, 100; Δt = 0.01, Ttr = 10)

s = vec(tr[:, 1]) # input timeseries = x component of Lorenz
theiler = estimate_delay(s, "mi_min") # estimate a Theiler window
Tmax = 100 # maximum possible delay

Y, τ_vals, ts_vals, Ls, εs = pecuzal_embedding(s; τs = 0:Tmax , w = theiler, econ = true)

println("τ_vals = ", τ_vals)
println("Ls = ", Ls)
println("L_total_uni: $(sum(Ls))")
print(Matrix(Y))