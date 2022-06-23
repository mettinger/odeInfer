using DynamicalSystems


function lorenzTrajectory()
    dt = .002
    x0 = [-8., 8., 27.]
    totalTime = 100
    transientTime = 0

    lorenz = Systems.lorenz(x0)
    tr = trajectory(lorenz, totalTime; Δt = dt, Ttr = transientTime)
    return tr, dt
end

function embed(traj, w, Tmax)
    Y, τ_vals, ts_vals, Ls, εs = pecuzal_embedding(traj; τs = 0:Tmax, w=w, econ=true)
    return Matrix(Y), τ_vals, ts_vals, traj
end

function lorenzEmbed(index)

    tr, dt = lorenzTrajectory()

    s = vec(tr[:, index])
    theiler = estimate_delay(s, "mi_min")
    Tmax = 100
    Y, τ_vals, ts_vals, traj = embed(s, theiler, Tmax)
    return Y, τ_vals, ts_vals, traj, dt
end
