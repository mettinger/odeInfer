#%%
using PyCall
using DifferentialEquations, Plots
plotlyjs()

#%%
@pyimport importlib.util as util
cd("C:/Users/the_m/github/uFlorida/python/odeInfer")
path = "nnAsFunction.py"

spec=util.spec_from_file_location("dummy", path)
nnVectorField = util.module_from_spec(spec)
spec.loader.exec_module(nnVectorField)


#print(foo.modelEval((0,0)))

function limitCycle!(du, u, p, t)

    result = nnVectorField.modelEval((u[1], u[2]))

    du[1] = result[1]
    du[2] = result[2]
end

y0 = 0.0
x0 = .5

u0 = [x0, y0]
tspan = (0., 100.)

prob = ODEProblem(limitCycle!, u0, tspan, [])
sol = solve(prob)

x = [i[1] for i in sol.u]
y = [i[2] for i in sol.u]
plot(x, y, seriestype = :scatter)

#=
xt = plot(sol, vars=(1))
yt = plot(sol, vars=(2))
plot(xt,yt, layout = (2,1))
=#