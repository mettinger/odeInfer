##
using DifferentialEquations, Plots, LinearAlgebra, Polynomials
plotlyjs()

function parametersMake(z0)

    flag = 1

    if flag == 0
        A = [.3448, .0228, .2014]
        B = [.3351, .0746, .2053]
        c = .001
        dstar = .3
        xs = .55
    elseif flag == 1
        A = [.2552,-.0637,.3014]
        B = [.3496, .0795, .1774]
        c = .001
        dstar = .3
        xs = .6
    end

    R = norm(A)
    e = A/R
    crossProduct = cross(cross(A,B), A)
    f = crossProduct/norm(crossProduct)

    muTwo, minusMuOne, nu = R * ((e * cos(z0)) + (f * sin(z0)))
    allRoots = roots(Polynomial([minusMuOne, -muTwo, 0, 1]))

    [e, f, R, c, dstar, xs]
end


function saggioUpdate!(du, u, p, t)

    e, f, R, c, dstar, xs = p
    muTwo, minusMuOne, nu = R * ((e * cos(u[3])) + (f * sin(u[3])))

    du[1] = -u[2]
    du[2] = u[1]^3 - (muTwo * u[1]) + minusMuOne - (u[2] * (nu + u[1] + u[1]^2))
    du[3] = -c * (sqrt( (u[1] - xs)^2 + u[2]^2) - dstar)
end

function saggioSolve(u0, tspan, p)
    prob = ODEProblem(saggioUpdate!, u0, tspan, p)
    sol = solve(prob)
    sol
end


##
y0 = 0.0
z0 = 0.0
p = parametersMake(z0)
x0 = p[6]

u0 = [x0, y0, z0]
tspan = (0., 6000.)


sol = saggioSolve(u0, tspan, p)

xt = plot(sol, vars=(1))
zt = plot(sol, vars=(3))
plot(xt,zt, layout = (2,1))

