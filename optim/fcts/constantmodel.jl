# Population prediction for the Eleveld data set. Constant PK parameters throughout the population

# Date: 230105


using CSV, DataFrames, Optim, Plots, Random, LineSearches
# using JLD2

include("get_elevelddata.jl")
# include("fcts_trainingdata.jl")
include("fastpksim.jl")
# include("pkmodels.jl")

seed = 1234
Random.seed!(seed)
x_all, y_all = getelevelddata()

np = length(x_all)
startp = 1
x = x_all[startp:startp+np-1]
y = y_all[startp:startp+np-1]


function cost(costfct, x::InputData, y, ϕhat_scaled)
    u = x.u
    v = x.v
    hs = x.hs

    ϕhat = ϕhat_scaled .* x.normalization

    V1inv, λ, λinv, R = PK3(ϕhat)

    totalloss = 0.0
    # losses = zeros(eltype(ϕhat_scaled[1]),length(y),)
    j = 1 # counter to keep track of next free spot in y
    x_state = zeros(eltype(u), 3) # initial state
    for i in eachindex(hs)
        if i in x.youts # if we want to compute output
            x_state, yi = @inbounds updatestateoutput(x_state, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            # totalloss += abs2(yi - y[j]) # mse loss. compute only when we have observations
            totalloss += costfct(y[j], yi)
            # losses[j] = costfct(y[j],yi)
            j += 1
        else
            x_state = @inbounds updatestate(x_state, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return totalloss/length(y) # sum(losses) #
end

# compute_squarederror(y, yhat) = abs2(y - yhat)

function cost(costfct, x, y, ϕhat_scaled)
    totalloss = 0.0
    # losses = zeros(eltype(ϕhat_scaled[1]), length(y),)
    for i in eachindex(y)
        totalloss += cost(costfct, x[i], y[i], ϕhat_scaled)
    end
    return totalloss / length(y)
end

# cost(compute_squarederror,x, y, ϕ0)

lower = 1e-10 * ones(Float64, 6)
upper = 10 * ones(Float64, 6) #[Inf, Inf, Inf, Inf, Inf, Inf]
ϕ0 = 0.5 * ones(Float64, 6)
# optimizer = GradientDescent()
optimizer = LBFGS(linesearch=BackTracking(order=2))
# ϕhats_mse = Vector{Vector{Float64}}(undef, length(y))

# @time res = Optim.optimize(z -> cost(compute_squarederror, x, y, z), lower, upper, ϕ0, Fminbox(optimizer), Optim.Options(show_every=10, time_limit=60.0); autodiff=:forward) # short version



# Optim.minimizer(res)
# Optim.minimum(res)

# ϕhat_mse = Optim.minimizer(res) #.* x[1].normalization # final ϕ
# cost_final_mse = Optim.minimum(res) # final cost

# jldsave("jld/poppred.jld2"; res)

##### ALE error

compute_ALE(y, yhat) = abs(log(y / abs(yhat)))

ϕhats_ale = Vector{Vector{Float64}}(undef, length(y))

@time res = Optim.optimize(z -> cost(compute_ALE, x, y, z), lower, upper, ϕ0, Fminbox(optimizer), Optim.Options(show_every=10, time_limit=120.0); autodiff=:forward) # short version

ϕhat_ale = Optim.minimizer(res) #.* x[1].normalization # final ϕ
cost_final_ale = Optim.minimum(res) # final cost