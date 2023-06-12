# Individual predictions (individual sets of PK parameters) for the Eleveld data set.
# MSE and ALE loss
# Date: 230125

# using Pkg
# cd(@__DIR__)
# Pkg.activate()

# Date: 230105


using CSV, DataFrames, Optim, Plots, Random
using LineSearches

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


# ϕ0 = ones(Float32, 6) # initial guess

function cost(costfct, x::InputData, y, ϕhat_scaled)
    u = x.u
    v = x.v
    hs = x.hs

    ϕhat = ϕhat_scaled .* x.normalization

    V1inv, λ, λinv, R = PK3(ϕhat)

    totalloss = 0.0
    losses = zeros(eltype(ϕhat_scaled[1]), length(y),)
    j = 1 # counter to keep track of next free spot in y
    x_state = zeros(eltype(u), 3) # initial state
    for i in eachindex(hs)
        if i in x.youts # if we want to compute output
            x_state, yi = @inbounds updatestateoutput(x_state, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            # totalloss += abs2(yi - y[j]) # mse loss. compute only when we have observations
            totalloss += costfct(y[j], yi)
            # losses[j] = costfct(y[j], yi)
            j += 1
        else
            x_state = @inbounds updatestate(x_state, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return totalloss / length(y) #median(losses) # sum(losses) #
    # return median(losses) # sum(losses) #

end


# lower = 1e-10 * ones(Float64, 6) # Set lowe limit (non-zero)
# upper = 10 * ones(Float64, 6) # Set upper limit
# ϕ0 = 0.5 * ones(Float64, 6) # Initial guess
# optimizer = LBFGS(linesearch=BackTracking(order=2)) # Optimizer
# ϕhats_mse = Vector{Vector{Float64}}(undef, length(y)) # Store result

# @time Threads.@threads for i in eachindex(y)
#     res = Optim.optimize(z -> cost(compute_squarederror, x[i], y[i], z), lower, upper, ϕ0, Fminbox(optimizer), Optim.Options(time_limit=10.0); autodiff=:forward)
#     ϕhats_mse[i] = Optim.minimizer(res)
# end

# #Check for NaN
# for i in eachindex(ϕhats_mse)
#     if any(isnan, ϕhats_mse[i])
#         @show i
#     end
# end


##############################################################################################################
# Loss function, ALE instead


compute_ALE(y, yhat) = abs(log(y / abs(yhat)))

upper = 10 * ones(Float64, 6) # Set upper limit
ϕ0 = 0.5 * ones(Float64, 6) # Initial guess
optimizer = LBFGS(linesearch=BackTracking(order=2)) # Optimizer
ϕhats_ale = Vector{Vector{Float64}}(undef, length(y)) # Store result

@time Threads.@threads for i in eachindex(y)
    println(i)
    res = Optim.optimize(z -> cost(compute_ALE, x[i], y[i], z), lower, upper, ϕ0, Fminbox(optimizer), Optim.Options(time_limit=10.0); autodiff=:forward)
    ϕhats_ale[i] = Optim.minimizer(res)
end

# Check for NaN
for i in eachindex(ϕhats_ale)
    if any(isnan, ϕhats_ale[i])
        @show i
    end
end

# Rerun patient
# i = 836
# res = optimize(z -> cost(compute_ALE, x[i], y[i], z), lower, upper, x0, Fminbox(GradientDescent(linesearch=BackTracking())), Optim.Options(time_limit=30.0))
# ϕhats_ale[i] = Optim.minimizer(res)


