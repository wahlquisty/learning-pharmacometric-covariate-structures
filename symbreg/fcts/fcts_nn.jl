####################################### # Custom split layer. See flux documentation for information.
# Date: 221220

struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = tuple(map(f -> f(x), m.paths))


################################### # Implements a Dense layer with mask so that the true parameter value can be computed as weight*W_mask
# Does not mutate arrays
# Date: 221213
using Flux

struct DenseWithMask{F,M<:AbstractMatrix,B}
    weight::M
    bias::B
    σ::F
    W_mask::M
    b_mask::B

    # Initialize struct, takes the same arguments as a normal dense layer.
    function DenseWithMask(args...; kwargs...)
        # Create a normal layer just to get identical syntax and initialization as a
        # normal Dense layer. A bit overkill but works.
        l = Dense(args...; kwargs...)

        # Create mask for the params
        W_mask = ones(size(l.weight))
        b_mask = ones(length(l.bias))

        M, B, F = typeof.((l.weight, l.bias, l.σ))
        return new{F,M,B}(l.weight, l.bias, l.σ, W_mask, b_mask)
    end
end

# I am not completely sure what these lines do, but the second specify which
# fields in the struct should be consider parameters when calling Flux.params()
Flux.@functor DenseWithMask
Flux.trainable(l::DenseWithMask) = (l.weight, l.bias)

# This is the function that is called when the layer is evaluated
function (l::DenseWithMask)(x::AbstractVecOrMat)
    W = l.weight .* l.W_mask
    return l.σ.(W * x .+ (l.bias .* l.b_mask))
end



# Date: 230105

using Flux

# Struct for input data
struct InputData
    covariates # vector or real
    u::Vector
    v::Vector
    time::Vector # needed?
    hs::Vector
    youts::Vector
    normalization::Vector # to convert volumes and clearances
end

# include("masked_dense.jl")   # Masked dense layer, to mask parameters in network
# include("split.jl")          # Split layer (for parallel network structure. Takes one input and outputs several outputs)

const T_parallelnn = Chain{Tuple{Split{Vector{Any}}}}  # type of full model (several parallel nodes in a nested Split structure)
const T_subnn = Chain                                # type of subnetwork

# Struct for SymbRegNN
struct SymbRegNN
    nn::T_parallelnn
    loss::Function # loss function
end

"""
    activation_1(x)

Activation function for first layer. Assumes input is of size 7.

# Arguments:
- `x`: Input vector (or matrix).
"""
function activation_1(x)
    return [x[1, :] (x[2, :] .* x[3, :]) powerfunction.(x[4, :], x[5, :])]'
end


"""
    activation_2(x)

Activation function for second layer. Assumes input is of size 5.

# Arguments:
- `x`: Input vector (or matrix).
"""
function activation_2(x)
    return [x[1, :] (x[2, :] .* x[3, :]) (x[4, :] ./ (x[5, :] .+ one(eltype(x))))]'
end

# Special power function called by activation functions.
function powerfunction(x1, x2)
    z = zero(eltype(x1))
    if x1 ≈ z
        return z # Return zero if base is zero.
    else
        return abs.(x1) .^ x2 # |a|^b, to avoid for example (-0.5)^1.2
    end
end

# Create one network with three dense layers, activation functions. 
function createonenn(n_in)
    nn = Chain(
        DenseWithMask(n_in, 5),     # Nbr of inputs to network is n_in
        x -> activation_1(x),
        DenseWithMask(3, 5),
        x -> activation_2(x),
        DenseWithMask(3, 1, abs))   # To assure positive outputs

    nn = init_powerfunction(nn)          # Restrict exponent in power function of first layer to only take numerical values.
    return nn
end

# Initiate power function. Exponent in a^b only takes numerical values, means only bias term. Remove weight so that we cannot get x^x and only x^a.
function init_powerfunction(nn)
    nn[1].weight[end, :] .= 0.0
    nn[1].W_mask[end, :] .= 0.0
    return nn
end

"""
    create_model(n_networks, n_in)

Create Flux model with parallel networks that has the same amount of inputs and one output each, using the Split layer.

# Arguments:
- `n_networks`: Int, the number of parallel networks.
- `n_in`: Int, the number of inputs to each network.

Returns a Flux model in a nested structure.
"""
function create_model(n_networks, n_in, lossfct) # for now, 6 parallel n_networks
    chain = Vector{Any}(undef, n_networks)

    for i = 1:n_networks
        chain[i] = createonenn(n_in)
    end
    nn = Chain(Split(chain))

    model = SymbRegNN(nn, lossfct)
    return model
end


"""
    get_nparams(ps)

Computes the total number of (trainable) parameters in a Flux model with parallel networks from a parameter vector.

# Arguments:
- `ps`: Params object, output from Flux.params(model) over a nested Flux model.

Returns the total number of (trainable) parameters in a Flux model.
"""
function get_nparams(ps)
    nparams = 0
    for i in ps
        nparams += length(i)
    end
    return nparams
end

using Flux




###################################### Functions for PK simulation (three compartment model)


include("fastpksim.jl")
# include("simulation.jl")


"""
    get_modeloutput(model::T_parallelnn, x)

Returns the Flux model output given an input x.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.
- `x`: Matrix, Vector or Real, input to model.
"""
function get_modeloutput(parallelnn::T_parallelnn, x)
    return vec(reduce(vcat, parallelnn(x)[1]))
end

"""
    get_modeloutput(model::T_subnn, x)

Returns the output of a network given an input x.

# Arguments:
- `model`: Neural network with length(x) inputs.
- `x`: Matrix, Vector or Real, input to model.
"""
function get_modeloutput(nn::T_subnn, x)
    return nn(x)[1]
end

"""
     train!(model, x, y, n_epochs, opt, ps)

Trains the Flux model for n_epochs given parameters ps and optimizer opt on training data x and y. Computes the gradient of the loss function loss(model,x,y) and updates the model based on the choice of optimizer.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.
- `x`: Vector{InputData} struct objects. x training data.
- `y`: Vector{Vector}. y training data.
- `n_epochs`: Number of training epochs.
- `opt`: Optimizer object. For example Flux.Adam(0.005).
- `ps`: Params object. Model parameters from Flux.params(model).

Returns the updated model and training losses at each epoch.
"""
function train!(model::SymbRegNN, x, y, n_epochs, opt, ps; verbose=true)
    if verbose
        print("Loss before training: ", round(model.loss(model.nn, x, y), digits=6), "\n") # loss before training
    end
    losses = zeros(n_epochs)
    for epoch = 1:n_epochs
        if verbose
            if iszero(epoch % 500)
                print("Training epoch: ", epoch, "\n")
            end
        end
        loss_epoch = 0.0
        for i in eachindex(y)
            loss_i, grads = Flux.withgradient(ps) do
                model.loss(model.nn, x[i], y[i])
            end
            Flux.update!(opt, ps, grads)
            loss_epoch += loss_i
        end
        losses[epoch] = loss_epoch
    end
    if verbose
        print("Loss after training: ", round(model.loss(model.nn, x, y), digits=6), "\n") # loss after training
    end
    return model, losses
end

# ALE loss function
# Date: 230210

"""
     loss_ALE(model, x::InputData, y)

Computes the ALE (absolute logarithmic error) over a Flux model given x and y data (one patient).
Output from network model results in a vector of covariates, which inputs to a simulation together with simulation data (input, time) from x object. ALE is computed between predicted output concentrations (at time (x.time)[x.youts]) and measured concentrations y.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.
- `x`: InputData struct object. x training data.
- `y`: Vector. y training data, measurements at instances x.youts.

Returns the ALE loss over all x and y at instances (x.time)[x.youts].
"""
function loss_ALE(nn::T_subnn, x::InputData, y) # compute mse loss
    u = x.u # Infusion rates
    v = x.v # Bolus doses
    hs = x.hs # Time differences between rate changes. Δ(t+1)-Δ(t) = hs

    ϕhat_scaled = get_modeloutput(nn, x.covariates) # Vector with predicted PK parameters (k10, k12 etc) scaled using x.normalization
    ϕhat = ϕhat_scaled .* x.normalization # Scale predicted PK parameters from (0,1) to range (0,maxval in dataset)

    V1inv, λ, λinv, R = PK3(ϕhat) # Create necessary matrices for simulation of 3rd order compartment model

    totalloss = 0.0 # squared error
    j = 1 # counter to keep track of next free spot in y
    x_state = zeros(eltype(u), 3) # initial state
    for i in eachindex(hs) # Iterate through all time samples
        if i in x.youts # if we want to compute output
            x_state, yi = @inbounds updatestateoutput(x_state, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            if yi > 0 # If predicted concentration is zero, do not add to total loss
                totalloss += compute_ALE(y[j], yi) # Compute ALE
            #elseif isnan(yi)
            #    return NaN
            end
            j += 1
        else
            x_state = @inbounds updatestate(x_state, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return totalloss/length(y)
end

compute_ALE(y, yhat) = abs(log(y / abs(yhat)))

"""
     loss(model:T_parallelnn, x::Vector{InputData}, y)

Computes the error over a Flux model given x and y data (many patients) (by calling the function loss(model,x,y))
Output from network model results in a vector of covariates, which inputs to a simulation together with simulation data (input, time) from xdata object. Mse is computed between predicted output concentrations (at time (x.time)[x.youts]) and measured concentrations y.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.
- `x`: Vector{InputData} struct objects. x training data.
- `y`: Vector{Vector}. y training data, measurements at instances xdata.youts.

Returns the total loss over all training data.
"""
function loss_ALE(parallelnn::T_parallelnn, x::Vector{InputData}, y)
    totalloss = 0.0
    for i in eachindex(x)
        totalloss += loss_ALE(parallelnn, x[i], y[i]) # Compute loss for each patient
    end
    return totalloss
end


# MSE loss function
# Date: 230210

"""
     loss_MSE(model, x::InputData, y)

Computes the mse (mean squared error) over a Flux model given x and y data (one patient).
Output from network model results in a vector of covariates, which inputs to a simulation together with simulation data (input, time) from x object. Mse is computed between predicted output concentrations (at time (x.time)[x.youts]) and measured concentrations y.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.
- `x`: InputData struct object. x training data.
- `y`: Vector. y training data, measurements at instances x.youts.

Returns the mse loss over all x and y at instances (x.time)[x.youts].
"""
function loss_MSE(nn::T_subnn, x::InputData, y) # compute mse loss
    u = x.u # Infusion rates
    v = x.v # Bolus doses
    hs = x.hs # Time differences between rate changes. Δ(t+1)-Δ(t) = hs

    ϕhat_scaled = get_modeloutput(nn, x.covariates) # Vector with predicted PK parameters (k10, k12 etc) scaled using x.normalization
    ϕhat = ϕhat_scaled .* x.normalization # Scale predicted PK parameters from (0,1) to range (0,maxval in dataset)

    V1inv, λ, λinv, R = PK3(ϕhat) # Create necessary matrices for simulation of 3rd order compartment model

    totalloss = 0.0 # squared error
    j = 1 # counter to keep track of next free spot in y
    x_state = zeros(eltype(u), 3) # initial state
    for i in eachindex(hs) # Iterate through all time samples
        if i in x.youts # if we want to compute output
            x_state, yi = @inbounds updatestateoutput(x_state, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            totalloss += compute_squarederror(y[j], yi) # only compute loss when we have observations
            j += 1
        else
            x_state = @inbounds updatestate(x_state, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return totalloss / length(x.youts) # Mean squared error

end

compute_squarederror(y, yhat) = abs2(y - yhat)

"""
     loss(model:T_parallelnn, x::Vector{InputData}, y)

Computes the error over a Flux model given x and y data (many patients) (by calling the function loss(model,x,y))
Output from network model results in a vector of covariates, which inputs to a simulation together with simulation data (input, time) from xdata object. Mse is computed between predicted output concentrations (at time (x.time)[x.youts]) and measured concentrations y.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.
- `x`: Vector{InputData} struct objects. x training data.
- `y`: Vector{Vector}. y training data, measurements at instances xdata.youts.

Returns the total loss over all training data.
"""
function loss_MSE(parallelnn::T_parallelnn, x::Vector{InputData}, y)
    totalloss = 0.0
    for i in eachindex(x)
        totalloss += loss_MSE(parallelnn, x[i], y[i]) # Compute loss for each patient
    end
    return totalloss
end


