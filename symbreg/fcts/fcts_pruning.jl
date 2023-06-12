# Functions for inputpruning and parameter pruning of parallel neural networks (symbolic regression networks)
# Date: 230105

using Flux, Zygote, ForwardDiff

"""
    prunemodelinputs(model, N_in, x, y; verbose=true)

Prunes N_in inputs of each subnetwork in the symbolic regression network.

Computes the salience based on the diagonal Hessian (with respect to the training data) and removes the N_in inputs with the lowest salience. Salience = h_{kk}*u^2 where h_{kk} is the diagonal Hessian.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.
- `N_in`: Int. Number of inputs to prune of each subnetwork.
- `x`: Vector{InputData} struct objects. x training data.
- `y`: Vector{Vector}. y training data.
- `verbose` Bool. Optional argument, defaults to true. If true, prints loss and number of parameters left in each subnetwork after pruning.

Returns the updated (pruned) model.
"""
function prunemodelinputs(model::SymbRegNN, N_in, x, y; verbose=true)
    diaghess = sumdiaghessian_input(model.nn, x, y) # Compute sum of diagonal hessian of the inputs over all patients
    salience = computesalience_input(diaghess, x) # Compute the salience
    input_masks = getinputmaskforpruning(model.nn, n_in) # Get the masks of the inputs to not include the already masked inputs
    model = removeNinputs(model, N_in, salience, input_masks) # Remove N_in inputs with respect to the input masks and salience
    if verbose
        print("Loss after pruning ", N_in, " input(s) per network: ", round(model.loss(model.nn, x, y), digits=6), "\n")
        print("Number of parameters left in each subnetwork: ", getnbrofparamsleft(model.nn), "\n")
    end
    return model
end


"""
    prunemodel(model::SymbRegNN, N_prune, n_params, x, y; verbose=true)

Prunes N_prune parameters in each subnetwork of the symbolic regression network.

Computes the salience based on the diagonal Hessian (with respect to the training data) and removes the N_params parameters with the lowest salience. Salience = h_{kk}*u^2 where h_{kk} is the diagonal Hessian and u is the parameter value.

# Arguments:
- `model`:               SymbRegNN, Flux model of parallel neural networks using Split.
- `N_prune`:             Int. Number of parameters to prune of each subnetwork.
- `n_params`:            Int. Number of total trainable parameters. Output from get_nparams(ps).
- `x`:                   Vector{InputData} struct objects. x training data.
- `y`:                   Vector{Vector}. y training data.
- `verbose`              Bool. Optional argument, defaults to true. If true, prints loss and number of parameters left in each subnetwork after pruning.

Returns the updated (pruned) model (SymbRegNN).
"""
function prunemodel(model::SymbRegNN, N_prune, n_params, x, y; verbose=true)
    flats = destructure(model, n_params) # Destructure model (get parameter values, masks, sizes etc) to be able to compute hessian over each parameter value.

    # Compute hessian and salience
    diaghess = sumdiaghessian_params(flats, x, y) # compute sum of diaghessian of all patients over all model parameters
    salience, unmaskedind = computesalience_params(flats, diaghess) # compute salience
    model = removeNparams(model, N_prune, salience, unmaskedind, flats) # Remove N_prune parameter from each subnetwork with respect to the mask and update each network model.
    if verbose
        print("Loss after pruning ", N_prune, " parameters per network: ", round(model.loss(model.nn, x, y), digits=6), "\n")
        # Parameters left
        print("Number of parameters left in each subnetwork: ", getnbrofparamsleft(model.nn), "\n")
    end
    return model
end



"""
    getnbrofparamsleft(model)

Get the number of parameters left in each subnetwork (after pruning).

# Arguments:
- `model`: Flux model of parallel neural networks using Split.

Returns a vector of number of parameters left in each subnetwork.
"""
function getnbrofparamsleft(model::T_parallelnn)
    parallelnn = model[1].paths
    np = zeros(Int64, length(parallelnn))
    for i in eachindex(parallelnn)
        chain = parallelnn[i]
        for l in [1, 3, 5]
            np[i] += Int64(sum(chain[l].W_mask) + sum(chain[l].b_mask))
        end
    end
    return np
end


"""
    isnnconnected(model::T_parallelnn)

Check if the network is connected between input and output.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.

Returns a vector of Bool, where true means subnetwork is connected, false means disconnected.
"""
function isnnconnected(parallelnn::T_parallelnn) # check if nn is connected between in- and output
    chain = parallelnn[1].paths
    n_networks = length(chain)
    connectivity = Vector{Bool}(undef, n_networks)
    for i = 1:n_networks
        if !isnnconnected(chain[i])
            connectivity[i] = false
        else
            connectivity[i] = true
        end
    end
    return connectivity
end

function isnnconnected(nn::T_subnn) # check if nn is connected between in- and output
    n_in = size(nn[1].weight, 2)
    if nn(ones(n_in))[1] == nn(1.1 * ones(n_in))[1] # then no inputs affects the output
        return false
    else
        return true
    end
end



## Helping functions inputpruning below

# Sum up diagonal hessian for all patients to remove inputs for full network structure
function sumdiaghessian_input(parallelnn::T_parallelnn, x, y)
    chain = parallelnn[1].paths
    n_networks = length(chain)
    diaghess = Vector{Vector{Float32}}(undef, n_networks)
    for i = 1:n_networks
        diaghess[i] = sumdiaghessian_input(chain[i], x, y)
    end
    return diaghess
end

# Sum up diagonal hessian for all patients for pruning of inputs for one subnetwork
function sumdiaghessian_input(nn::T_subnn, x, y)
    n = length(y) # nbr of patients
    n_inputs = length(x[1].covariates)
    diaghess = zeros(eltype(x[1].covariates), n_inputs) # sum of diagonal hessian for all patients, for each parameter
    for i = 1:n
        diaghess += abs.(Zygote.diaghessian(v -> loss_input(nn, v, y[i], x[i]), x[i].covariates)[1])
    end
    return diaghess
end

# Compute loss with respect to the input parameters
function loss_input(nn::T_subnn, x_covariates::Vector, y, xdata::InputData)
    xtrain = x_covariates
    youts = xdata.youts
    u = xdata.u
    v = xdata.v
    hs = xdata.hs
    normalization = xdata.normalization

    θhat = get_modeloutput(nn, xtrain) # vector or number
    θ = θhat .* normalization

    totalloss = 0.0
    V1inv, λ, λinv, R = PK3(θ)
    j = 1 # counter to keep track of next free spot in y
    x = zeros(eltype(u), 3) # initial state
    for i in eachindex(hs)
        if i in youts # if we want to compute output
            x, yi = @inbounds updatestateoutput(x, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            totalloss += compute_squarederror(y[j], yi) # only compute loss when we have observations
            j += 1
        else
            x = @inbounds updatestate(x, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return totalloss / length(youts)
end


# Compute "salience" for inputs over full network structure
function computesalience_input(diaghess::Vector{Vector{Float32}}, x)
    n_networks = length(diaghess)
    salience = Vector{Vector{Float32}}(undef, n_networks)
    for i = 1:n_networks
        salience[i] = computesalience_input(diaghess[i], x) # compute salience for each subnetwork
    end
    return salience
end

# Compute "salience" for inputs over a subnetwork
function computesalience_input(diaghess::Vector, x)
    salience = zeros(eltype(diaghess), size(diaghess))
    for i in eachindex(x)
        salience .+= abs.(diaghess .* (x[i].covariates) .^ 2) # salience = |h_kk * u^2|
    end
    return salience
end

# Get input masks for inputpruning. Check which inputs are already masked so that they are not removed a second time. For full network structure
function getinputmaskforpruning(parallelnn::T_parallelnn, n_in)
    chain = parallelnn[1].paths
    input_masks = Vector{Vector{Int64}}(undef, length(chain))
    for i in eachindex(chain)
        mask = ones(Int64, n_in)
        wmask_layer1 = chain[i][1].W_mask
        wsum = sum(wmask_layer1, dims=1)
        for j in eachindex(wsum)
            if wsum[j] == 0.0
                mask[j] = 0
            end
        end
        input_masks[i] = mask
    end
    return input_masks
end

# remove N inputs from full network based on salience
function removeNinputs(model::SymbRegNN, N, salience, input_masks)
    nn = model.nn
    n_networks = length(salience)
    chain = nn[1].paths
    for i = 1:n_networks
        removeNinputs(chain[i], N, salience[i], input_masks[i]) # Remove N inputs for each subnetwork
    end
    return model # not sure this works
end

# Remove N inputs for a subnetwork based on the salience
function removeNinputs(nn::T_subnn, N, salience, input_mask)
    unmaskedinputs = findall(!iszero, input_mask)
    indsort = sortperm(salience[unmaskedinputs]) # later, replace with sortperm!
    indleft = unmaskedinputs[indsort]
    indtoremove = indleft[1:N]
    nn = pruneinputandupdatemodel(nn, indtoremove)
    return nn
end

# Prune inputs (mask and set elements to zero) in a model based on indices with small salience.
function pruneinputandupdatemodel(nn::T_subnn, indtoremove)
    for i in indtoremove
        nn[1].weight[:, i] .= 0.0
        nn[1].bias[i] = 0.0
        nn[1].W_mask[:, i] .= 0.0
        nn[1].b_mask[i] = 0.0
    end
    return nn
end


## Helping functions parameterpruning below

# Compute sumdiaghessian over all patients for all subnetworks
function sumdiaghessian_params(flats::Vector{FlatStructure}, x, y)
    n_networks = length(flats)
    diaghess = Vector{Vector{Float32}}(undef, n_networks)
    for i = 1:n_networks
        diaghess[i] = sumdiaghessian_params(flats[i], x, y)
    end
    return diaghess
end

# Sum up diagonal hessian for all patients over a subnetwork
function sumdiaghessian_params(flat::FlatStructure, x, y)
    nparams = length(flat.params) # nbr of model parameters
    n = size(y, 2) # nbr of patients
    diaghess = zeros(eltype(x[1].covariates), nparams) # sum of diagonal hessian for all patients, for each parameter
    for i = 1:n
        diaghess += abs.(Zygote.diaghessian(v -> loss_params(v, flat, x[i], y[i]), flat.params)[1]) # second derivative
    end
    return diaghess
end


# Compute loss with respect to the parameters (for computation of diaghessian). Uses rebuild function.
function loss_params(params::Vector, flat, x, y)
    subnn = rebuild(params, flat)
    return flat.lossfct(subnn, x, y)
end

# Compute salience based on the diaghessian for each subnetwork. Only compute salience for those without a mask
function computesalience_params(flats::Vector{FlatStructure}, diaghess)
    n_networks = length(flats)
    salience = Vector{eltype(params)}(undef, n_networks)
    unmaskedind = Vector{Vector{Int64}}(undef, n_networks)
    for i = 1:n_networks
        salience[i], unmaskedind[i] = computesalience_params(flats[i], diaghess[i])
    end
    return salience, unmaskedind
end

# Compute salience based on the diaghessian for a subnetwork. Only compute salience for those without a mask
function computesalience_params(flat::FlatStructure, diaghess)
    unmaskedind = findall(x -> x == one(eltype(flat.params)), flat.params_masks) # indices that are left and unmasked
    salience = abs.(diaghess[unmaskedind] .* flat.params[unmaskedind] .^ 2)  # diaghess*u^2 (abs parameter value) (salience from optimal brain damage paper)
    return salience, unmaskedind
end

# Remove N parameters from each subnetwork in the full model structure based on salience.
function removeNparams(model::SymbRegNN, N, salience, unmaskedind, flats::Vector{FlatStructure})
    chain = model.nn[1].paths
    lossfct = model.loss
    n_networks = length(chain)
    for i = 1:n_networks
        submodel = removeNparams(chain[i], N, salience[i], unmaskedind[i], flats[i].params_indices)
        chain[i] = submodel # also changes model
    end
    nn = Chain(Split(chain)) # create parallel network structure
    return SymbRegNN(nn, lossfct)
end

# Remove N parameters from a subnetwork in the full model structure based on salience.
function removeNparams(nn::T_subnn, N, salience, unmaskedind, params_indices)
    indsort = sortperm(salience) # later, replace with sortperm!
    indtoremove = indsort[1:N]
    indsmallsalience = unmaskedind[indtoremove] # indices to remove. index corresponds to indec in modelparams
    nn = pruneparamsandupdatemodel(nn, params_indices, indsmallsalience)
    return nn
end

# Prune (mask and set elements to zero) a model based on indices with small salience.
function pruneparamsandupdatemodel(nn::T_subnn, params_indices, indsmallsalience)
    for i in indsmallsalience # set to zero in the original model
        layer = params_indices[i, 1]
        w1b2 = params_indices[i, 4]
        ind_to_remove = params_indices[i, 2:3]
        if w1b2 == 1 # weight
            nn[layer].weight[ind_to_remove[1], ind_to_remove[2]] = 0
            nn[layer].W_mask[ind_to_remove[1], ind_to_remove[2]] = 0
        else
            nn[layer].bias[ind_to_remove[1]] = 0
            nn[layer].b_mask[ind_to_remove[1]] = 0
        end
    end
    return nn
end
