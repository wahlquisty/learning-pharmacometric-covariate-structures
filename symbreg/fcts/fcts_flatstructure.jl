# Functions to destructure (and rebuild) a symbolic regression network (put together with the split.jl function)
# Date: 230210

using Flux

struct FlatStructure
    params
    params_masks
    params_indices
    params_startstop
    params_sizes
    lossfct::Function
end

"""
    destructure(model::T_subnn, n_params)

Function that destructures a network and returns data to be able to rebuild it. See `rebuild`.

# Arguments:
- `nn`: Symbolic regression network.

# Returns:
- `params`:            Vector, model parameters in a vector (weight and biases)
- `params_masks`:      Vector, masks for each parameter in params
- `params_indices`:    Matrix, original indices in original weight/bias matrix for each parameter in params. For reconstruction.
- `params_startstop`:  Matrix, start and stop index for each weight/bias matrix. For reconstruction.
- `params_sizes`:    Matrix, sizes of all weight and bias matrices. For reconstruction.
"""
function destructure(nn::T_subnn, n_params, lossfct)
    # Preallocate outputs
    nlayers = 3 # length(m) hardcoded for now
    params = zeros(Float32, n_params) # parameter values for each layer, with weight and biases (in that order)
    params_startstop = zeros(Int64, nlayers * 2, 2) # vector with start and stop indices for each weight and bias in params
    params_sizes = zeros(Int64, nlayers * 2, 2) # sizes of all weight matrices and bias vectors
    params_masks = zeros(Float32, n_params) # masks for each parameter, corresponds to params
    params_indices = zeros(Int64, n_params, 4) # Order: [layer, x index, y index, weight = 1/bias = 2]

    startpos = 1 # Positions to fill params. These values will also be put in params_indices
    endpos = 1

    for layer in [1, 3, 5] # = 1:nlayers # For each dense layer
        w, wmask, b, bmask = get_weightbias(nn, layer) # get weights, biases and their masks

        # Reshape weight matrix (and mask)
        wr = reshape(w, length(w),) # Reshaped weight matrix
        wmaskr = reshape(wmask, length(wmask),) # Reshaped wmask

        endpos = startpos + length(wr) - 1 # Compute end position of this weight matrix
        indw = layer # Indices to fill params_sizes and params_startstop
        indb = layer + 1

        # Fill outputs for weight
        params[startpos:endpos] = wr # add weight parameters to vector 
        params_startstop[indw, :] = [startpos, endpos] # start and stop value in params for this weight
        params_sizes[indw, :] = [size(w, 1), size(w, 2)] # weight matrix size
        params_masks[startpos:endpos] = wmaskr # weight mask
        params_indices[startpos:endpos, :] = [layer * ones(length(wr)) get_indicesmatrix(w) ones(length(wr),)] # For all weights: [layer, x index, y index, weight = 1/bias = 2]

        # Update start and end position
        startpos = endpos + 1
        endpos = startpos + length(b) - 1

        # Fill outputs for bias
        params[startpos:endpos] = b # add bias parameters to vector 
        params_startstop[indb, :] = [startpos, endpos]
        params_sizes[indb, :] = [length(b), 1]
        params_masks[startpos:endpos] = bmask
        params_indices[startpos:endpos, :] = [layer * ones(length(b)) (1:length(b))[:] ones(length(b)) 2 * ones(length(b),)] # For all biases: [layer, x index, y index, weight = 1/bias = 2]

        # Update start position
        startpos = endpos + 1
    end

    return FlatStructure(params, params_masks, params_indices, params_startstop, params_sizes, lossfct)
end

"""
    destructure(model::SymbRegNN, n_params)

Function that destructures a Flux model with parallel networks and returns data to be able to rebuild it. See `rebuild`.

# Arguments:
- `model`:             SymbRegNN, Flux model of parallel neural networks using Split.
- `n_params`:          Int, total number of parameters in SymbRegNN

# Returns:
- `flats`:             FlatStructure, a flat representation of the SymbRegNN, which includes:

- `params`:            Vector{Vector}, model parameters in a vector (weight and biases)
- `params_masks`:      Vector{Vector}, masks for each parameter in params
- `params_indices`:    Vector{Matrix}, original indices in original weight/bias matrix for each parameter in params. For reconstruction.
- `params_startstop`:  Vector{Matrix}, start and stop index for each weight/bias matrix. For reconstruction.
- `params_sizes`:      Vector{Matrix}, sizes of all weight and bias matrices. For reconstruction.
- `loss`:              Function, loss function
"""
function destructure(model::SymbRegNN, n_params) # preferably do not hradcode the types, use oftype, eltype instead
    chain = model.nn[1].paths
    n_networks = length(chain)
    n_params_sub = Int64.(n_params / n_networks)

    flats = Vector{FlatStructure}(undef, n_networks)

    for i = 1:n_networks
        submodel = chain[i]
        flats[i] = destructure(submodel, n_params_sub, model.loss)
    end
    return flats
end

# Get weights, biases and their masks from model at a specific layer.
function get_weightbias(nn, layer)
    weight = nn[layer].weight
    wmask = nn[layer].W_mask
    bias = nn[layer].bias
    bmask = nn[layer].b_mask
    return weight, wmask, bias, bmask
end

# Get all indices of a matrix m, for example [1 2; 3 4] returns indices in a matrix [1 1; 2 1; 1 2; 2 2].
function get_indicesmatrix(nn)
    indices = zeros(Int64, length(nn), 2)
    k = 1
    for i in axes(nn, 2)
        for j in axes(nn, 1)
            indices[k, 1] = j
            indices[k, 2] = i
            k += 1
        end
    end
    return indices
end

"""
    rebuild(params::Vector, params_startstop, params_sizes)

Function that rebuilds a network based on information from `destructure` and the parameters.

# Arguments:
- `params`:             Parameters for network.
- `params_startstop`:   start and stop index for each weight/bias matrix.
- `params_sizes`:     sizes of all weight and bias matrices

Returns the rebuilt network `m` as a Flux Chain.
"""
function rebuild(params::Vector, flat::FlatStructure) # rebuild Chain
    nn = Chain(
        rebuild_createlayer(1, params, flat),
        x -> activation_1(x),
        rebuild_createlayer(3, params, flat),
        x -> activation_2(x),
        rebuild_createlayer(5, params, flat))
    return nn
end

# Rebuild one layer (masked dense)
function rebuild_createlayer(l, params, flat::FlatStructure)
    params_sizes = flat.params_sizes
    params_startstop = flat.params_startstop

    # Indices to reconstruct from params_sizes and params_startstop
    indw = l
    indb = l + 1

    # reconstruct weight and bias matrices
    params_w = params[params_startstop[indw, 1]:params_startstop[indw, 2]] # get all weight parameters
    w = reshape(params_w, params_sizes[indw, 1], params_sizes[indw, 2]) # reshape to matrix with sizes in params_sizes
    params_b = params[params_startstop[indb, 1]:params_startstop[indb, 2]]
    b = reshape(params_b, params_sizes[indb, 1])

    # Recreate layer 
    if l == 5 # last layer
        layer = DenseWithMask(w, b, abs) # positive outputs
    else
        layer = DenseWithMask(w, b)
    end
    return layer
end

"""
    rebuild(params::Vector, flats::Vector{FlatStructure})

Function that rebuilds a Flux model with parallel networks using Chain and Split based on information from `destructure` and the parameters.

# Arguments:
- `params`:             Vector{Vector{Float32}}. Parameters for network.
- `params_startstop`:   Vector{Matrix}, start and stop index for each weight/bias matrix.
- `params_sizes`:     Vector{Matrix}, sizes of all weight and bias matrices

Returns the rebuilt network `model` as a Flux Chain(Split).
"""
function rebuild(params::Vector, flats::Vector{FlatStructure}) # has some typeinfer on chain (type is any so I don't know how to fix)
    n_networks = length(flats)
    chain = Vector(undef, n_networks)
    for i = 1:n_networks
        chain[i] = rebuild(params, flat[i])
    end
    nn = Chain(Split(chain)) # create parallel network structure
    return SymbRegNN(nn, lossfct)
end