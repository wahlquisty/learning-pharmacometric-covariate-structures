# Date: 230111

# Five-fold cross validation of symbolic regression network on Eleveld data set. 

# Date: 230612
# Binary inputs gender and arterial/venous. -0.5, +0.5. 
# Modified power function again. |a|^b unless a = 0, then a^b = 0.


using Pkg
cd(@__DIR__)
cd("..")
Pkg.activate(".")

using Distributions, Flux, ForwardDiff, Random, Zygote
using BSON: @save                           # Format for saving Flux neural networks

include("../fcts/fcts_nn.jl")            # Function to generate and train symbolic regression network
include("../fcts/get_elevelddata.jl")          # Functions to generate training data from Eleveld data set
# include("../fcts/fcts_train.jl")               # Functions for network training
include("../fcts/fcts_flatstructure.jl")       # Functions to destructure (and rebuild) symbolic regression network to a flat structure for Hessian computations
include("../fcts/fcts_pruning.jl")             # Functions to prune inputs and parameters of symbolic regression network
include("../fcts/fcts_eqreader.jl")            # Functions to read symbolic regression network into readable functions.



"""
    symbolic_regression_nn(i, x, y, opt, seed, nepochs_init, nepochs_prune)

Training and pruning of a symbolic regression regression network on training data.

Creates a symbolic regression network and performs pruning. If the pruning was successful, the resulting neural network is written to file. During training, prints to terminal shows when initial training, inputpruning and parameterpruning are finished.

# Arguments:
- `i`: Int. Thread numbering, to simplify writing to terminal
- `x`: Vector{InputData}. Training x data with length(x) == nbr of patients.
- `y`: Vector{Vector{Number}}. Training y data with length(y) == nbr of patients.
- `opt`: Optimizer object, for example Flux.Adam(0.0001).
- `seed`: Int. Seed for reproducibility.
- `nepochs_init`: Int. Nbr of epochs to train before start of pruning and during inputpruning.
- `nepochs_prune`: Int. Nbr of epochs to train between parameter pruning steps.

Returns false if no connected network was produced. Otherwise, writes to file "bson/nn_threadi.bson" with name `model`.
"""
function symbolic_regression_nn(x, y, opt, lossfct, seed, nepochs_init, nepochs_prune, fold)

    # i = 1
    # lossfct = loss_ALE

    Random.seed!(seed)  # for reproducibility

    model = create_model(length(x[1].normalization), n_in, lossfct) # Create model (with parallel nn)

    ps = Flux.params(model.nn)     # parameters to train
    n_params = get_nparams(ps)  # nbr of traininable parameters

    print("Loss before training: ", round(model.loss(model.nn, x, y), digits=6), "\n")

    # Training
    model, loss_train = train!(model, x, y, nepochs_init, opt, ps, verbose=false)

    # Prune based on inputs
    N_in = 1 # nbr of inputs to prune in each network
    model = prunemodelinputs(model, N_in, x, y, verbose=false) # prune input
    model, loss_train = train!(model, x, y, nepochs_init, opt, ps, verbose=false) # train
    model = prunemodelinputs(model, N_in, x, y, verbose=false)
    model, loss_train = train!(model, x, y, nepochs_init, opt, ps, verbose=false)
    model = prunemodelinputs(model, N_in, x, y, verbose=false)
    model, loss_train = train!(model, x, y, nepochs_init, opt, ps, verbose=false)

    print("Loss after input pruning: ", round(model.loss(model.nn, x, y), digits=6), "\n") # loss after input pruning and training

    if isnan(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
        print("Loss is NaN \n")
        return false
    end

    # Network parameter pruning
    N_prune = 10 # nbr of parameters to prune in each network
    model = prunemodel(model, N_prune, n_params, x, y, verbose=false) # prune network parameters
    model, loss_train = train!(model, x, y, nepochs_prune, opt, ps, verbose=false) # train

    if isnan(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
        print("Loss is NaN or Inf \n")
        return false
    end

    # Now, prune fewer parameters at a time
    N_prune = 2 # nbr of parameters to prune in each network
    for j = 1:5#11
        model = prunemodel(model, N_prune, n_params, x, y, verbose=false) # prune network parameters

        if isnan(model.loss(model.nn, x, y)) || isinf(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
            print("Loss is NaN or Inf \n")
            return false
        end

        model, loss_train = train!(model, x, y, nepochs_prune, opt, ps, verbose=false) # train
    end

    N_prune = 1 # nbr of parameters to prune in each network
    for j = 1:2 # results in 12 parameters of final expression
        model = prunemodel(model, N_prune, n_params, x, y, verbose=false) # prune network parameters

        if isnan(model.loss(model.nn, x, y)) || isinf(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
            print("Loss is NaN or Inf \n")
            return false
        end

        model, loss_train = train!(model, x, y, nepochs_prune, opt, ps, verbose=false) # train
    end

    # if isnan(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
    #     print("Loss is NaN in thread ", i, "\n")
    #     return false
    # end

    print("Final loss after training and pruning: ", round(model.loss(model.nn, x, y), digits=6), "\n") # loss after pruning and training

    print("All networks are connected: ", all(isnnconnected(model.nn)), "\n") # Check if all networks are still connected between input and output

    @save "bson/nn_"*"fold"*"$fold.bson" model # Save nn in a bson-file

    # print(isnnconnected(model.nn))
    return true
end
##

"""
    main(x, y, nthreads, opt, error_fct, seed, nepochs_init, nepochs_prune)

Main function for training and pruning of several symbolic regression regression networks on training data.

Creates nthreads symbolic regression networks and performs pruning of each. The computing is done in parallel (only if julia is started with several threads) using Threads.@threads. If the pruning was successful, the resulting neural networks are written to file (type bson) in folder "bson/".

# Arguments:
- `x`: Vector{InputData}. Training x data with length(x) == nbr of patients.
- `y`: Vector{Vector{Number}}. Training y data with length(y) == nbr of patients.
- `nthreads`: Int, number of threads to run.
- `opt`: Optimizer object, for example Flux.Adam(0.0001).
- `error_fct`: Function, which loss function to use.
- `seed`: Int. Seed for reproducibility.
- `nepochs_init`: Int. Nbr of epochs to train before start of pruning and during inputpruning.
- `nepochs_prune`: Int. Nbr of epochs to train between parameter pruning steps.

Writes to file for each thread "bson/nn_threadi.bson" with name `model`.
"""
function main(x, y, opt, error_fct, seed, nepochs_init, nepochs_prune, fold)
    if error_fct == "MSE" # change error function depending on input
        print("Error function: ", error_fct, "\n")
        lossfct = loss_MSE
    elseif error_fct == "ALE"
        print("Error function: ", error_fct, "\n")
        lossfct = loss_ALE
    else
        print("Error function ", error_fct, " not implemented.")
        return false
    end

    # @time Threads.@threads for i = 1:nthreads
    #     seed_i = seed + i - 1
        symbolic_regression_nn(x, y, opt, lossfct, seed, nepochs_init, nepochs_prune,fold)
    # end
    return true
end



## Five fold cross-validation
seed = 1234
Random.seed!(seed)

x_all, y_all = get_elevelddata()
np = 1031
n_in = size(x_all[1].covariates, 1) # nbr of inputs to each network (number of covariates)
percentfold = 0.8
opt = Flux.Adam(0.0001) # Optimizer

shuffled_indices = randperm(np)
npat_eachtest = Int(ceil(np * (1 - percentfold)))

nepochs_init = 1000
nepochs_prune = nepochs_init

@time Threads.@threads for fold = 1:5
# @time for fold = 1
    # Training data
    # Patient data

    test_ind_low = (fold-1)*npat_eachtest+1
    test_ind_high = test_ind_low+npat_eachtest-1

    if test_ind_high > np
        test_ind_high = np
    end

    test_indices = shuffled_indices[test_ind_low:test_ind_high]
    training_indices = filter(x -> !(x in test_indices), shuffled_indices)

    xtrain = x_all[training_indices]
    ytrain = y_all[training_indices]
    xtest = x_all[test_indices]
    ytest = y_all[test_indices]

    main(xtrain, ytrain, opt, "ALE", seed, nepochs_init, nepochs_prune, fold)
    # main(xtrain, ytrain, 1, opt, "ALE", seed, 0, 0, fold)

    # main(xtrain, ytrain, 1, opt, "ALE", seed, 10, 10, fold)

    # @show test_ind_low, test_ind_high
end

##




