# Symbolic regression network on the Eleveld data set (1031 patients) for propofol
# Training and pruning of symbolic regression with PK simulation of three-compartment model
# The network takes scaled input data (0->1)

# Training data:
# x: Five covariates as inputs (age (scaled 0->1), weight (scaled 0->1), gender (Male -0.5, Female 0.5), Sampling site (Arterial -0.5, Venous 0.5))
# y: Blood plasma concentration measurements together with time data, infusion data (constant infusion and bolus).
# Training data is put in a struct: InputData, that holds u (infusion rates),v (bolus doses),hs (time differences t(i+1).t(i)),covariates (scaled) and covariate scaling (to recover from scaling).

# To run main function with 8 threads, each creating a symbolic regression network with training and pruning, run
# main(x, y, 8, opt, error_fct, seed, nepochs_init, nepochs_prune)
# where error_fct is either "MSE" or "ALE".
# The function writes the resulting (connected) network(s) to file "bson/nn_thread$i.bson" with name `model`, which can be opened by calling using BSON:@load; @load "bson/nn_thread5.bson" model (replace with the thread number that you are interested in)


# Date: 230608


using Pkg
cd(@__DIR__)
Pkg.activate(".")

using Distributions, Flux, ForwardDiff, Random, Zygote
using BSON: @save                           # Format for saving Flux neural networks

include("fcts/fcts_nn.jl")                  # Functions to generate and train symbolic regression network
include("fcts/get_elevelddata.jl")          # Functions to generate training data from Eleveld data set
include("fcts/fcts_flatstructure.jl")       # Functions to destructure (and rebuild) symbolic regression network to a flat structure for Hessian computations
include("fcts/fcts_pruning.jl")             # Functions to prune inputs and parameters of symbolic regression network
include("fcts/fcts_eqreader.jl")            # Functions to read symbolic regression network into readable functions.


## Setup (training data)
seed = 1234
Random.seed!(seed) # for reproducibility
x_all, y_all = get_elevelddata() # patient data of all patients in the data set.
n_in = size(x_all[1].covariates, 1) # nbr of inputs to each network (= number of covariates)

np = length(x_all)  # 1031 # number of patients
x = x_all[1:1+np-1] # Training x data
y = y_all[1:1+np-1] # Training y data

opt = Flux.Adam(0.0001) # Optimizer with learning rate


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
function symbolic_regression_nn(i, x, y, opt, lossfct, seed, nepochs_init, nepochs_prune)
    Random.seed!(seed)  # for reproducibility

    model = create_model(length(x[1].normalization), n_in, lossfct) # Create model (with parallel nn)

    ps = Flux.params(model.nn)     # parameters to train
    n_params = get_nparams(ps)  # nbr of traininable parameters

    print("Loss before training, thread ", i, ": ", round(model.loss(model.nn, x, y), digits=6), "\n")

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

    print("Loss after input pruning, thread ", i, ": ", round(model.loss(model.nn, x, y), digits=6), "\n") # loss after input pruning and training

    if isnan(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
        print("Loss is NaN in thread ", i, "\n")
        return false
    end

    # Network parameter pruning
    N_prune = 10 # nbr of parameters to prune in each network
    model = prunemodel(model, N_prune, n_params, x, y, verbose=false) # prune network parameters
    model, loss_train = train!(model, x, y, nepochs_prune, opt, ps, verbose=false) # train

    if isnan(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
        print("Loss is NaN in thread ", i, "\n")
        return false
    end

    # Now, prune fewer parameters at a time
    N_prune = 2 # nbr of parameters to prune in each network
    for j = 1:5#11
        model = prunemodel(model, N_prune, n_params, x, y, verbose=false) # prune network parameters

        if isnan(model.loss(model.nn, x, y)) || isinf(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
            print("Loss is NaN or Inf in thread ", i, "\n")
            return false
        end

        model, loss_train = train!(model, x, y, nepochs_prune, opt, ps, verbose=false) # train
    end

    N_prune = 1 # nbr of parameters to prune in each network
    for j = 1:2 # results in 12 parameters of final expression
        model = prunemodel(model, N_prune, n_params, x, y, verbose=false) # prune network parameters

        if isnan(model.loss(model.nn, x, y)) || isinf(model.loss(model.nn, x, y)) # If network is disconnected, abort and return false
            print("Loss is NaN or Inf in thread ", i, "\n")
            return false
        end

        model, loss_train = train!(model, x, y, nepochs_prune, opt, ps, verbose=false) # train
    end

    print("Final loss after training and pruning, thread ", i, ": ", round(model.loss(model.nn, x, y), digits=6), "\n") # loss after pruning and training
    print("All networks are connected, thread ", i, ": ", all(isnnconnected(model.nn)), "\n") # Check if all networks are still connected between input and output

    @save "bson/nn_thread$i$lossfct.bson" model # Save nn in a bson-file

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
function main(x, y, nthreads, opt, error_fct, seed, nepochs_init, nepochs_prune)
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

    @time Threads.@threads for i = 1:nthreads
        seed_i = seed + i - 1
        symbolic_regression_nn(i, x, y, opt, lossfct, seed_i, nepochs_init, nepochs_prune)
    end
    return true
end


# Testing
# seed = 1234
# n_epochs_init = 1            # Nbr of epochs to train before pruning and during inputpruning
# n_epochs_prune = n_epochs_init # Nbr of epochs to train between parameter pruning steps
# main(x, y, 2, opt, "MSE", seed, 10, 10)
# main(x[1:10], y[1:10], 4, opt, "ALE", seed, n_epochs_init, n_epochs_prune)
# main(x, y, 2, opt, "MdALE", seed, 1, 1)

