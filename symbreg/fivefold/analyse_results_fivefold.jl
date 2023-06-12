# Get results from symbolic regression network five fold cross validation
# Date: 230601

using Pkg
cd(@__DIR__)
cd("..")
Pkg.activate(".")

using Random, Statistics, StatsPlots
using Flux
using BSON: @load

include("../fcts/fcts_nn.jl")            # Function to generate and train symbolic regression network
include("../fcts/get_elevelddata.jl")          # Functions to generate training data from Eleveld data set
# include("../fcts/fcts_train.jl")               # Functions for network training
include("../fcts/fcts_flatstructure.jl")       # Functions to destructure (and rebuild) symbolic regression network to a flat structure for Hessian computations
include("../fcts/fcts_pruning.jl")             # Functions to prune inputs and parameters of symbolic regression network
# include("../fcts/fcts_eqreader.jl")            # Functions to read symbolic regression network into readable functions.
include("../fcts/fcts_analyseresults.jl")
# include("../fcts/fcts_predictionerrors.jl")


x, y = get_elevelddata() # Get x and y data
np = length(x)

## Compute average errors over the test set(s)

# Get the test data
seed = 1234
Random.seed!(seed)

x_all, y_all = get_elevelddata()
np = 10 #1031
n_in = size(x_all[1].covariates, 1) # nbr of inputs to each network (number of covariates)
percentfold = 0.8
opt = Flux.Adam(0.0001) # Optimizer

MdALEs = zeros(5)
MdLEs = zeros(5)
MdAPEs = zeros(5)
MdPEs = zeros(5)

for fold = 1:5
    # Patient data
    shuffled_indices = randperm(np)

    training_indices = shuffled_indices[1:Int(ceil(np * percentfold))]
    # training_indices = shuffled_indices[1:Int(ceil(np / 2))]
    test_indices = shuffled_indices[Int(ceil(np * percentfold))+1:np]
    xtrain = x_all[training_indices]
    ytrain = y_all[training_indices]
    xtest = x_all[test_indices]
    ytest = y_all[test_indices]

    # read resulting model
    @load "bson/bson_fivefold/nn_loss_ALEfold$fold.bson" model

    # Get predictions
    tobs_train, ypred_train = getpredictions(model.nn, xtrain, ytrain) # Get predictions from nn
    tobs_test, ypred_test = getpredictions(model.nn, xtest, ytest) # Get predictions from nn

    # Errors
   print("Costs for Symbreg model with cost ALE on training set for fold $fold: \n")
    # print("mean(MSE): ", mean(MSE.(y, ypred_ALE)), "\n") # 3.03
    print("mean(MdALE): ", mean(MdALE.(ytrain, ypred_train)), "\n") # 0.2793
    print("mean(MdLE): ", mean(MdLE.(ytrain, ypred_train)), "\n") # -0.04889
    print("mean(MdAPE): ", mean(MdAPE.(ytrain, ypred_train)), "\n") # 27.37
    print("mean(MdPE): ", mean(MdPE.(ytrain, ypred_train)), "\n") # -0.26

    print("Costs for Symbreg model with cost ALE on test set for fold $fold: \n")
    # print("mean(MSE): ", mean(MSE.(y, ypred_ALE)), "\n") # 3.03
    print("mean(MdALE): ", mean(MdALE.(ytest, ypred_test)), "\n") # 0.2793
    print("mean(MdLE): ", mean(MdLE.(ytest, ypred_test)), "\n") # -0.04889
    print("mean(MdAPE): ", mean(MdAPE.(ytest, ypred_test)), "\n") # 27.37
    print("mean(MdPE): ", mean(MdPE.(ytest, ypred_test)), "\n") # -0.26

    MdALEs[fold] = mean(MdALE.(ytest, ypred_test))
    MdLEs[fold] = mean(MdLE.(ytest, ypred_test))
    MdAPEs[fold] = mean(MdAPE.(ytest, ypred_test))
    MdPEs[fold] = mean(MdPE.(ytest, ypred_test))

end

mean(MdALEs)
mean(MdLEs)
mean(MdAPEs)
mean(MdPEs)

# Get max and min
minimum(MdALEs)
maximum(MdALEs)
