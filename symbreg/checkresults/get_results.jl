# Get results from symbolic regression network
# Date: 230414

using Pkg
cd(@__DIR__)
cd("..")
Pkg.activate(".")

using Statistics, StatsPlots
using Flux
using BSON: @load

include("../fcts/fcts_nn.jl")               # Function to generate symbolic regression network and network training
include("../fcts/get_elevelddata.jl")          # Functions to generate training data from Eleveld data set
# include("../fcts/fcts_nn.jl")            # Function to generate symbolic regression network
# include("fcts/fcts_train.jl")               # Functions for network training
include("../fcts/fcts_eqreader.jl")            # Functions to read symbolic regression network into readable functions.
include("../fcts/fcts_analyseresults.jl")            # Plotting functions
# include("../fcts/fcts_predictionerrors.jl")    # Functions for prediction errors (MSE,ALE,MdALE etc)


x, y = get_elevelddata() # Get x and y data
np = length(x)

@load "bson/bson_ALE/nn_thread1loss_ALE.bson" model # thread 1 & 3

# @load "bson/bson_MSE/nn_thread5loss_MSE.bson" model # 4 and 5 are good.
# model_MSE = deepcopy(model)
# @load "bson/bson_ALE/nn_thread5loss_ALE.bson" model
model_ALE = deepcopy(model)

# Get predictions
# tobs_MSE, ypred_MSE = getpredictions(model_MSE.nn, x, y) # Get predictions from nn
tobs_ALE, ypred_ALE = getpredictions(model_ALE.nn, x, y) # Get predictions from nn

# Get function expressions from network
# func_exp_MSE = get_fctsfrommodel(model_MSE.nn) # Resulting (callable) functions for PK parameters in order [k10,k12,k13,k21,k31,V1]
func_exp_ALE = get_fctsfrommodel(model_ALE.nn) # Resulting (callable) functions for PK parameters in order [k10,k12,k13,k21,k31,V1] (note, these are scaled t0 0->1)

# tobs, yfunc_MSE = getpredictions(func_exp_MSE, x) # Check that equation reader is correct.
tobs, yfunc_ALE = getpredictions(func_exp_ALE, x) # Check that equation reader is correct.



# Errors

# include("fcts/getoutput_eleveldmodel.jl") # Eleveld model on the data set
θeleveld, y_eleveld, y_meas = get_predictions_eleveld()

# Print final errors
print("Costs for Eleveld model on data set: \n")
# print("mean(MSE): ", mean(MSE.(y, y_eleveld)), "\n") # 3.46
print("mean(MdALE): ", mean(MdALE.(y, y_eleveld)), "\n") # 0.3245
print("mean(MdLE): ", mean(MdLE.(y, y_eleveld)), "\n") # 0.0791
print("mean(MdAPE): ", mean(MdAPE.(y, y_eleveld)), "\n") # 34.78
print("mean(MdPE): ", mean(MdPE.(y, y_eleveld)), "\n") # 14.4

# print("Costs for Symbreg model with cost MSE on data set: \n")
# print("mean(MSE): ", mean(MSE.(y, ypred_MSE)), "\n") # 2.552
# print("mean(MdAPE): ", mean(MdAPE.(y, ypred_MSE)), "\n") # 35.37
# print("mean(MdPE): ", mean(MdPE.(y, ypred_MSE)), "\n") # 3.577
# print("mean(MdALE): ", mean(MdALE.(y, ypred_MSE)), "\n") # 0.345
# print("mean(MdLE): ", mean(MdLE.(y, ypred_MSE)), "\n") # -0.043

print("Costs for Symbreg model with cost ALE on data set: \n")
# print("mean(MSE): ", mean(MSE.(y, ypred_ALE)), "\n") # 3.03
print("mean(MdALE): ", mean(MdALE.(y, ypred_ALE)), "\n") # 0.2793
print("mean(MdLE): ", mean(MdLE.(y, ypred_ALE)), "\n") # -0.04889
print("mean(MdAPE): ", mean(MdAPE.(y, ypred_ALE)), "\n") # 27.37
print("mean(MdPE): ", mean(MdPE.(y, ypred_ALE)), "\n") # -0.26


## Get readable expressions
@syms x_age x_wgt x_bmi x_gdr x_AV # Symbolics, matches inputs.
# expr_MSE = get_readableexpr(model_MSE.nn) # Prints readable expressions to terminal (note scaling!)
expr_ALE = get_readableexpr(model_ALE.nn) # Prints readable (unscaled) expressions to terminal

##

# Rescale functions to "true", where all values are unscaled.
function get_true_equations(model, input)
    parallelnn = model.nn
    n_networks = length(parallelnn[1].paths)
    func_exp = Vector{Function}(undef, n_networks)

    chain = parallelnn[1].paths
    eqout = Vector{String}(undef, length(chain))

    for i = 1:n_networks
        m = chain[i]
        Y1 = layer2string(m, 1, input, false)
        Y2 = layer2string(m, 2, Y1, false)
        Y3 = layer2string(m, 3, Y2, false)
        Y4 = layer2string(m, 4, Y3, false)
        Y = layer2string(m, 5, Y4, false)[1]
        Yscaled = "$(x[1].normalization[i])" * Y
        eqout[i] = string(expand(eval(Meta.parse(Yscaled))))
        func_exp[i] = eval(Meta.parse("(x_age,x_wgt,x_bmi,x_gdr,x_AV) -> " * eqout[i]))
    end
    return eqout, func_exp
end

# print(expand(eval(Meta.parse(eqout[5]))))

# Full expressions, scale inputs to normal
# maxbmi = 52.84713965
input = ["(x_age/88)"; "(x_wgt/160)"; "(x_bmi/52.84713965)"; "x_gdr"; "x_AV"] # scale inpute back to normal
eqout, func_exp = get_true_equations(model_ALE, input)

print("Final k10 expression: \n", eqout[1], "\n") # Final k10 expression [1/s]
print("Final k12 expression: \n", eqout[2], "\n") # Final k12 expression [1/s]
print("Final k21 expression: \n", eqout[4], "\n") # Final k21 expression [1/s]
print("Final V1 expression: \n", eqout[6], "\n") # Final V1 expression, [ml]

# Male
input = ["(x_age/88)"; "(x_wgt/160)"; "(x_bmi/52.84713965)"; "-0.5"; "x_AV"]
eqout, func_exp = get_true_equations(model_ALE, input)

print("Final k13 expression male: \n", eqout[3], "\n") # Final k13 expression male [1/s]
print("Final k31 expression male: \n", eqout[5], "\n") # Final k31 expression male [1/s]

# Female
input = ["(x_age/88)"; "(x_wgt/160)"; "(x_bmi/52.84713965)"; "0.5"; "x_AV"]
eqout, func_exp = get_true_equations(model_ALE, input)

print("Final k13 expression female: \n", eqout[3], "\n") # Final k13 expression female [1/s]
print("Final k31 expression female: \n", eqout[5], "\n") # Final k31 expression female [1/s]



# ## Write results to file

# errors
# df_mdale = DataFrame("MdALE errors symbolic regression" => mdale, "MdALE errors Eleveld" => mdale_eleveld)
# CSV.write("csv/MdALE_symbreg_eleveld.csv", df_mdale)

# predictions
# df_pred_obs = DataFrame("Observed concentration"=>y_obs,"Predicted concentration symbolic regression ALE"=> pred_ale, "Predicted concentration Eleveld"=> pred_eleveld)
# CSV.write("csv/predicted_observed_conc.csv", df_pred_obs)