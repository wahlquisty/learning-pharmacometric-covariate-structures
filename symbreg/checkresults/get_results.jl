# Get results from symbolic regression network files in bson/ folder

# Date: 230414

using Pkg
cd(@__DIR__)
cd("..")
Pkg.activate(".")

using Statistics, StatsPlots
using Flux
using BSON: @load

include("../fcts/fcts_nn.jl")                  # Function to generate symbolic regression network and network training
include("../fcts/get_elevelddata.jl")          # Functions to generate training data from Eleveld data set
include("../fcts/fcts_eqreader.jl")            # Functions to read symbolic regression network into readable functions.
include("../fcts/fcts_analyseresults.jl")      # Plotting functions and prediction errors (MSE,ALE,MdALE etc)

# Get training data
x, y = get_elevelddata() # Get x and y data
np = length(x)

# Load bson file for final result
@load "bson/bson_ALE/nn_thread1loss_ALE.bson" model # thread 1 & 3
model_ALE = deepcopy(model)

tobs_ALE, ypred_ALE = getpredictions(model_ALE.nn, x, y) # Get predictions from nn

# Get function expressions from network
func_exp_ALE = get_fctsfrommodel(model_ALE.nn) # Resulting (callable) functions for PK parameters in order [k10,k12,k13,k21,k31,V1] (note, these are scaled t0 0->1)
tobs, yfunc_ALE = getpredictions(func_exp_ALE, x) # Check that equation reader is correct.


## Errors
θeleveld, y_eleveld, y_meas = get_predictions_eleveld() # Eleveld predictions on the data set

# Print final errors
print("Costs for Eleveld model on data set: \n")
# print("mean(MSE): ", mean(MSE.(y, y_eleveld)), "\n") # 3.46
print("mean(MdALE): ", mean(MdALE.(y, y_eleveld)), "\n") # 0.3245
print("mean(MdLE): ", mean(MdLE.(y, y_eleveld)), "\n") # 0.0791
print("mean(MdAPE): ", mean(MdAPE.(y, y_eleveld)), "\n") # 34.78
print("mean(MdPE): ", mean(MdPE.(y, y_eleveld)), "\n") # 14.4

print("Costs for Symbreg model with cost ALE on data set: \n")
# print("mean(MSE): ", mean(MSE.(y, ypred_ALE)), "\n") # 3.03
print("mean(MdALE): ", mean(MdALE.(y, ypred_ALE)), "\n") # 0.2793
print("mean(MdLE): ", mean(MdLE.(y, ypred_ALE)), "\n") # -0.04889
print("mean(MdAPE): ", mean(MdAPE.(y, ypred_ALE)), "\n") # 27.37
print("mean(MdPE): ", mean(MdPE.(y, ypred_ALE)), "\n") # -0.26


## Get readable expressions
@syms x_age x_wgt x_bmi x_gdr x_AV # Symbolics, matches inputs.
expr_ALE = get_readableexpr(model_ALE.nn) # Prints readable (unscaled!) expressions to terminal

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
# max bmi = 52.84713965, max age = 88, max wgt = 160
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



## Write results to file

# errors
# df_mdale = DataFrame("MdALE errors symbolic regression" => mdale, "MdALE errors Eleveld" => mdale_eleveld)
# CSV.write("csv/MdALE_symbreg_eleveld.csv", df_mdale)

# predictions
# df_pred_obs = DataFrame("Observed concentration"=>y_obs,"Predicted concentration symbolic regression ALE"=> pred_ale, "Predicted concentration Eleveld"=> pred_eleveld)
# CSV.write("csv/predicted_observed_conc.csv", df_pred_obs)