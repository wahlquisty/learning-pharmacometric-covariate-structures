# Get results from optimization using Optim.jl to find individual PK parameters for each patient.

using Pkg
cd(@__DIR__)
Pkg.activate(".")

using Statistics

# include("fcts/get_elevelddata.jl")       # Predictions from Eleveld model on the data set
include("fcts/individualmodels.jl")         # Optimization for individual PK parameters. Running in threads.

#######################################################################################################################

## Get predictions

function getpredictions(ϕhats_scaled::Vector{Vector{Float64}}, x, y)
    ysim = Vector{Vector{Float32}}(undef, length(y))
    tobs = Vector{Vector{Float32}}(undef, length(y))

    for i in eachindex(y)
        ϕhat = ϕhats_scaled[i] .* x[i].normalization

        ysim[i] = pk3sim(ϕhat, x[i].u, x[i].v, x[i].hs, x[i].youts)
        tobs[i] = x[i].time[x[i].youts]
    end
    return tobs, ysim
end

# tobs, y_indpred_mse = getpredictions(ϕhats_mse, x, y);
tobs, y_indpred_ale = getpredictions(ϕhats_ale, x, y);


## Prediction errors
# print("Individual PK models with MSE as cost function: \n")
# print("mean(MSE): ", mean(MSE.(y, y_indpred_mse)), "\n") # 0.4599
# print("mean(MdAPE): ", mean(MdAPE.(y, y_indpred_mse)), "\n") # 11.42 
# print("mean(MdPE): ", mean(MdPE.(y, y_indpred_mse)), "\n") # -0.149 
# print("mean(MdALE): ", mean(MdALE.(y, y_indpred_mse)), "\n") # 0.1136
# print("mean(MdLE): ", mean(MdLE.(y, y_indpred_mse)), "\n") # -0.00507

print("Individual PK models with ALE as cost function: \n")
# print("mean(MSE): ", mean(MSE.(y, y_indpred_ale)), "\n") # 0.735
print("mean(MdALE): ", mean(MdALE.(y, y_indpred_ale)), "\n") # 0.0623
print("mean(MdLE): ", mean(MdLE.(y, y_indpred_ale)), "\n") # 1.65e-4
print("mean(MdAPE): ", mean(MdAPE.(y, y_indpred_ale)), "\n") # 6.21 
print("mean(MdPE): ", mean(MdPE.(y, y_indpred_ale)), "\n") # 0.0178 

# Save to CSV
# pred_eleveld = reduce(vcat, y_eleveld)
# pred_ind = reduce(vcat, y_indpred_mse)
# y_obs = reduce(vcat, y)

# df_mse = DataFrame("MSE errors individual fit optim" => MSE.(y, y_indpred_mse), "MSE errors Eleveld" => MSE.(y, y_eleveld))
# CSV.write("csv/mse_individual_eleveld.csv", df_mse)
# df_pred_obs = DataFrame("Observed concentration" => y_obs, "Predicted concentration individual mse optim" => pred_ind, "Predicted concentration Eleveld" => pred_eleveld)
# CSV.write("csv/pred_obs_individual_mse.csv", df_pred_obs)


# Save to CSV
# pred_eleveld = reduce(vcat, y_eleveld)
# pred_ind = reduce(vcat, y_indpred_ale)
# y_obs = reduce(vcat, y)

# df_ale = DataFrame("MdALE errors individual fit optim" => mean(MdALE.(y, y_indpred_ale)), "MdALE errors Eleveld" => mean(MdALE.(y, y_eleveld)))
# CSV.write("csv/MdALE_individual_eleveld.csv", df_ale)
# df_pred_obs = DataFrame("Observed concentration" => y_obs, "Predicted concentration individual ALE optim" => pred_pop, "Predicted concentration Eleveld" => pred_eleveld)
# CSV.write("csv/pred_obs_individual_ale.csv", df_pred_obs)
