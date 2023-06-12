# Get results from optimization where we seek a constant PK model over all patients.

using Pkg
cd(@__DIR__)
Pkg.activate(".")

using Statistics

include("fcts/constantmodel.jl")

function getpredictions(ϕhat_scaled::Vector{Float64}, x, y)
    ysim = Vector{Vector{Float32}}(undef, length(y))
    tobs = Vector{Vector{Float32}}(undef, length(y))
    ϕhat = ϕhat_scaled .* x[1].normalization

    for i in eachindex(y)
        ysim[i] = pk3sim(ϕhat, x[i].u, x[i].v, x[i].hs, x[i].youts)
        tobs[i] = x[i].time[x[i].youts]
    end
    return tobs, ysim, ϕhat
end

# get predictions
tobs, y_poppred_ale,constantmodel = getpredictions(ϕhat_ale, x, y);

print("Constant PK model parameters are: ", constantmodel, "\n")

## Prediction errors
print("Constant PK model with ALE as cost function: \n")
print("mean(MdALE): ", mean(MdALE.(y, y_poppred_ale)), "\n") # 0.5007
print("mean(MdLE): ", mean(MdLE.(y, y_poppred_ale)), "\n") # 0.1405
print("mean(MdAPE): ", mean(MdAPE.(y, y_poppred_ale)), "\n") # 210.5
print("mean(MdPE): ", mean(MdPE.(y, y_poppred_ale)), "\n") # 181.27 

## Save to CSV
# pred_eleveld = reduce(vcat, y_eleveld)
# pred_pop = reduce(vcat, y_poppred_ale)
# y_obs = reduce(vcat, y)

# df_ale = DataFrame("MdALE errors population fit optim" => mean(MdALE.(y, y_poppred_ale)), "MdALE errors Eleveld" => mean(MdALE.(y, y_eleveld)))
# CSV.write("csv/MdALE_population_eleveld.csv", df_ale)
# df_pred_obs = DataFrame("Observed concentration" => y_obs, "Predicted concentration population ALE optim" => pred_pop, "Predicted concentration Eleveld" => pred_eleveld)
# CSV.write("csv/pred_obs_population_ale.csv", df_pred_obs)

