# Analyse results from pruned symbolic regression network
# Reads model from jld or bson file.
# Date: 230130

using Pkg
cd(@__DIR__)
cd("..")
Pkg.activate(".")

using Plots, Statistics, StatsPlots

include("get_results.jl") # Get results from BSON file

# Check that equationreader gives the correct output. Plot prediction for first 5 patients.
plotprediction(tobs, y, ypred_ALE)
plotprediction(tobs, y, yfunc_ALE)


## MdALE for ALE symbreg (and Eleveld model)
mdale = MdALE.(y, ypred_ALE)
mdale_eleveld = MdALE.(y, y_eleveld)


## Plot results

# Scatter plot predicitions vs observations

# ALE
p = scatter(line=2, markersize=3, shape=:circle, label="Prediction", xlim=(0.001, 50), ylim=(0.001, 50), xlabel="Predicted blood plasma concentration [ug/ml]", ylabel="Observed concentration [ug/ml]", size=(600, 600), xscale=:log, yscale=:log)
for i in eachindex(y)
    scatter!(p, ypred_ALE[i], y[i], label="", color=1)
end
plot!(0.001:1:50, 0.001:1:50, color=:red, label="")
display(p)


## Histogram over prediction errors
bins = 0:1:105
# ALE
histogram(mdale_eleveld, bins=bins, alpha=0.5, label="Eleveld MdALE errors", xlabel="MdALE", ylabel="# prediction errors")
histogram!(mdale, bins=bins, alpha=0.5, label="Symb regression MdALE errors")


# Scatter plot predicitions vs observations vs Eleveld model (ALE only)
p = scatter(line=2, markersize=3, shape=:circle, xaxis=:log, yaxis=:log, xlim=(0.001, 50), ylim=(0.001, 50), xlabel="Predicted blood plasma concentration [ug/ml]", ylabel="Observed concentration [ug/ml]", size=(600, 600))
i = 1
scatter!(p, y_eleveld[i], y[i], label="Eleveld model prediction", color=1, alpha=1, markersize=2)
scatter!(p, ypred_ALE[i], y[i], label="Symbolic regression prediction (ALE)", color=2, alpha=1, markersize=2)
for i = 2:length(y)
    scatter!(p, y_eleveld[i], y[i], label="", color=1, alpha=1, markersize=2)
    scatter!(p, ypred_ALE[i], y[i], label="", color=2, alpha=1, markersize=2)
end
display(p)


## Box plots
pred_eleveld = reduce(vcat, y_eleveld)
pred_ale = reduce(vcat, ypred_ALE)
y_obs = reduce(vcat, y)


## Box plot of errors

# ALE
boxplot(repeat(["Eleveld", "Symbolic regression"], outer=length(mdale_eleveld)), [mdale; mdale_eleveld], ylabel="MdALE errors", legend=false, size=(600, 600))



## Write results to file

# errors
# df_mdale = DataFrame("MdALE errors symbolic regression" => mdale, "MdALE errors Eleveld" => mdale_eleveld)
# CSV.write("csv/MdALE_symbreg_eleveld.csv", df_mdale)

# predictions
# df_pred_obs = DataFrame("Observed concentration"=>y_obs,"Predicted concentration symbolic regression ALE"=> pred_ale, "Predicted concentration Eleveld"=> pred_eleveld)
# CSV.write("csv/predicted_observed_conc.csv", df_pred_obs)