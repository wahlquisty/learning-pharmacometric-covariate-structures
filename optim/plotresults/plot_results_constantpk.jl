
# Plot results from population and population predictions

include("../get_results_constantpk.jl") # Get results from constant PK model over the data set. Predictions are given by y_poppred_mse and y_poppred_ale


#######################################################################################################################
## ALE

θeleveld, y_eleveld, y_meas = get_y_eleveld() # Get outpot from Eleveld model

## Plotting ALE errors
mdale_poppred = MdALE.(y, y_poppred_ale)
mdale_eleveld = MdALE.(y, y_eleveld)

histogram((mdale_poppred), bins=50, bar_width=0.2, alpha=0.5, color=2, label="Population predicted (ale) errors (same PK params)")
histogram!((mdale_eleveld), bins=50, bar_width=0.1, alpha=0.5, color=1, label="Eleveld (MdALE) errors", xlabel="(MdALE)", ylabel="# prediction errors")


# ELEVELD Scatter plot predicitions vs observations
p = scatter(line=2, markersize=3, shape=:circle, xaxis=:log, yaxis=:log, xlim=(0.001, 100), ylim=(0.001, 100), xlabel="Predicted blood plasma concentration [ug/ml]", ylabel="Observed concentration [ug/ml]", size=(600, 600))
i = 1
scatter!(p, y_eleveld[i], y[i], label="Eleveld model prediction", color=1, alpha=1)
scatter!(p, y_poppred_ale[i], y[i], label="Population prediction (same PK, cost ALE)", color=2, alpha=1)
for i = 2:length(y)
    scatter!(p, y_eleveld[i], y[i], label="", color=1, alpha=1)
    scatter!(p, y_poppred_ale[i], y[i], label="", color=2, alpha=1)
end
plot!(p, 0.001:100, 0.001:100, linewidth=3, color=:red, label="")
display(p)

