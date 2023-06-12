
# Plot results from population and individual predictions

include("../get_results_individualpk.jl")

θeleveld, y_eleveld, y_meas = get_y_eleveld() # Get output from Eleveld model

## Plotting ALE errors
mdale_indpred = MdALE.(y, y_indpred_ale)
mdale_eleveld = MdALE.(y, y_eleveld)

histogram((mdale_eleveld), bins=50, bar_width=0.1, alpha=0.5, label="Eleveld (ALE) errors", xlabel="(MdALE)", ylabel="# prediction errors")
histogram!((mdale_indpred), bins=50, bar_width=0.02, alpha=0.5, label="Individual fitted (MdALE) errors")


# # ELEVELD Scatter plot predicitions vs observations
p = scatter(line=2, markersize=3, shape=:circle, xaxis=:log, yaxis=:log, xlim=(0.001, 100), ylim=(0.001, 100), xlabel="Predicted blood plasma concentration [ug/ml]", ylabel="Observed concentration [ug/ml]", size=(600, 600))
i = 1
scatter!(p, y_eleveld[i], y[i], label="Eleveld model prediction", color=1, alpha=1)
scatter!(p, y_indpred_ale[i], y[i], label="Individual prediction (ALE loss)", color=2, alpha=1)
for i = 2:length(y)
    scatter!(p, y_eleveld[i], y[i], label="", color=1, alpha=1)
    scatter!(p, y_indpred_ale[i], y[i], label="", color=2, alpha=1)
end
plot!(p, 0.001:100, 0.001:100, linewidth=3, color=:red, label="")
display(p)
