
using Plots

function getpredictions(parallelnn::T_parallelnn, x, y)
    # θpred = reshape(θpred, 6, length(y))
    ysim = Vector{Vector{Float32}}(undef, length(y))
    tobs = Vector{Vector{Float32}}(undef, length(y))
    # θhat = Vector{Vector{Float32}}(undef, length(y))

    for i in eachindex(y)
        θhat_scaled = get_modeloutput(parallelnn, x[i].covariates)
        θhat = θhat_scaled .* x[i].normalization # Scale PK parameters (volumes and clearances) from (0,1) to range (0,max(V/CL)) [l] and [l/min]
        # tobs[i] = [1,2,3,4,5,6]
        # end
        # return tobs, θhat
        # V1, V2, V3, CL, Q2, Q3 = θhat
        # # Create PK model
        # CL /= 60 # [l/s]
        # Q2 /= 60 # [l/s]
        # Q3 /= 60 # [l/s]
        # # drug transfer rate constants
        # k10 = CL / V1                     # [1/s]
        # k12 = Q2 / V1                     # [1/s]
        # k13 = Q3 / V1                     # [1/s]
        # k21 = Q2 / V2                     # [1/s]
        # k31 = Q3 / V3                     # [1/s]

        # V1ml = V1 * 1000 # [ml] to match units of input

        # ϕ = [k10, k12, k13, k21, k31, V1ml] # use rate constants at input to simulation

        # θhat = θpred .* x[i].normalization # scaling to normal values
        # ysim[i] = pk3sim(ϕ, x[i].u, x[i].v, x[i].hs, x[i].youts)
        ysim[i] = pk3sim(θhat, x[i].u, x[i].v, x[i].hs, x[i].youts)

        tobs[i] = x[i].time[x[i].youts]
    end
    return tobs, ysim
end

# Plot predictions again using the functions instead to see that they match (check equationreader result)
function getpredictions(func_exp, x)
    yfunc = Vector{Vector{Float32}}(undef, np)
    tobs = Vector{Vector{Float32}}(undef, np)
    for i = 1:np
        cov1 = x[i].covariates
        θfunc = [func_exp[1](cov1...), func_exp[2](cov1...), func_exp[3](cov1...), func_exp[4](cov1...), func_exp[5](cov1...), func_exp[6](cov1...)]
        yhat = pk3sim(θfunc .* x[i].normalization, x[i].u, x[i].v, x[i].hs, x[i].youts)
        yfunc[i] = yhat
        tobs[i] = (x[i].time)[x[i].youts]
    end
    return tobs, yfunc
end


function plottrainingloss(training_loss)
    p = scatter(training_loss, title="Training loss", xlabel="Epoch", ylabel="Loss", label="")
    display(p)
end

function plotprediction(tobs, y, ypred)
    p1 = scatter(tobs[1] ./ 60, y[1], label="y", ylabel="Cp", xlabel="Time (min)")
    scatter!(p1, tobs[1] ./ 60, ypred[1], label="ŷ")
    p2 = scatter(tobs[2] ./ 60, y[2], label="y")
    scatter!(p2, tobs[2] ./ 60, ypred[2], label="ŷ", ylabel="Cp", xlabel="Time (min)")
    p3 = scatter(tobs[3] ./ 60, y[3], label="y")
    scatter!(p3, tobs[3] ./ 60, ypred[3], label="ŷ", ylabel="Cp", xlabel="Time (min)")
    p4 = scatter(tobs[4] ./ 60, y[4], label="y")
    scatter!(p4, tobs[4] ./ 60, ypred[4], label="ŷ", ylabel="Cp", xlabel="Time (min)")
    p5 = scatter(tobs[5] ./ 60, y[5], label="y")
    scatter!(p5, tobs[5] ./ 60, ypred[5], label="ŷ", ylabel="Cp", xlabel="Time (min)")
    # p6 = scatter(y[6], label="y")
    # scatter!(p6, ypred[6], label="ŷ", ylabel="Q3", xlabel="Patient #")

    p = plot(p1, p2, p3, p4, p5, plot_title="Predicted concentrations (first 5 patients)", layout=(3, 2), size=(500, 600))
    display(p)
end

function plotpredictedPKparams(parallelnn::T_parallelnn, x) # [k10, k12, k13, k21, k31, V1ml]
    
    θhat = zeros(length(x), 6)

    for i in eachindex(x)
        θhat_scaled = get_modeloutput(parallelnn, x[i].covariates)
        θhat[i, :] = θhat_scaled .* x[i].normalization
    end
    # θtrue = θnorm.*x[1].normalization
    p1 = scatter(θhat[:, 1], label="Prediction", ylabel="k10 [1/s]", xlabel="Patient #")
    p2 = scatter(θhat[:, 2], label="Prediction", ylabel="k12", xlabel="Patient #")
    p3 = scatter(θhat[:, 3], label="Prediction", ylabel="k13", xlabel="Patient #")
    p4 = scatter(θhat[:, 4], label="Prediction", ylabel="k21", xlabel="Patient #")
    p5 = scatter(θhat[:, 5], label="Prediction", ylabel="k31", xlabel="Patient #")
    p6 = scatter(θhat[:, 6]./1000, label="Prediction", ylabel="V1 [l]", xlabel="Patient #")

    p = plot(p1, p2, p3, p4, p5, p6, plot_title="Predicted PK parameters", layout=(3, 2), size=(500, 600))
    display(p)
end


###################################### Function for computing prediction errors
using Statistics

function MSE(y_meas, y_pred)
    mse = sum(abs2.(y_meas .- y_pred)) ./ length(y_pred)
    return mse
end

function MdAPE(y_meas, y_pred) # Median Absolute Prediction Error: median(abs((C_observed - C_predicted)/C_predicted* 100 %))
    mdape = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdape = 100 * median(abs.((y_meas[ind] .- y_pred[ind]) ./ y_pred[ind])) # %
    end
    return mdape
end

function MdPE(y_meas, y_pred) # Absolute Prediction Error: abs((C_observed - C_predicted)/C_predicted* 100 %))
    ape = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        ape = median(100 * (y_meas[ind] .- y_pred[ind]) ./ y_pred[ind]) # %
    end
    return ape
end

function MdALE(y_meas, y_pred) # Median Absolute Logarithmic Error: median(abs(log(C_observed/C_predicted)))
    mdale = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdale = median(abs.(log.(y_meas[ind] ./ abs.(y_pred[ind]))))
    end
    return mdale
end


function ALE(y_meas, y_pred) # Absolute Logarithmic Error: sum(abs(log(C_observed/C_predicted)))
    ale = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        ale = sum(abs.(log.(y_meas[ind] ./ abs.(y_pred[ind]))))
    end
    return ale
end

function MdLE(y_meas, y_pred)
    le = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        le = median(log.(y_meas[ind] ./ abs.(y_pred[ind])))
    end
    return le
end