# Evaluate the cost of the Eleveld model in terms of prediction errors, for comparison
# Date: 230118

using CSV, DataFrames, StaticArrays

##############
# include("../data/get_data.jl") # Functions to get input data from files (model parameters and input data)
# include("fcts_predictionerrors.jl")

# using CSV, DataFrames, StaticArrays
# using BenchmarkTools

using Distributions, Random

struct InputData
    covariates # vector or real
    u::Vector
    v::Vector
    time::Vector # needed?
    hs::Vector
    youts::Vector
    normalization::Vector # to convert volumes and clearances
end

##############
# include("getdata.jl") # Functions to get input data from files (model parameters and input data)


## All patients data
function getelevelddata()
    # # Load datafiles
    modeldf = CSV.read("data/eleveld_modelparams.csv", DataFrame) # Model parameters V1,V2,... for PK model
    inputdf = CSV.read("data/eleveld_infusiondata.csv", DataFrame) # Input vector for simulation, with time and rates/ doses for infusions and boluses and if a time instance was a measurement
    df = CSV.read("data/dataframe.csv", DataFrame) # covariate data

    np = 1033
    y = Vector{Vector{Float32}}(undef, np)
    x = Vector{InputData}(undef, np)
    # input_normalization, pkparams_normalization = getnormalizations() # normalizations for input covariates and pkparams
    input_normalization = [88.0, 160.0]
    pkparams_normalization = [0.008468822, 0.013328808, 0.00356759, 0.0057506226, 0.000102905105, 9440.525] # for k10, k12 etc
    # normalization = [0.005, 0.005, 0.002, 0.001, 0.0001, 10000] # for k10, k12 etc

    nstudy = 30 # nbr of studies
    # studynbr = 1
    for studynbr = 1:nstudy
        study_df, firstid, lastid = getstudydata(modeldf, studynbr) # get dataframe for this study
        # id = 1
        for id = firstid:lastid
            if id in (893, 897) # no measurements exists for these patients
                continue
            end
            θ, infusionrate, bolusdose, t, hs, youts = getpatientdata(id, modeldf, inputdf) # get patient data
            age, wgt, hgt, m1f2, a1v2, noadd1add2 = getcovariates(df, id)
            u = infusionrate# ug/min             / 1000 # mg/min
            v = bolusdose # ug             /1000 # mg

            continuous_inputs = Float32.([age, wgt])

            covariates = [continuous_inputs[1] / input_normalization[1], continuous_inputs[2] / input_normalization[2], m1f2 - 1.5, a1v2 - 1.5] # scaled continuous variables
            # covariates = [continuous_inputs[1] / input_normalization[1], continuous_inputs[2] / input_normalization[2], m1f2 - 1, a1v2 - 1] # scaled continuous variables

            y_meas = getymeas(df, id)
            y[id] = y_meas
            # y[id] = θ

            x[id] = InputData(covariates, u, v, t, hs, youts, pkparams_normalization)
        end
    end

    # remove those without measurements
    deleteat!(y, 897)
    deleteat!(x, 897)
    deleteat!(y, 893)
    deleteat!(x, 893)

    return x, y
end


# Functions to get data from data files put in global variables modeldf, inputdf, idxdf
# modeldf: Model parameters V1,V2,... for PK model
# inputdf: Input vector for simulation, with time and rates/ doses for infusions and boluses and times for observation

# Functions to extract input data and model parameters from files
function getstudydata(modeldf, studynbr)
    study_df = modeldf[in([studynbr]).(modeldf.StudyNbr), :] # get all rows with nbr = studynbr
    idnbrs = study_df[!, :ID] # get array with all id-numbers in current study
    last = idnbrs[length(idnbrs)] # last id-number of current study
    first = idnbrs[1] # first id-number of current study
    return study_df, first, last
end

function getindmeas(ismeasurements)
    indmeas = zeros(Int64, Int64(sum(ismeasurements))) # final vector
    ind_meas = findall(x -> x >= 1, ismeasurements) #
    f = 1 # first available index in indmeas
    for i in ind_meas
        for j = 1:Int64(ismeasurements[i])
            indmeas[f] = Int64(i)
            f += 1
        end
    end
    return indmeas
end

function getpatientdata(id, modeldf, inputdf)
    subject = modeldf[in([id]).(modeldf.ID), :] # get all rows with nbr = id

    V1 = subject[!, :V1][1]
    V2 = subject[!, :V2][1]
    V3 = subject[!, :V3][1]
    CL = subject[!, :CL][1]
    Q2 = subject[!, :Q2][1]
    Q3 = subject[!, :Q3][1]

    # drug transfer rate constants
    # CL /= 60 # [l/s]
    # Q2 /= 60
    # Q3 /= 60

    θ = @SVector [V1, V2, V3, CL, Q2, Q3] # PK model parameter vector

    # k10 = CL / V1 # [1/s]
    # k12 = Q2 / V1
    # k13 = Q3 / V1
    # k21 = Q2 / V2
    # k31 = Q3 / V3
    # V1 *= 1000 # [ml] to match units of input

    # θ = @SVector [k10, k12, k13, k21, k31, V1] # PK model parameter vector

    # Input data
    inputs = inputdf[in([id]).(inputdf.ID), :]

    time = Float32.(inputs[!, :Time]) # [s]
    infusionrate = Float32.(inputs[!, :InfusionRate]) # [ug/min]
    bolusdose = Float32.(inputs[!, :Bolus]) # [ug]
    ismeas = Float32.(inputs[!, :IsMeasurement]) # 1 = yes, 0 = no

    h = diff(time) # durations between inputs
    h = [h; h[end]] # assume last h is same as second last.
    # Indices of measured concentrations
    youts = getindmeas(ismeas)
    # time = time[youts]
    return Float32.(θ), infusionrate, bolusdose, time, h, youts
end

function getcovariates(df, id)
    subject = df[in([id]).(df.ID), :] # get all rows with nbr = id

    AGE = subject[!, :AGE][1]
    WGT = subject[!, :WGT][1]
    HGT = subject[!, :HT][1]
    M1F2 = subject[!, :M1F2][1]
    A1V2 = subject[!, :A1V2][1]
    NOADD1ADD2 = subject[!, :NOADD1ADD2][1]

    return AGE, WGT, HGT, M1F2, A1V2, NOADD1ADD2
end

function getymeas(df, id)
    subject = df[in([id]).(df.ID), :] # get all rows with nbr = id

    CP = subject[!, :CP]
    true_conc = CP[findall(x -> x > 0, CP)]
    y_meas = Float32.(true_conc)
    return y_meas
end

function get_y_eleveld()

    modeldf = CSV.read("data/eleveld_modelparams.csv", DataFrame) # Model parameters V1,V2,... for PK model
    inputdf = CSV.read("data/eleveld_infusiondata.csv", DataFrame) # Input vector for simulation, with time and rates/ doses for infusions and boluses and if a time instance was a measurement
    df = CSV.read("data/dataframe.csv", DataFrame) # covariate data

    y = Vector{Vector{Float32}}(undef, 1033)
    θ = Vector{Vector{Float32}}(undef, 1033)
    x_reduced = Vector{InputData}(undef, 1033)

    nstudy = 30 # nbr of studies
    # studynbr = 1
    for studynbr = 1:nstudy
        study_df, firstid, lastid = getstudydata(modeldf, studynbr) # get dataframe for this study
        # id = 1
        for id = firstid:lastid
            if id in (893, 897) # no measurements ex_reducedists for these patients
                continue
            end
            θi, infusionrate, bolusdose, t, hs, youts = getpatientdata(id, modeldf, inputdf) # get patient data
            age, wgt, hgt, m1f2, a1v2, noadd1add2 = getcovariates(df, id)
            u = infusionrate# ug/min             / 1000 # mg/min
            v = bolusdose # ug             /1000 # mg
            covariates = Float32.([age, wgt])

            y_meas = getymeas(df, id)
            y[id] = y_meas
            θ[id] = θi
            # y[id] = θ

            x_reduced[id] = InputData(covariates, u, v, t, hs, youts, ones(Float32, 6))
        end
    end

    # remove those without measurements
    deleteat!(y, 897)
    deleteat!(y, 893)
    deleteat!(θ, 897)
    deleteat!(θ, 893)
    deleteat!(x_reduced, 897)
    deleteat!(x_reduced, 893)

    # Eleveld predictions
    y_eleveld = similar(y)
    tobs = similar(y)
    for i in eachindex(y)
        V1, V2, V3, CL, Q2, Q3 = θ[i]

        # Create PK model
        CL /= 60 # [l/s]
        Q2 /= 60 # [l/s]
        Q3 /= 60 # [l/s]
        # drug transfer rate constants
        k10 = CL / V1                     # [1/s]
        k12 = Q2 / V1                     # [1/s]
        k13 = Q3 / V1                     # [1/s]
        k21 = Q2 / V2                     # [1/s]
        k31 = Q3 / V3                     # [1/s]

        V1ml = V1 * 1000 # [ml] to match units of input

        ϕ = [k10, k12, k13, k21, k31, V1ml] # use rate constants at input to simulation

        yhat = pk3sim(ϕ, x_reduced[i].u, x_reduced[i].v, x_reduced[i].hs, x_reduced[i].youts)
        y_eleveld[i] = yhat
        tobs[i] = (x_reduced[i].time)[x_reduced[i].youts]
    end
    return θ, y_eleveld, y
end

# θeleveld, y_eleveld, y_meas = get_y_eleveld()

## Functions for computing prediction errors
using Statistics

function MSE(y_meas, y_pred)
    mse = sum(abs2.(y_meas .- y_pred)) ./ length(y_pred)
    return mse
end

function MdAPE(y_meas, y_pred) # Median Absolute Prediction Error: median(abs((C_observed - C_predicted)/C_predicted* 100 %))
    mdape = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdape = 100*median(abs.((y_meas[ind] .- y_pred[ind]) ./ y_pred[ind])) # %
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