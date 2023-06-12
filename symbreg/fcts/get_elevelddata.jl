# Get patient data for all patients in the Eleveld model set

# Date: 230109

# Pkg.add(url="https://github.com/wahlquisty/FastPKSim.jl") # FastPKSim.jl from github
# using FastPKSim

# using Distributions, Random
using CSV, DataFrames, StaticArrays



##############

"""
    get_elevelddata()

Get patient data from Eleveld data set (1031 patients).

Stores data in x::Vector{InputData} structure and y::Vector{Float32}.
InputData is a struct that holds covariates (scaled), u (infusion rate), v (bolus dose), time (time vector in seconds), hs (time differences between two rate changes), youts (observed indices), normalization (scaling for PK parameters)
y contains all observations for that patient.
    
# Arguments:

Returns the data in x::Vector{InputData} structure and y::Vector{Float32}.
"""
function get_elevelddata()
    # Load datafiles
    modeldf = CSV.read("data/eleveld_modelparams.csv", DataFrame)    # Model parameters V1,V2,... for PK model
    inputdf = CSV.read("data/eleveld_infusiondata.csv", DataFrame)   # Input vector for simulation, with time and rates/ doses for infusions and boluses and if a time instance was a measurement
    df = CSV.read("data/dataframe.csv", DataFrame) # covariate data

    np = 1033 # nbr of patients
    y = Vector{Vector{Float32}}(undef, np)  # To save measurements
    x = Vector{InputData}(undef, np)        # To save input data: covariates, infusion rates, time vector, measurements and scaling

    input_normalization = [maximum(df.AGE), maximum(df.WGT), maximum(df.BMI)] #[88.0, 160.0,] # Normalization factor for AGE and WGT (max value in dataset)
    pkparams_normalization = [0.008468822, 0.013328808, 0.00356759, 0.0057506226, 0.000102905105, 9440.525] # Normalization factor for k10, k12 etc

    nstudy = 30 # nbr of studies
    for studynbr = 1:nstudy
        study_df, firstid, lastid = get_studydata(modeldf, studynbr) # get dataframe for this study
        for id = firstid:lastid
            if id in (893, 897) # no measurements exists for these patients
                continue
            end
            _,infusionrate, bolusdose, t, hs, youts = get_patientdata(id, modeldf, inputdf) # get patient data
            age, wgt, hgt, m1f2, a1v2, noadd1add2 = get_covariates(df, id) # get covariates from dataframe
            bmi = wgt/((hgt/100)^2) # bmi [kg/m²]
            u = infusionrate # [ug/min]
            v = bolusdose # [ug]

            continuous_inputs = Float32.([age, wgt, bmi])
            covariates = [continuous_inputs[1] / input_normalization[1], continuous_inputs[2] / input_normalization[2], continuous_inputs[3] / input_normalization[3], m1f2 - 1.5, a1v2 - 1.5] # scaled input variables, AGE (0->1), WGT (0->1), GDR (male -0.5, female 0.5), AV (arterial -0.5, venous 0.5)

            # Save patient data
            x[id] = InputData(covariates, u, v, t, hs, youts, pkparams_normalization)
            y[id] = get_ymeas(df, id) # Concentration observations
        end
    end

    # remove patients without measurements (patient 893 and 897 in the data set)
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
function get_studydata(modeldf, studynbr)
    study_df = modeldf[in([studynbr]).(modeldf.StudyNbr), :] # get all rows with nbr = studynbr
    idnbrs = study_df[!, :ID] # get array with all id-numbers in current study
    last = idnbrs[length(idnbrs)] # last id-number of current study
    first = idnbrs[1] # first id-number of current study
    return study_df, first, last
end

function get_indmeas(ismeasurements)
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

function get_patientdata(id, modeldf, inputdf)
    subject = modeldf[in([id]).(modeldf.ID), :] # get all rows with nbr = id

    # Volumes and clearances from Eleveld model
    V1 = subject[!, :V1][1] # [l]
    V2 = subject[!, :V2][1] # [l]
    V3 = subject[!, :V3][1] # [l]
    CL = subject[!, :CL][1] # [l/min]
    Q2 = subject[!, :Q2][1] # [l/min]
    Q3 = subject[!, :Q3][1] # [l/min]

    Ω = @SVector [V1, V2, V3, CL, Q2, Q3] # PK model parameter vector (in volumes and clearances)

    # Input data
    inputs = inputdf[in([id]).(inputdf.ID), :]

    time = Float32.(inputs[!, :Time]) # [s]
    infusionrate = Float32.(inputs[!, :InfusionRate]) # [ug/min]
    bolusdose = Float32.(inputs[!, :Bolus]) # [ug]
    ismeas = Float32.(inputs[!, :IsMeasurement]) # 1 = yes, 0 = no

    h = diff(time) # durations between inputs
    h = [h; h[end]] # assume last h is same as second last.
    # Indices of measured concentrations
    youts = get_indmeas(ismeas)
    # time = time[youts]
    # return infusionrate, bolusdose, time, h, youts
    return Float32.(Ω), infusionrate, bolusdose, time, h, youts

end

function get_covariates(df, id)
    subject = df[in([id]).(df.ID), :] # get all rows with nbr = id

    AGE = subject[!, :AGE][1]
    WGT = subject[!, :WGT][1]
    HGT = subject[!, :HT][1]
    M1F2 = subject[!, :M1F2][1]
    A1V2 = subject[!, :A1V2][1]
    NOADD1ADD2 = subject[!, :NOADD1ADD2][1]

    return AGE, WGT, HGT, M1F2, A1V2, NOADD1ADD2
end

function get_ymeas(df, id)
    subject = df[in([id]).(df.ID), :] # get all rows with nbr = id

    CP = subject[!, :CP]
    true_conc = CP[findall(x -> x > 0, CP)]
    y_meas = Float32.(true_conc)
    return y_meas
end

############### Get outputs from Eleveld model and evaluate the cost of the Eleveld model in terms of prediction errors, for comparison
# Date: 230118

# using CSV, DataFrames, StaticArrays

# include("get_elevelddata.jl") # Functions to get input data from files (model parameters and input data)
include("fcts_analyseresults.jl")

function get_predictions_eleveld()

    modeldf = CSV.read("data/eleveld_modelparams.csv", DataFrame) # Model parameters V1,V2,... for PK model
    inputdf = CSV.read("data/eleveld_infusiondata.csv", DataFrame) # Input vector for simulation, with time and rates/ doses for infusions and boluses and if a time instance was a measurement
    df = CSV.read("data/dataframe.csv", DataFrame) # covariate data

    y = Vector{Vector{Float32}}(undef, 1033)
    Ω = Vector{Vector{Float32}}(undef, 1033)
    x_reduced = Vector{InputData}(undef, 1033)

    nstudy = 30 # nbr of studies
    # studynbr = 1
    for studynbr = 1:nstudy
        study_df, firstid, lastid = get_studydata(modeldf, studynbr) # get dataframe for this study
        # id = 1
        for id = firstid:lastid
            if id in (893, 897) # no measurements ex_reducedists for these patients
                continue
            end
            Ωi, infusionrate, bolusdose, t, hs, youts = get_patientdata(id, modeldf, inputdf) # get patient data
            age, wgt, hgt, m1f2, a1v2, noadd1add2 = get_covariates(df, id)
            u = infusionrate# ug/min             / 1000 # mg/min
            v = bolusdose # ug             /1000 # mg
            covariates = Float32.([age, wgt])

            y_meas = get_ymeas(df, id)
            y[id] = y_meas
            Ω[id] = Ωi
            # y[id] = θ

            x_reduced[id] = InputData(covariates, u, v, t, hs, youts, ones(Float32, 6))
        end
    end

    # remove those without measurements
    deleteat!(y, 897)
    deleteat!(y, 893)
    deleteat!(Ω, 897)
    deleteat!(Ω, 893)
    deleteat!(x_reduced, 897)
    deleteat!(x_reduced, 893)

    # Eleveld predictions
    y_eleveld = similar(y)
    tobs = similar(y)
    for i in eachindex(y)
        V1, V2, V3, CL, Q2, Q3 = Ω[i]

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
    return Ω, y_eleveld, y
end


# ## Losses for comparison (average)
# print("Costs for Eleveldmodel on data set: \n")
# print("mean(MSE): ", mean(MSE.(y, y_eleveld)), "\n") # 3.46
# print("mean(MdAPE): ", mean(MdAPE.(y, y_eleveld)), "\n") # 34.78
# print("mean(MdPE): ", mean(MdPE.(y, y_eleveld)), "\n") # 14.4
# print("mean(MdALE): ", mean(MdALE.(y, y_eleveld)), "\n") # 0.3245
# print("mean(MdLE): ", mean(MdLE.(y, y_eleveld)), "\n") # 0.0791
