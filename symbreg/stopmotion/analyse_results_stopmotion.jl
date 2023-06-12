# Histogram of results from jld file.
# Date: 230130

using Pkg
cd(@__DIR__)
cd("..")
# Pkg.activate("..")
# using JLD2 # requires version 0.4.28

using Flux
using BSON: @load

# include("../fcts/fcts_trainingdata.jl")
include("../fcts/fcts_createnn.jl")
include("../fcts/get_elevelddata.jl")
include("../fcts/fcts_train.jl")
include("../fcts/fcts_plotting.jl")
include("../fcts/fcts_eqreader.jl")
include("../fcts/fcts_predictionerrors.jl")

# seed = 1
# Random.seed!(seed)
x_all, y_all = get_elevelddata()
n_in = size(x_all[1].covariates, 1) # nbr of inputs to each network (number of covariates)

np = 1031
startp = 1
x = x_all[startp:startp+np-1]
y = y_all[startp:startp+np-1]

## open bson file
cd(@__DIR__)

i = 1

@load "bson/model$i" * "_inputpruning1.bson" model;
model1 = deepcopy(model);
@load "bson/model$i" * "_inputpruning2.bson" model;
model2 = deepcopy(model);
@load "bson/model$i" * "_inputpruning3.bson" model;
model3 = deepcopy(model);
@load "bson/model$i" * "_parampruning1.bson" model;
model4 = deepcopy(model);
@load "bson/model$i" * "_parampruning3.bson" model;
model5 = deepcopy(model);
@load "bson/model$i" * "_parampruning5.bson" model;
model6 = deepcopy(model);
@load "bson/model$i" * "_parampruning7.bson" model;
model7 = deepcopy(model);
@load "bson/model$i" * "_parampruning8.bson" model;
model8 = deepcopy(model);
@load "bson/nn_thread$i"*"loss_ALE.bson" model;
model_final = deepcopy(model);

# f = jldopen("../jld_mse_1000_1000_seed1234/nn_thread5.jld2", "r") # 4 & 5
# model_final = f["model"]
# close(f)


## studying one covariate only
cov = 2

# inputpruning 1 covariate
modelk10_1 = model1.nn[1].paths[cov]
modelk10_1[1].W_mask
modelk10_1[1].b_mask

# inputpruning 2 covariates
modelk10_2 = model2.nn[1].paths[cov]
modelk10_2[1].W_mask
modelk10_2[1].b_mask

modelk10_3 = model3.nn[1].paths[cov]
modelk10_3[1].W_mask
modelk10_3[1].b_mask
modelk10_3[3].W_mask
modelk10_3[3].b_mask
modelk10_3[5].W_mask
modelk10_3[5].b_mask

modelk10_4 = model4.nn[1].paths[cov]
modelk10_4[1].W_mask
modelk10_4[1].b_mask
modelk10_4[3].W_mask
modelk10_4[3].b_mask
modelk10_4[5].W_mask
modelk10_4[5].b_mask

modelk10_5 = model5.nn[1].paths[cov]
modelk10_5[1].W_mask
modelk10_5[1].b_mask
modelk10_5[3].W_mask
modelk10_5[3].b_mask
modelk10_5[5].W_mask
modelk10_5[5].b_mask

modelk10_6 = model6.nn[1].paths[cov]
modelk10_6[1].W_mask
modelk10_6[1].b_mask
modelk10_6[3].W_mask
modelk10_6[3].b_mask
modelk10_6[5].W_mask
modelk10_6[5].b_mask

modelk10_7 = model7.nn[1].paths[cov]
modelk10_7[1].W_mask
modelk10_7[1].b_mask
modelk10_7[3].W_mask
modelk10_7[3].b_mask
modelk10_7[5].W_mask
modelk10_7[5].b_mask

modelk10_final = model_final.nn[1].paths[cov]
modelk10_final[1].W_mask
modelk10_final[1].b_mask
modelk10_final[3].W_mask
modelk10_final[3].b_mask
modelk10_final[5].W_mask
modelk10_final[5].b_mask