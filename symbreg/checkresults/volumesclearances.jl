# Check max & min for Eleveld and my model for volumes and clearances
# Date: 230601

# Order: AGE,WGT,BMI,GDR,AV
# Order: [k10,k12,k13,k21,k31,V1]

# covariate_normalization = [88.0, 160.0, 52.84713965]
#AGE (0->1), WGT (0->1), GDR (male -0.5, female 0.5), AV (arterial -0.5, venous 0.5)

# x hold all patient information
pkparams_normalization = [0.008468822, 0.013328808, 0.00356759, 0.0057506226, 0.000102905105, 9440.525] # Normalization factor for k10, k12 etc

# wgt = 80
# f1 = func_exp[1]
# f1(0,1,0,0,0,0)
k10_symreg = zeros(1031)
k12_symreg = zeros(1031)
k13_symreg = zeros(1031)
k21_symreg = zeros(1031)
k31_symreg = zeros(1031)
V1_symreg = zeros(1031)
for pat = 1:1031
    cov = x[pat].covariates
    k10_symreg[pat] = func_exp_ALE[1](cov...)*pkparams_normalization[1]*60 # These fcts are scaled! (input between 0 and 1)
    k12_symreg[pat] = func_exp_ALE[2](cov...) * pkparams_normalization[2]*60
    k13_symreg[pat] = func_exp_ALE[3](cov...) * pkparams_normalization[3]*60
    k21_symreg[pat] = func_exp_ALE[4](cov...) * pkparams_normalization[4]*60
    k31_symreg[pat] = func_exp_ALE[5](cov...) * pkparams_normalization[5]*60
    V1_symreg[pat] = func_exp_ALE[6](cov...) * pkparams_normalization[6]/1000
end

CL_symreg = k10_symreg.*V1_symreg
Q2_symreg = k12_symreg.*V1_symreg
Q3_symreg = k13_symreg.*V1_symreg
V2_symreg = k12_symreg./k21_symreg
V3_symreg = k13_symreg./k31_symreg

print("Symbreg compartment volumes and clearances \n")
print("Min V1: ", minimum(V1_symreg), "\nMax V1: ", maximum(V1_symreg), '\n')
print("Min V2: ", minimum(V2_symreg), "\nMax V2: ", maximum(V2_symreg), '\n')
print("Min V3: ", minimum(V3_symreg), "\nMax V3: ", maximum(V3_symreg), '\n')
print("Min CL: ", minimum(CL_symreg), "\nMax CL: ", maximum(CL_symreg), '\n')
print("Min Q2: ", minimum(Q2_symreg), "\nMax Q2: ", maximum(Q2_symreg), '\n')
print("Min Q3: ", minimum(Q3_symreg), "\nMax Q3: ", maximum(Q3_symreg), '\n')

# Get eleveld volumes and clearances
modeldf = CSV.read("data/eleveld_modelparams.csv", DataFrame)    # Model parameters V1,V2,... for PK model

print("Eleveld compartment volumes and clearances \n")
print("Min V1: ",minimum(modeldf.V1), "\nMax V1: ",maximum(modeldf.V1),'\n')
print("Min V2: ",minimum(modeldf.V2), "\nMax V2: ",maximum(modeldf.V2),'\n')
print("Min V3: ",minimum(modeldf.V3), "\nMax V3: ",maximum(modeldf.V3),'\n')
print("Min CL: ",minimum(modeldf.CL), "\nMax CL: ",maximum(modeldf.CL),'\n')
print("Min Q2: ",minimum(modeldf.Q2), "\nMax Q2: ",maximum(modeldf.Q2),'\n')
print("Min Q3: ",minimum(modeldf.Q3), "\nMax Q3: ",maximum(modeldf.Q3),'\n')
