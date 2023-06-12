# Learning pharmacometric covariate model structures with symbolic regression networks

This repo contains code to replicate the results and figures in the paper "Learning pharmacometric covariate model structures with symbolic regression networks".

We present a novel methodology for automatic and simultaneous covariate model structure discovery and parameter optimization. This model is demonstrated using an example of the anesthetic drug propofol. The final model outperforms state-of-the-art modeling, in that it finds expressions that match data slightly better, while relying on notably fewer covariates.


## Code examples

### Get started
Start by cloning the repo
```
git clone https://github.com/wahlquisty/learning-pharmacometric-covariate-structures.git
```

Install julia and instantiate the environment by running `julia` and in the Julia REPL, run:
```julia 
using Pkg
Pkg.instantiate(".")
```

### Main results - Covariate model from symbolic regression

To get the main results of the paper, run in the Julia REPL:

```julia
include("symbreg/SymbRegEleveld.jl")
main(x,y,1,opt,"ALE",seed,1000,1000) # This saves the final model in a BSON-file
include("checkresults/get_results.jl") # Get the model from the BSON file and get the final prediction errors
```
This outputs the prediction errors and covariate expressions of the final covariate model.


The inputs and ouputs of the `main` function is given by the following
```
    main(x, y, nthreads, opt, error_fct, seed, nepochs_init, nepochs_prune)

Main function for training and pruning of several symbolic regression regression networks on training data.

Creates nthreads symbolic regression networks and performs pruning of each. The computing is done in parallel (only if julia is started with several threads) using Threads.@threads. If the pruning was successful, the resulting neural networks are written to file (type bson) in folder "bson/".

# Arguments:
- `x`: Vector{InputData}. Training x data with length(x) == nbr of patients.
- `y`: Vector{Vector{Number}}. Training y data with length(y) == nbr of patients.
- `nthreads`: Int, number of threads to run.
- `opt`: Optimizer object, for example Flux.Adam(0.0001).
- `error_fct`: Function, which loss function to use.
- `seed`: Int. Seed for reproducibility.
- `nepochs_init`: Int. Nbr of epochs to train before start of pruning and during inputpruning.
- `nepochs_prune`: Int. Nbr of epochs to train between parameter pruning steps.

Writes to file for each thread "bson/nn_threadi.bson" with name `model`.
```


### Models for comparison
In the paper, we use two non-covariate models for comparison to the symbolic regression model and the Eleveld model. These are: a PK model constant over the population and individual PK models over all individuals.

#### Constant PK model
To optimize the constant PK model and get the resulting model and prediction errors, run

```julia
include("optim/get_results_constantpk.jl")
```

#### Individual PK models
Similarly, to optimize individual PK models and get the resulting prediction errors, run

```julia
include("optim/get_results_individualpk.jl")
```

## Figures

The tikz code to the figures of the paper is found in `tikz/`. Note that the scatter plot figure `tikz/scatterplot/scatterplot_symbreg_eleveld_mdale.tex` has to be compiled with `lualatex` due to limited memory capacity in `pdflatex`.

## Background 

### Data set
In this paper, we use a data set for the anesthetic drug propofol with data from 1,031 individuals. The data set can be found in `symbreg/data/dataframe.csv`, originally published in Eleveld et al. (2018).

### Three compartment model
The three compartment model is described by the following differential equations

$$ \dfrac{dx_1}{dt} = - (k_{10} + k_{12} + k_{13}) x_1 + k_{21} x_2 + k_{31} x_3 + \dfrac{1}{V_1} u $$

$$ \dfrac{dx_2}{dt} = k_{12} x_1 - k_{21} x_2 $$

$$ \dfrac{dx_3}{dt} = k_{13} x_1 - k_{31} x_3, $$

where
- $x_i(t)$ is the ith state
- $k_{ij}$ is the transfer rate from compartment $j$ to $i$.
- $u(t)$ is the input
- $V_1$ is the central compartment volume.

## License

This project is licensed under the [MIT License](LICENSE).


