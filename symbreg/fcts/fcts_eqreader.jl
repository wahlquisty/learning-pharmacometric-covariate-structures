# Get equations in readable form from network structure.
# Date: 230105

using SymbolicUtils

"""
     layer2string(m:T_parallelnn, layer, input, [rounding=false])

Converts neural network to readable string. Is called for each layer on the network with the specific input (also treating activation functions as specific layers). There are no explicit checks on the size of the input.

Matches the following network structure
Unit, Multiplication, Power function (only in first layer), Division (only in second layer)
activation_1(x) = [x[1, :] (x[2, :] .* x[3, :]) abs.(x[4, :]) .^ (x[5, :])]'
activation_2(x) = [x[1, :] (x[2, :] .* x[3, :]) (x[4, :] ./ (x[5, :] .+ 1.0))]'
m = Chain(
    DenseWithMask(n_in, 5),     
    x -> activation_1(x),
    DenseWithMask(3, 5),
    x -> activation_2(x),
    DenseWithMask(3, 1, abs))  

# Arguments:
- `m`: Neural network
- `layer`: Int, which layer we are considering.
- `input`: Vector, input to this layer.
- `rounding`: Bool. If parameter values should be rounded for visibility. Default: rounding = false.

Returns a readable String from this layer. (In a vector if the layer has several outputs).
"""
function layer2string(m::T_subnn, layer, input, rounding=false)
    if layer in [1 3 5] # Dense layers
        W = m[layer].weight .* m[layer].W_mask  # Parameter value of weight with mask
        B = m[layer].bias .* m[layer].b_mask    # Parameter value of bias with mask

        if rounding # rounding for visibility
            W = round.(W, digits=2) # round for visibility
            B = round.(B, digits=2)
        end

        n_outputs = size(W, 1)
        n_inputs = size(W, 2)

        l_str = String[]
        for j = 1:n_outputs
            push!(l_str, "")
            for k = 1:n_inputs
                w0 = W[j, k]
                if w0 != 0 # weights
                    if w0 < 0
                        l_str[j] = "$(l_str[j])+($(w0))*($(input[k]))"
                    else
                        l_str[j] = "$(l_str[j])+$(w0)*($(input[k]))"
                    end
                end
            end
            b0 = B[j]
            if b0 != 0 # bias
                if b0 < 0
                    l_str[j] = "$(l_str[j])+($(b0))"
                else
                    l_str[j] = "$(l_str[j])+$(b0)"
                end
            end
            if layer == 5 # put abs on output
                l_str[j] = "(abs($(l_str[j])))"
            end
        end

    elseif layer == 2 || layer == 4 # Activation functions (here treated as specific layers)
        l_str = String[]
        for i in eachindex(input)
            if i == 3 || i == 5
                # do nothing
            elseif !isempty(input[i])
                if i == 1
                    push!(l_str, "$(input[i])")
                elseif i == 2
                    if isempty(input[i]) || isempty(input[i+1])
                        push!(l_str, "0")
                    else
                        push!(l_str, "($(input[i]))*($(input[i+1]))")
                    end
                elseif i == 4 && layer == 2 # First layer
                    if isempty(input[i])
                        push!(l_str, "0")
                    elseif isempty(input[i+1])
                        push!(l_str, "1")
                    else
                        push!(l_str, "(abs($(input[i])))^($(input[i+1]))")
                    end
                elseif i == 4 && layer == 4 # Second layer
                    if isempty(input[i])
                        push!(l_str, "0")
                    else
                        push!(l_str, "($(input[i]))/($(input[i+1]) + 1)")
                    end
                end
            else
                push!(l_str, "0") # fixme for better solution to not have *0 in some cases
            end
        end
    end
    return l_str
end


"""
    get_fctsfrommodel(model)

Converts Flux model to callable functions, one for each subnetwork.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.

Returns Vector{Function} with callable functions.
"""
function get_fctsfrommodel(parallelnn::T_parallelnn)
    eqout = get_outputeqreader(parallelnn)
    n_networks = length(parallelnn[1].paths)
    func_exp = Vector{Function}(undef, n_networks)
    for i = 1:n_networks
        func_exp[i] = eval(Meta.parse("(x_age,x_wgt,x_bmi,x_gdr,x_AV) -> " * eqout[i]))
    end
    return func_exp
end

"""
    get_readableexpr(model)

Converts Flux model to readable expressions and prints them.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.

Returns Vector{expression} and prints readable expressions.
"""
function get_readableexpr(parallelnn::T_parallelnn, rounding=false)
    n_networks = length(parallelnn[1].paths)
    expr = Vector{Any}(undef, n_networks)
    eqout = get_outputeqreader(parallelnn, rounding)
    for i = 1:n_networks
        expression_simple = expand(eval(Meta.parse(eqout[i])))
        expr[i] = expression_simple
    end
    print("k10 (all scaled) = \n", expr[1], "\n", "k12 (all scaled) = \n", expr[2], "\n", "k13 (all scaled) = \n", expr[3], "\n", "k21 (all scaled) = \n", expr[4], "\n", "k31 (all scaled) = \n", expr[5], "\n", "V1 (all scaled) = \n", expr[6], "\n")
    return expr
end


"""
    get_outputeqreader(model, rounding=false)

Function that calls layer2string for each layer in our current network structure. Assumes four inputs.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.

Returns Vector{Function} with callable functions.
"""
function get_outputeqreader(parallelnn::T_parallelnn, rounding=false)
    input = ["x_age"; "x_wgt"; "x_bmi"; "x_gdr"; "x_AV"]
    chain = parallelnn[1].paths
    eqout = Vector{String}(undef, length(chain))

    for i in eachindex(chain)
        m = chain[i]
        Y1 = layer2string(m, 1, input, rounding)
        Y2 = layer2string(m, 2, Y1, rounding)
        Y3 = layer2string(m, 3, Y2, rounding)
        Y4 = layer2string(m, 4, Y3, rounding)
        Y = layer2string(m, 5, Y4, rounding)[1]
        eqout[i] = Y
    end
    return eqout
end
