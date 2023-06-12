# Mamillary three-compartment model functions
# Computes eigenvalues and R column vector of the three compartment mammillary model.

using StaticArrays, LinearAlgebra

# Initiate/update parameters
function PK3(θ)
    V1inv = 1 / θ[end]
    λ = getλ_threecomp(θ)
    λinv = 1 ./ λ
    R = getR_threecomp(θ, λ)
    return V1inv, λ, λinv, R
end

# Compute eigenvalues λ for 3 compartment mammillary model
@inline @fastmath function getλ_threecomp(θ::AbstractVector{T}) where {T}
    k10, k12, k13, k21, k31, _ = θ
    b1 = k10 + k12 + k13 + k21 + k31
    b2 = k21 * (k10 + k13 + k31) + k31 * (k10 + k12)
    b3 = k10 * k21 * k31

    # Wengert list used to compute λ.
    a1 = b1 / 3
    a2 = a1^2
    a3 = a1 * a2
    a4 = b2 / 3
    a5 = a4 - a2
    a6 = (b1 * a4 - b3) / 2
    a7 = 2(a6 + sqrt(complex(a5^3 + (a3 - a6)^2)) - a3)^(1 / T(3))
    a8 = -real(a7)
    a9 = imag(a7)
    a10 = a9 * sqrt(T(3)) / 2
    a11 = a1 - a8 / 2

    return [-a1 - a8, -a10 - a11, a10 - a11] # The eigenvalues of the continuous-time system matrix
end

# Computation of R, nominators of the first-order systems
@inline function getR_threecomp(θ, λ)
    _, _, _, k21, k31, _ = θ
    l1, l2, l3 = λ
    a1 = l2 - l1
    a2 = l3 - l1
    a3 = l3 - l2
    d1 = a2 * a1
    d2 = -a3 * a1
    d3 = a2 * a3
    d1inv = 1 / d1
    d2inv = 1 / d2
    d3inv = 1 / d3
    Qinv = [(l1*d1inv)*l1 l1*d1inv d1inv; (l2*d2inv)*l2 l2*d2inv d2inv; (l3*d3inv)*l3 l3*d3inv d3inv]
    b = [1, k21 + k31, k21 * k31] # Quite often we would only be interested in the first column (first output). See paper for computations.
    return Qinv * b
end


# We are only looking for the concentration of the central compartment volume

# using StaticArrays, SLEEFPirates

####

# update state for bolus
@inline function bolus(x, v)
    return x .+ v
end

# update state with input step
@inline function step(x, λinv, Φdiag, u)
    Γdiag = @. λinv * (Φdiag - 1)
    return @. Φdiag * x + Γdiag * u
end

# output computation
function gety(x, V1inv, R)
    return (V1inv*R'*x)[1]
end

# Diagonal of discrete time system matrix
@inline @fastmath function getΦdiag(λ, h)
    Φdiag = @. exp(λ * h) # cannot differentiate through VectorizationBase.vexp
    # return @. exp(λ * h) 
    return Φdiag
end

# Initiate/update state
@inline function updatestate(x::AbstractVector{T}, h, λ, λinv, u=zero(T), v=zero(T)) where {T}
    Φdiag = getΦdiag(λ, h) # compute Φ
    x = bolus(x, v) # update state for bolus
    x = step(x, λinv, Φdiag, u) # infusion affect next sample
    return x
end

# Update state and compute output
@inline function updatestateoutput(x::AbstractVector{T}, h, V1inv, λ, λinv, R, u=zero(T), v=zero(T)) where {T}
    Φdiag = getΦdiag(λ, h) # compute Φ
    x = bolus(x, v) # update state for bolus
    y = gety(x, V1inv, R) # compute output
    x = step(x, λinv, Φdiag, u) # infusion affect next sample
    return x, y
end


"""
    pk3sim!(y, θ, u, v, hs, youts)
Fast simulation of the three compartment mammillary PK model.

The parameter vector θ has the following structure
```
θ = [k10, k12, k13, k21, k31, V1]
```
# Arguments:
- `y`: Preallocated output vector of size length(youts)
- `θ`: Parameter vector, see above.
- `u`: Infusion rate vector of size length(hs)
- `v`: Bolus dose vector of size length(hs)
- `hs`: Step size, should have the size of [diff(time) diff(time)[end]] where time is the matching time vector to u, v
- `youts`: Indices for output observations, corresponding to times in hs

Updates `y` with simulated outputs `x_1` at time instances `youts`.
"""
function pk3sim(θ, u, v, hs, youts)
    y = zeros(eltype(Real), length(youts))
    V1inv, λ, λinv, R = PK3(θ)
    j = 1 # counter to keep track of next free spot in y
    x = zeros(eltype(u), 3) # initial state
    for i in eachindex(u, hs, v)
        if i in youts # if we want to compute output
            x, yi = @inbounds updatestateoutput(x, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            y[j] = yi
            j += 1
        else
            x = @inbounds updatestate(x, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return y
end