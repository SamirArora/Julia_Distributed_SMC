using Distributed

@everywhere mutable struct Block 
  size::Int
  states::Matrix{Float64}
  log_weights::Vector{Float64}
  cdf::Vector{Float64}
  Block(size, states, lw) = new(size, states, lw, Float64[])
end


function distributed_resample!(u)
    # computing per-block cumulative cdfs
    totals = [fetch(@spawnat w compute_local_cumsum!()) for w in workers()]
    block_upper = cumsum(totals)
    Z = block_upper[end]
    block_lower = [0.0; block_upper[1:end-1]]
    step = Z / N

    # computing k_min and k_max for each block
    k_min = [ceil(Int, block_lower[b]/step - u) + 1 for b in 1:NUM_PROCS]
    k_max = [ceil(Int, block_upper[b]/step - u) + 1 for b in 1:NUM_PROCS]

    futures = [@spawnat w block_resample!(k_min[b], k_max[b], block_lower[b], u, step) for (b, w) in enumerate(workers())]
    foreach(fetch, futures)
    return nothing
end

@everywhere function compute_local_cumsum!()
    b = LOCAL_BLOCK[]
    b.cdf = cumsum(exp.(b.log_weights))   
    return b.cdf[end]                     
end

@everywhere function block_resample!(k_min, k_max, block_lower, u, step)
    b = LOCAL_BLOCK[]
    n_out = k_max - k_min
    new_states = Matrix{Float64}(undef, dim, n_out)

    a = 1
    out = 1
    for k in k_min:k_max-1
        u_k = (u + (k-1)) * step - block_lower
        while b.cdf[a] < u_k && a < b.size
            a += 1
        end
        @views new_states[:, out] .= b.states[:, a]
        out += 1
    end
    b.states = new_states
    b.size = n_out
    b.log_weights = log.(fill(1/N, b.size))
    return nothing
end

# function to perform systematic resampling
function systematic_resampling(particles, log_weights, u, N)
    Z = sum(exp.(log_weights))
    step = Z/N
    cdf = cumsum(exp.(log_weights))
    new_states = similar(particles)

    j = 1
    for k in 1:N 
        target = u*step + (k-1)*step 
        while cdf[j] < target && j < N 
            j+=1 
        end
        new_states[:,k] = particles[:,j]
    end
    return new_states
end


# function to gather states.
function gather_states()
    parts = [fetch(@spawnat w LOCAL_BLOCK[].states) for w in workers()]
    return reduce(hcat, parts)          
end
