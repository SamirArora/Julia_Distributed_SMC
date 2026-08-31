using Distributed


# number of worker processes
const NUM_PROCS = 8
nw = nprocs() - 1
if nw < NUM_PROCS
    addprocs(NUM_PROCS - nw)
elseif nw > NUM_PROCS
    rmprocs(workers()[NUM_PROCS+1:end])
end
@assert nprocs() - 1 == NUM_PROCS "have $(nprocs()-1), want $NUM_PROCS"


@everywhere using Random
@everywhere include("distributed_resample.jl")

# other global variables
@everywhere const N = 800000
const block_size = N ÷ NUM_PROCS
@everywhere const dim = 100
const seed = 1
n_high = floor(Int, N * 0.02)
n_medium = floor(Int, N * 0.1)
n_low = N - n_high - n_medium
weights = vcat(fill(0.02, n_high), fill(0.01, n_medium), fill(0.0033, n_low))
log_weights = log.(weights)

@everywhere const LOCAL_BLOCK = Ref{Block}()

@everywhere function initialize_block!(block_idx, block_size, dim, seed, log_weights)
    rng = Xoshiro(seed + block_idx)
    states = rand(rng, dim, block_size)
    LOCAL_BLOCK[] = Block(block_size, states, log_weights)
end

# WARM-UP run ........
foreach(fetch, [remotecall(initialize_block!, w, k, block_size, dim, seed,
                           log_weights[(k-1)*block_size+1 : k*block_size])
                for (k, w) in enumerate(workers())])
distributed_resample!(rand(Xoshiro(seed)))


# MAIN RUN ....................
foreach(fetch, [remotecall(initialize_block!, w, k, block_size, dim, seed,
                           log_weights[(k-1)*block_size+1 : k*block_size])
                for (k, w) in enumerate(workers())])
old_particles = gather_states()

# calling distributed implementation of resample  
rng =Xoshiro(seed)
u = rand(rng)
@time distributed_resample!(u)
new_particles_distributed = gather_states()

# calling serial implementation of resample for comparison.
new_particles_serially = systematic_resampling(old_particles, log_weights, u, N)

# assertions to confirm two methods give same answer.
println("match: ", new_particles_distributed == new_particles_serially)
println("change: ", new_particles_distributed != old_particles)


