from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 100
chunk_size = N // size

start = rank * chunk_size + 1
end = start + chunk_size - 1

partial_sum = 0
for i in range(start, end + 1):
    partial_sum += i

total_sum = comm.reduce(partial_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"[Rank 0] Total sum of 1 to {N} is {total_sum}")
