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

if rank == 0:
    total = partial_sum
    for src in range(1, size):
        recv_sum = comm.recv(source=src, tag=src)
        total += recv_sum
    print(f"[Rank 0] Total sum of 1 to {N} is {total}")
else:
    comm.send(partial_sum, dest=0, tag=rank)
