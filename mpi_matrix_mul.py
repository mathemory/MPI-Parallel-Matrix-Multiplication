import argparse
from typing import Tuple

import numpy as np
from mpi4py import MPI


DTYPE_MAP = {
    "float32": (np.float32, MPI.FLOAT),
    "float64": (np.float64, MPI.DOUBLE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPI parallel matrix multiplication with row-wise decomposition."
    )
    parser.add_argument("--m", type=int, required=True, help="Rows of matrix A.")
    parser.add_argument("--k", type=int, required=True, help="Columns of A / rows of B.")
    parser.add_argument("--n", type=int, required=True, help="Columns of matrix B.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for matrix generation.")
    parser.add_argument(
        "--input-mode",
        choices=("random", "manual"),
        default="random",
        help="Use randomly generated matrices or manually provided matrices.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="float64",
        help="Floating-point data type used for matrices.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the parallel result against a serial NumPy implementation.",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print matrices for small test cases.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra per-rank debugging information.",
    )
    parser.add_argument(
        "--a-values",
        type=str,
        default=None,
        help="Manual matrix A values in row-major order, separated by commas or spaces.",
    )
    parser.add_argument(
        "--b-values",
        type=str,
        default=None,
        help="Manual matrix B values in row-major order, separated by commas or spaces.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("m", "k", "n"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if args.input_mode == "random" and (args.a_values or args.b_values):
        raise ValueError("--a-values and --b-values can only be used with --input-mode manual.")


def build_row_partition(total_rows: int, world_size: int) -> Tuple[np.ndarray, np.ndarray]:
    rows_per_rank = np.full(world_size, total_rows // world_size, dtype=np.int32)
    rows_per_rank[: total_rows % world_size] += 1

    row_offsets = np.zeros(world_size, dtype=np.int32)
    if world_size > 1:
        row_offsets[1:] = np.cumsum(rows_per_rank[:-1], dtype=np.int32)

    return rows_per_rank, row_offsets


def make_random_matrix(rows: int, cols: int, dtype: np.dtype, rng: np.random.Generator) -> np.ndarray:
    return rng.random((rows, cols), dtype=dtype)


def parse_flat_values(raw_values: str, expected_count: int, name: str, dtype: np.dtype) -> np.ndarray:
    normalized = raw_values.replace(",", " ")
    tokens = [token for token in normalized.split() if token]
    if len(tokens) != expected_count:
        raise ValueError(
            f"{name} expects {expected_count} values, but received {len(tokens)} values."
        )
    return np.array([float(token) for token in tokens], dtype=dtype)


def prompt_matrix(rows: int, cols: int, name: str, dtype: np.dtype) -> np.ndarray:
    print(f"Please input matrix {name} row by row. Each row should contain {cols} numbers.")
    values = []
    for row_idx in range(rows):
        while True:
            row_input = input(f"{name} row {row_idx}: ").strip()
            normalized = row_input.replace(",", " ")
            tokens = [token for token in normalized.split() if token]
            if len(tokens) != cols:
                print(f"Row {row_idx} of {name} must contain exactly {cols} numbers. Please try again.")
                continue
            try:
                values.extend(float(token) for token in tokens)
                break
            except ValueError:
                print(f"Row {row_idx} of {name} contains invalid numbers. Please try again.")
    return np.array(values, dtype=dtype).reshape(rows, cols)


def build_input_matrices(args: argparse.Namespace, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    if args.input_mode == "random":
        rng = np.random.default_rng(args.seed)
        matrix_a = make_random_matrix(args.m, args.k, dtype, rng)
        matrix_b = make_random_matrix(args.k, args.n, dtype, rng)
        return matrix_a, matrix_b

    if args.a_values is not None and args.b_values is not None:
        matrix_a = parse_flat_values(args.a_values, args.m * args.k, "matrix A", dtype).reshape(args.m, args.k)
        matrix_b = parse_flat_values(args.b_values, args.k * args.n, "matrix B", dtype).reshape(args.k, args.n)
        return matrix_a, matrix_b

    print("Manual input mode selected. You can type matrix values directly.")
    matrix_a = prompt_matrix(args.m, args.k, "A", dtype)
    matrix_b = prompt_matrix(args.k, args.n, "B", dtype)
    return matrix_a, matrix_b


def serial_matrix_multiply(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    return matrix_a @ matrix_b


def main() -> None:
    args = parse_args()
    validate_args(args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    np_dtype, mpi_dtype = DTYPE_MAP[args.dtype]

    rows_per_rank = None
    row_offsets = None
    matrix_a = None
    matrix_b = None
    matrix_c = None

    if rank == 0:
        matrix_a, matrix_b = build_input_matrices(args, np_dtype)
        rows_per_rank, row_offsets = build_row_partition(args.m, size)
    else:
        matrix_b = np.empty((args.k, args.n), dtype=np_dtype)

    rows_per_rank = comm.bcast(rows_per_rank, root=0)
    row_offsets = comm.bcast(row_offsets, root=0)

    local_rows = int(rows_per_rank[rank])
    local_a = np.empty((local_rows, args.k), dtype=np_dtype)

    sendcounts_a = (rows_per_rank * args.k).astype(np.int32)
    displs_a = (row_offsets * args.k).astype(np.int32)

    sendbuf_a = None
    if rank == 0:
        sendbuf_a = [matrix_a.ravel(), sendcounts_a, displs_a, mpi_dtype]

    comm.Scatterv(sendbuf_a, local_a.ravel(), root=0)
    comm.Bcast(matrix_b, root=0)

    if args.debug:
        if local_rows == 0:
            print(f"[Rank {rank}] assigned no rows.")
        else:
            start_row = int(row_offsets[rank])
            end_row = start_row + local_rows - 1
            print(f"[Rank {rank}] rows {start_row} to {end_row}")

    comm.Barrier()
    start_time = MPI.Wtime()

    local_c = local_a @ matrix_b

    recvcounts_c = (rows_per_rank * args.n).astype(np.int32)
    displs_c = (row_offsets * args.n).astype(np.int32)

    recvbuf_c = None
    if rank == 0:
        matrix_c = np.empty((args.m, args.n), dtype=np_dtype)
        recvbuf_c = [matrix_c.ravel(), recvcounts_c, displs_c, mpi_dtype]

    comm.Gatherv(local_c.ravel(), recvbuf_c, root=0)

    comm.Barrier()
    elapsed = MPI.Wtime() - start_time
    max_elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)

    if rank != 0:
        return

    print("=== MPI Matrix Multiplication ===")
    print(f"Process count: {size}")
    print(f"Matrix dimensions: A({args.m} x {args.k}), B({args.k} x {args.n}), C({args.m} x {args.n})")
    print(f"Input mode: {args.input_mode}")
    print(f"Data type: {args.dtype}")
    print("Row assignment:")
    for proc in range(size):
        proc_rows = int(rows_per_rank[proc])
        if proc_rows == 0:
            print(f"  Rank {proc}: no rows assigned")
        else:
            start_row = int(row_offsets[proc])
            end_row = start_row + proc_rows - 1
            print(f"  Rank {proc}: rows {start_row} to {end_row} ({proc_rows} rows)")
    print(f"Parallel elapsed time: {max_elapsed:.6f} s")

    if args.print_matrix:
        print("Matrix A:")
        print(matrix_a)
        print("Matrix B:")
        print(matrix_b)
        print("Matrix C:")
        print(matrix_c)

    if args.check:
        reference_c = serial_matrix_multiply(matrix_a, matrix_b)
        max_error = float(np.max(np.abs(matrix_c - reference_c))) if matrix_c.size else 0.0
        if np.allclose(matrix_c, reference_c, rtol=1e-6, atol=1e-8):
            print(f"Correctness check: PASSED (max abs error = {max_error:.6e})")
        else:
            print(f"Correctness check: FAILED (max abs error = {max_error:.6e})")
            raise SystemExit(1)
    else:
        print("Correctness check: skipped")


if __name__ == "__main__":
    main()
