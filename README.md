# MPI Parallel Matrix Multiplication

本实验使用 `Python + Microsoft MPI + mpi4py` 实现矩阵乘法 `C = A x B` 的并行计算，采用按行划分的并行策略。

## 1. 环境准备

先确认本机已经安装 Python 和 Microsoft MPI：

```powershell
python --version
mpiexec -help
```

安装 `mpi4py`：

```powershell
python -m pip install mpi4py
```

用老师提供的样例验证 MPI 环境：

```powershell
mpiexec -n 4 python mpi_example\firstmpi.py
```

## 2. 程序说明

主程序为 `mpi_matrix_mul.py`。

- `rank 0` 负责准备矩阵 `A(m x k)` 和 `B(k x n)`
- 支持两种输入方式：
  - `random`：默认随机生成矩阵，适合性能测试
  - `manual`：手动输入矩阵，适合展示“给定矩阵”的实验过程
- 矩阵 `A` 按行分块发送给各进程
- 矩阵 `B` 广播给所有进程
- 各进程完成局部乘法后，由 `rank 0` 汇总得到完整矩阵 `C`

程序支持行数不能整除进程数的情况，也支持 `m < 进程数` 的情况。

## 3. 参数说明

```text
--m            矩阵 A 的行数
--k            矩阵 A 的列数，同时也是矩阵 B 的行数
--n            矩阵 B 的列数
--seed         随机种子，默认 42
--input-mode   输入模式，可选 random / manual，默认 random
--dtype        数据类型，可选 float32 / float64，默认 float64
--check        启用串行结果校验
--print-matrix 打印矩阵内容，适合小规模测试
--debug        输出每个 rank 的调试信息
--a-values     手动模式下矩阵 A 的元素，按行优先输入
--b-values     手动模式下矩阵 B 的元素，按行优先输入
```

## 4. 手动输入方式

手动输入有两种用法。

第一种是命令行直接传值，最适合自己测试：

```powershell
mpiexec -n 2 python mpi_matrix_mul.py --m 2 --k 3 --n 2 --input-mode manual --a-values "1 2 3 4 5 6" --b-values "7 8 9 10 11 12" --check --print-matrix
```

这里表示：

```text
A = [ [1, 2, 3],
      [4, 5, 6] ]

B = [ [7, 8],
      [9, 10],
      [11, 12] ]
```

第二种是交互输入。运行下面命令后，按提示逐行输入矩阵：

```powershell
mpiexec -n 2 python mpi_matrix_mul.py --m 2 --k 3 --n 2 --input-mode manual --check --print-matrix
```

## 5. 运行方式

随机矩阵正确性验证：

```powershell
mpiexec -n 2 python mpi_matrix_mul.py --m 2 --k 3 --n 2 --seed 1 --check --print-matrix
```

非整除场景验证：

```powershell
mpiexec -n 3 python mpi_matrix_mul.py --m 5 --k 4 --n 3 --seed 1 --check
```

`m < 进程数` 场景验证：

```powershell
mpiexec -n 4 python mpi_matrix_mul.py --m 2 --k 3 --n 2 --seed 1 --check
```

较大规模测试：

```powershell
mpiexec -n 4 python mpi_matrix_mul.py --m 200 --k 300 --n 150 --seed 1 --check
```

## 6. 输出内容

程序会输出以下信息：

- 进程数
- 矩阵维度
- 输入模式
- 每个进程负责的行范围
- 并行计算耗时
- 正确性校验结果

这些输出可以直接作为实验结果截图素材。

## 7. 常见问题

如果运行时提示找不到 `mpi4py`：

```powershell
python -m pip install mpi4py
```

如果运行时提示找不到 `mpiexec`，说明 Microsoft MPI 未正确安装或未加入环境变量，需要检查：

```text
C:\Program Files\Microsoft MPI\Bin
```

是否已加入系统 `PATH`。

如果使用交互输入时界面行为不稳定，优先改用 `--a-values` 和 `--b-values` 方式，这在 `mpiexec` 下通常更稳。

## 8. 提交前检查

- 主程序可通过 `mpiexec` 正常运行
- 已完成至少一组小规模正确性验证截图
- 已完成至少一组多进程测试截图
- 提交文件中包含源码和本说明文档
- 压缩包命名符合“第3次作业+学号+姓名.zip”
