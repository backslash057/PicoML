# PicoML

**PicoML** is a lightweight, pure-C (C11) matrix math library that provides a clean, ergonomic abstraction over OpenBLAS (CBLAS). Designed for CPU-bound numerical tasks and neural network primitives, it stores 2D matrices in contiguous row-major `float` buffers and compiles both as a test runner and a shared library (`.so`) ready for Python `ctypes` bindings.

---

## Features

- **Lifecycle & Utilities**: Zero/ones/random normal matrix initializers, array wrappers, transposed views, and formatted pretty-printing.
- **Hardware-Accelerated GEMM**: Fast matrix multiplication via OpenBLAS `cblas_sgemm` with direct transpose support (no temporary allocations).
- **Vectorized Math**: Fast additions, subtractions, Hadamard products, and scalar multiplications via `cblas_saxpy` and `cblas_sscal`.
- **Broadcasting & Reductions**: Row-vector addition (forward pass bias) and column summation (bias gradient accumulation).
- **Deep Learning Primitives**: Forward and backward ReLU activations.
- **Safety**: Built-in dimension validation and informative error reporting.

---

## Project Structure

```text
.
├── Makefile            # Build configuration (test runner & shared library)
├── include/
│   ├── matrix.h        # Matrix struct definition and lifecycle declarations
│   └── ops.h           # High-level math and BLAS operation declarations
├── src/
│   ├── matrix.c        # Allocation, free, transpose, print, and initializers
│   ├── ops.c           # CBLAS-accelerated math and element-wise kernels
│   └── main.c          # Comprehensive verification test suite
├── LICENSE             # MIT License
└── README.md
```

---

## Prerequisites

You need a C11-compliant compiler (`gcc` or `clang`) and OpenBLAS development headers:

### Linux (Ubuntu / Debian)
```bash
sudo apt update
sudo apt install build-essential libopenblas-dev
```

### macOS (Homebrew)
```bash
brew install openblas
```
*(On macOS, ensure your `CFLAGS` and `LDFLAGS` point to Homebrew's OpenBLAS path if not linked globally: `export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"`).*

---

## Build & Installation

### 1. Compile
Run `make` to compile both the verification executable and the shared library:
```bash
make
```

This will produce:
- `test_picoml`: Executable test suite.
- `libpicoml.so`: Shared library for linking or Python `ctypes`.

### 2. Run Tests
Verify all matrix operations:
```bash
./test_picoml
```

### 3. Clean
To remove build artifacts:
```bash
make clean
```

---

## Python Integration (Quick Example)

Because `libpicoml.so` is compiled with `-fPIC`, you can call it directly from Python:

```python
import ctypes

# Load the library
picoml = ctypes.CDLL("./libpicoml.so")

# Configure return types
picoml.matrix_create.restype = ctypes.c_void_p
picoml.matrix_free.argtypes = [ctypes.c_void_p]

# Create a 3x3 matrix and free it
mat = picoml.matrix_create(3, 3)
picoml.matrix_free(mat)
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
