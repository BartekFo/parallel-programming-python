import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, vectorize, float64

def sum_of_squares(x):
    s = 0
    for i in range(len(x)):
        s += x[i] * x[i]
    return s

@jit(nopython=True)
def sum_of_squares_jit(x):
    s = 0
    for i in range(len(x)):
        s += x[i] * x[i]
    return s

@jit(nopython=True, parallel=True)
def sum_of_squares_parallel(x):
    s = 0
    for i in prange(len(x)):
        s += x[i] * x[i]
    return s

Ns = np.linspace(10**4, 10**6, 10, dtype=int)

times_python = []
times_jit = []
times_parallel = []


# Warm-up for JIT compilation to avoid measuring compilation time
dummy = np.random.rand(10**6)
_ = sum_of_squares_jit(dummy)
_ = sum_of_squares_parallel(dummy)

for N in Ns:
    a = np.random.rand(N)
    
    start = time.time()
    result_python = sum_of_squares(a.tolist())
    end = time.time()
    times_python.append(end - start)
    
    start = time.time()
    result_jit = sum_of_squares_jit(a)
    end = time.time()
    times_jit.append(end - start)
    
    start = time.time()
    result_parallel = sum_of_squares_parallel(a)
    end = time.time()
    times_parallel.append(end - start)


plt.figure(figsize=(10, 6))
plt.plot(Ns, times_python, marker="o", label="Czysty Python")
plt.plot(Ns, times_jit, marker="s", label="Numba (nopython=True)")
plt.plot(Ns, times_parallel, marker="^", label="Numba (parallel=True)")
plt.xlabel("Rozmiar wektora N")
plt.ylabel("Czas wykonania [s]")
plt.title("Porównanie czasu wykonania funkcji sumy kwadratów")
plt.legend()
plt.grid(True)
plt.show()


def sum_matrix_seq(m):
    s = 0.0
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            s += m[i, j] * m[i, j]
    return s

@jit(nopython=True, parallel=True)
def sum_matrix_parallel(m):
    s = 0.0
    n = m.shape[0]
    for i in prange(n):
        for j in range(n):
            s += m[i, j] * m[i, j]
    return s

@vectorize([float64(float64)])
def square(x):
    return x * x

def sum_matrix_vectorized(m):
    return np.sum(square(m))

sizes = [1000, 2000, 3000, 4000]

times_seq = []
times_parallel = []
times_vectorized = []

# Warm-up for JIT compilation to avoid measuring compilation time
dummy = np.random.rand(1000, 1000)
_ = sum_matrix_seq(dummy)
_ = sum_matrix_parallel(dummy)
_ = sum_matrix_vectorized(dummy)

for N in sizes:
    print("Testujemy macierz o rozmiarze {}x{}".format(N, N))
    m = np.random.rand(N, N)

    start = time.time()
    result_seq = sum_matrix_seq(m)
    end = time.time()
    times_seq.append(end - start)
    
    start = time.time()
    result_parallel = sum_matrix_parallel(m)
    end = time.time()
    times_parallel.append(end - start)
    
    start = time.time()
    result_vectorized = sum_matrix_vectorized(m)
    end = time.time()
    times_vectorized.append(end - start)

plt.figure(figsize=(10, 6))
plt.plot(sizes, times_seq, marker="o", label="Sekwencyjne (czysty Python)")
plt.plot(sizes, times_parallel, marker="s", label="Numba równoległy (@jit, prange)")
plt.plot(sizes, times_vectorized, marker="^", label="Numba vectorize")
plt.xlabel("Rozmiar macierzy NxN (N)")
plt.ylabel("Czas wykonania [s]")
plt.title("Porównanie czasu wykonania sumowania kwadratów macierzy")
plt.legend()
plt.grid(True)
plt.show()