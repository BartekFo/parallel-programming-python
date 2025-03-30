import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

@njit
def create_sierpinski_carpet_sequential(n):
    """
    Sekwencyjne generowanie dywanu Sierpińskiego z optymalizacją Numba.

    @param n {int} - poziom rekurencji (głębokość fraktala)
    @return {numpy.ndarray} - macierz reprezentująca dywan Sierpińskiego
    """
    # Obliczenie rozmiaru macierzy (3^n x 3^n)
    size = 3 ** n

    # Inicjalizacja macierzy jedynkami
    carpet = np.ones((size, size), dtype=np.uint8)

    # Iteracyjne sprawdzanie każdego punktu
    for i in range(size):
        for j in range(size):
            x, y = i, j
            for k in range(n):
                if (x % 3 == 1) and (y % 3 == 1):
                    carpet[i, j] = 0
                    break
                x //= 3
                y //= 3

    return carpet

# Wersja równoległa z Numba
@njit(parallel=True)
def create_sierpinski_carpet_parallel(n):
    """
    Równoległe generowanie dywanu Sierpińskiego z optymalizacją Numba.

    @param n {int} - poziom rekurencji (głębokość fraktala)
    @return {numpy.ndarray} - macierz reprezentująca dywan Sierpińskiego
    """
    size = 3 ** n

    carpet = np.ones((size, size), dtype=np.uint8)

    for i in prange(size):  # Stały krok = 1
        for j in range(size):
            x, y = i, j
            for k in range(n):
                if (x % 3 == 1) and (y % 3 == 1):
                    carpet[i, j] = 0
                    break
                x //= 3
                y //= 3

    return carpet

def plot_carpet(carpet, title):
    """
    Wizualizacja dywanu Sierpińskiego.

    @param carpet {numpy.ndarray} - macierz reprezentująca dywan
    @param title {str} - tytuł wykresu
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(carpet, cmap='binary')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace('=', '')}.png", dpi=150)
    plt.close()

def benchmark_comparison(n_values):
    """
    Porównanie wydajności implementacji sekwencyjnej i równoległej.

    @param n_values {list} - lista poziomów rekurencji do przetestowania
    @return {tuple} - krotka (czasy_sekwencyjne, czasy_równoległe)
    """
    sequential_times = []
    parallel_times = []

    # Rozgrzewka Numba
    _ = create_sierpinski_carpet_sequential(2)
    _ = create_sierpinski_carpet_parallel(2)

    for n in n_values:
        print(f"Testowanie dla n={n}...")

        start_time = time.time()
        carpet_seq = create_sierpinski_carpet_sequential(n)
        seq_time = time.time() - start_time
        sequential_times.append(seq_time)
        print(f"  Sekwencyjnie z Numba: {seq_time:.4f}s")

        start_time = time.time()
        carpet_par = create_sierpinski_carpet_parallel(n)
        par_time = time.time() - start_time
        parallel_times.append(par_time)
        print(f"  Równolegle z Numba:   {par_time:.4f}s")

        if not np.array_equal(carpet_seq, carpet_par):
            print("  UWAGA: Wyniki implementacji sekwencyjnej i równoległej różnią się!")

        plot_carpet(carpet_seq, f"Dywan Sierpinskiego n={n}")

        speedup = seq_time / par_time if par_time > 0 else 0
        print(f"  Przyspieszenie: {speedup:.2f}x")
        print()

    return sequential_times, parallel_times

def plot_comparison(n_values, sequential_times, parallel_times):
    """
    Rysuje wykres porównujący wydajność implementacji sekwencyjnej i równoległej.

    @param n_values {list} - lista poziomów rekurencji
    @param sequential_times {list} - czasy wykonania dla implementacji sekwencyjnej
    @param parallel_times {list} - czasy wykonania dla implementacji równoległej
    """
    plt.figure(figsize=(10, 6))

    plt.plot(n_values, sequential_times, 'o-', label='Sekwencyjnie (Numba)')
    plt.plot(n_values, parallel_times, 's-', label='Równolegle (Numba parallel)')

    plt.title('Porównanie czasów wykonania')
    plt.xlabel('Poziom rekurencji (n)')
    plt.ylabel('Czas [s]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("porownanie_czasow.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))

    speedups = [seq_time / par_time if par_time > 0 else 0
                for seq_time, par_time in zip(sequential_times, parallel_times)]

    plt.bar(n_values, speedups)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)

    plt.title('Przyspieszenie obliczeń równoległych z Numba')
    plt.xlabel('Poziom rekurencji (n)')
    plt.ylabel('Przyspieszenie (x razy)')
    plt.grid(True, axis='y')

    for i, speedup in enumerate(speedups):
        plt.text(n_values[i], speedup + 0.1, f'{speedup:.2f}x',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("przyspieszenie.png", dpi=150)
    plt.close()

def main():
    print("Generator dywanu Sierpińskiego - z wykorzystaniem Numba")
    print("=" * 60)

    n_values = [3, 4, 5]
    print("Porównanie wydajności implementacji sekwencyjnej i równoległej:")
    sequential_times, parallel_times = benchmark_comparison(n_values)

    plot_comparison(n_values, sequential_times, parallel_times)

    print("\nWyniki zostały zapisane w plikach PNG.")
    print("Program zakończony.")

if __name__ == "__main__":
    main()
