import numpy as np
import matplotlib.pyplot as plt
import time

def create_sierpinski_carpet(n):
    """
    Sekwencyjne generowanie dywanu Sierpińskiego.

    @param n {int} - poziom rekurencji (głębokość fraktala)
    @return {numpy.ndarray} - macierz reprezentująca dywan Sierpińskiego
    """
    # Obliczenie rozmiaru macierzy (3^n x 3^n)
    size = 3 ** n

    # Inicjalizacja macierzy jedynkami
    carpet = np.ones((size, size), dtype=np.uint8)

    # Rekurencyjne usuwanie środkowych kwadratów
    def remove_center(x0, y0, size):
        """
        Rekurencyjnie usuwa środkowe kwadraty z dywanu.

        @param x0 {int} - początkowa współrzędna x
        @param y0 {int} - początkowa współrzędna y
        @param size {int} - rozmiar bieżącego kwadratu
        """
        # Warunek końcowy rekurencji
        if size < 3:
            return

        # Obliczenie nowego rozmiaru
        new_size = size // 3

        # Usunięcie środkowego kwadratu
        for i in range(x0 + new_size, x0 + 2*new_size):
            for j in range(y0 + new_size, y0 + 2*new_size):
                carpet[i, j] = 0

        # Rekurencyjne wywołanie dla 8 pozostałych kwadratów
        if new_size < 3:
            return
        for i in range(3):
            for j in range(3):
                if not (i == 1 and j == 1):
                    remove_center(x0 + i * new_size, y0 + j * new_size, new_size)

    # Rozpoczęcie rekurencji od całej macierzy
    remove_center(0, 0, size)

    return carpet

def plot_carpet(carpet, n):
    """
    Wizualizacja dywanu Sierpińskiego.

    @param carpet {numpy.ndarray} - macierz reprezentująca dywan
    @param n {int} - poziom rekurencji
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(carpet, cmap='binary')
    plt.title(f'Dywan Sierpińskiego (n={n})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"dywan_sierpinskiego_n{n}.png", dpi=150)
    plt.show()

def main():
    """
    Główna funkcja programu.
    """
    print("Generator dywanu Sierpińskiego")
    print("=" * 30)

    while True:
        try:
            n = int(input("Podaj poziom rekurencji (1-6): "))
            if 1 <= n <= 6:
                break
            else:
                print("Poziom rekurencji powinien być między 1 a 6.")
        except ValueError:
            print("Wprowadź poprawną liczbę całkowitą.")

    print(f"\nGenerowanie dywanu Sierpińskiego dla n={n}...")

    # Pomiar czasu generowania
    start_time = time.time()
    carpet = create_sierpinski_carpet(n)
    execution_time = time.time() - start_time

    print(f"Dywan wygenerowany w {execution_time:.4f} sekundy.")
    print(f"Rozmiar dywanu: {carpet.shape[0]}x{carpet.shape[1]} pikseli.")

    # Wizualizacja dywanu
    print("\nTworzenie wizualizacji...")
    plot_carpet(carpet, n)

    print("\nProgram zakończony.")

if __name__ == "__main__":
    main()
