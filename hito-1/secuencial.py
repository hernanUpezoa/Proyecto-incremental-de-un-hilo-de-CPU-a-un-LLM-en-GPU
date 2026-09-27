import sys
import time
import numpy as np


def suma_secuencial(data):
    acc = 0.0

    for i in range(len(data)):
        acc += float(data[i])

    return np.float32(acc)


def main():
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 2 ** 24

    if n < 1:
        print("N debe ser >= 1")
        return 1

    a = np.full(n, 1.0, dtype=np.float32)

    t0 = time.perf_counter()

    suma = suma_secuencial(a)

    t1 = time.perf_counter()

    tiempo_ms = (t1 - t0) * 1000

    print(f"N={n} suma={suma} tiempo_ms={tiempo_ms}")

    return 0


if __name__ == "__main__":
    main()