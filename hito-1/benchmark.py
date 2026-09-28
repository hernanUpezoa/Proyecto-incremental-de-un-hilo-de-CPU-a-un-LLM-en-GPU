import sys
import time
import threading
import multiprocessing
import numpy as np

# --- 1. Variables Globales y Sincronización ---
# threading.Lock() es el equivalente a std::mutex en C++
mtx = threading.Lock()
proximo = 0
BLOQUE = 1 << 16  # Tamaño de bloque: 65536 elementos

# --- 2. Lógica Secuencial ---
def suma_secuencial(data):
    acc = 0.0
    # Iteración explícita para generar una línea base comparable
    for i in range(len(data)):
        acc += float(data[i])
    return np.float32(acc)

# --- 3. Lógica Paralela (Patrón Worker) ---
def worker(data, n, parciales, thread_id):
    global proximo
    acc = 0.0
    
    while True:
        lo = 0
        hi = 0
        
        # --- SECCIÓN CRÍTICA ---
        # 'with mtx' reemplaza a std::lock_guard de C++
        with mtx:
            if proximo >= n:
                break  # Fin del trabajo
            lo = proximo
            # Evita salirse de los límites del vector
            hi = min(n, proximo + BLOQUE)
            proximo = hi
        # --- FIN SECCIÓN CRÍTICA ---
        
        # El trabajo intensivo se hace sin bloquear a otros hilos
        for i in range(lo, hi):
            acc += float(data[i])
            
    # Cada hilo guarda en su índice para evitar race conditions
    parciales[thread_id] = acc

def suma_par(data, n, t):
    global proximo
    proximo = 0  # Reinicio la variable compartida para múltiples corridas
    parciales = [0.0] * t
    hilos = []

    # Creación y despacho de hilos
    for i in range(t):
        hilo = threading.Thread(target=worker, args=(data, n, parciales, i))
        hilos.append(hilo)
        hilo.start()

    # Barrera de sincronización: esperar a que todos terminen (h.join() en C++)
    for hilo in hilos:
        hilo.join()

    # Reducción final en el hilo principal
    total = 0.0
    for p in parciales:
        total += p
    return np.float32(total)

# --- 4. Orquestación y Benchmark ---
def main():
    # Parámetro N por línea de comandos (por defecto 2^24)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 24)
    if n < 1:
        print("Error: N debe ser >= 1", file=sys.stderr)
        sys.exit(1)

    # Identificación de hilos lógicos del hardware
    hw = multiprocessing.cpu_count()
    
    # Inicialización del vector en Float32 usando NumPy
    a = np.full(n, 1.0, dtype=np.float32)

    print(f"hardware_concurrency={hw} N={n}")
    print("hilos,ms,speedup,suma")

    # Medición de la versión secuencial (T_seq)
    # Se utiliza perf_counter() como lo indica la cátedra
    t0 = time.perf_counter()
    s_seq = suma_secuencial(a)
    t1 = time.perf_counter()
    ms_seq = (t1 - t0) * 1000.0

    print(f"seq,{ms_seq:.2f},1.00,{s_seq}")

    # Lista de hilos a testear
    candidatos = [1, 2, 4, 8, 16]
    
    for t in candidatos:
        # Prevención de sobre-suscripción (over-subscription) extrema
        if hw != 0 and t > hw * 2:
            continue
            
        u0 = time.perf_counter()
        s_par = suma_par(a, n, t)
        u1 = time.perf_counter()
        
        ms_par = (u1 - u0) * 1000.0
        # Cálculo del Speedup: S = T_seq / T_par
        speedup = (ms_seq / ms_par) if ms_par > 0.0 else 0.0
        
        print(f"{t},{ms_par:.2f},{speedup:.4f},{s_par}")

if __name__ == "__main__":
    main()