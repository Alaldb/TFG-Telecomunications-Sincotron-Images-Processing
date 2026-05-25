import numpy as np


def extraer_runs_fila(fila_idx, fila):
    runs = []
    inicios = np.where(np.diff(fila) == 1)[0] + 1
    fines = np.where(np.diff(fila) == -1)[0]

    # caso especial: la fila empieza en 1
    if fila[0] == 1:
        inicios = np.concatenate([[0], inicios])

    # caso especial: la fila termina en 1
    if fila[-1] == 1:
        fines = np.concatenate([fines, [len(fila) - 1]])

    for inicio, fin in zip(inicios, fines):
        runs.append((fila_idx, int(inicio), int(fin)))

    return runs

def asignar_etiquetas(runs_por_fila, ancho):
    T = [0]
    etiqueta_nueva = [1]

    def new_tree():
        l = etiqueta_nueva[0]
        T.append(l)  # T[l] = l
        etiqueta_nueva[0] += 1
        return l

    def find_root(u):
        while T[u] != u:
            u = T[u]
        return u

    def set_root(u, r):
        while T[u] != u:
            v = T[u]
            T[u] = r
            u = v
        T[u] = r

    def union(u, v):
        r = find_root(u)
        s = find_root(v)
        if r == s:
            return r
        if r > s:
            r = s
        set_root(u, r)
        set_root(v, r)
        return r

    # asignar etiqueta a cada run
    etiquetas_runs = {}  # (fila, inicio, fin) -> etiqueta provisional

    for fila_idx, runs in enumerate(runs_por_fila):
        runs_anteriores = runs_por_fila[fila_idx - 1] if fila_idx > 0 else []

        for run in runs:
            _, inicio, fin = run

            # buscar runs de la fila anterior 8-conectados
            conectados = [
                r for r in runs_anteriores
                if inicio <= r[2] + 1 and fin >= r[1] - 1
            ]

            if len(conectados) == 0:
                # ningún vecino → etiqueta nueva
                etiquetas_runs[run] = new_tree()
            else:
                # heredar etiqueta del primero
                etiqueta = find_root(etiquetas_runs[conectados[0]])
                etiquetas_runs[run] = etiqueta

                # fusionar con el resto
                for r in conectados[1:]:
                    etiqueta = union(etiqueta, etiquetas_runs[r])
                    etiquetas_runs[run] = etiqueta

    return etiquetas_runs, T