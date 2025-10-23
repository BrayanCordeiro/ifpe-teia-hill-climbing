import random
import numpy as np
from typing import List, Tuple
from Knapsack import knapsack

Individuo = List[int]
def fitness(ind: Individuo, dim=20) -> int:
    return knapsack(ind, dim=dim)[0]

def popular(pop_tamanho: int, dim: int) -> List[Individuo]:
    return [[1 if random.random() < 0.3 else 0 for _ in range(dim)] for _ in range(pop_tamanho)]

def selecionar_torneio(pop: List[Individuo], fits: List[int], k: int = 3) -> Individuo:
    idxs = random.sample(range(len(pop)), k)
    melhor = max(idxs, key=lambda i: fits[i])
    return pop[melhor][:]

def crossover_um(p1: Individuo, p2: Individuo) -> Tuple[Individuo, Individuo]:
    n = len(p1)
    if n < 2:
        return p1[:], p2[:]
    cut = random.randint(1, n - 1)
    c1 = p1[:cut] + p2[cut:]
    c2 = p2[:cut] + p1[cut:]
    return c1, c2

def crossover_dois(p1: Individuo, p2: Individuo) -> Tuple[Individuo, Individuo]:
    n = len(p1)
    if n < 3:
        return p1[:], p2[:]
    a, b = sorted(random.sample(range(1, n), 2))
    c1 = p1[:a] + p2[a:b] + p1[b:]
    c2 = p2[:a] + p1[a:b] + p2[b:]
    return c1, c2

def crossover_uniforme(p1: Individuo, p2: Individuo, prob_troca: float = 0.5) -> Tuple[Individuo, Individuo]:
    n = len(p1)
    c1, c2 = p1[:], p2[:]
    for i in range(n):
        if random.random() < prob_troca:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def mutar_bitflip(ind: Individuo, pm: float = 0.02) -> None:
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] = 1 - ind[i]

def elitismo(pop: List[Individuo], fts: List[int], e: int = 2) -> List[Individuo]:
    order = sorted(range(len(pop)), key=lambda i: fts[i], reverse=True)
    return [pop[i][:] for i in order[:e]]

def evoluir(
    pop: List[Individuo],
    dim: int,
    torneio_k: int,
    pc: float,
    pm: float,
    tipo_crossover: str
) -> List[Individuo]:
    pop_tamanho = len(pop)
    fits = [fitness(ind, dim) for ind in pop]

    nov_pop = elitismo(pop, fits, e=2)

    if tipo_crossover == "um_ponto":
        cross = crossover_um
    elif tipo_crossover == "dois_pontos":
        cross = crossover_dois
    elif tipo_crossover == "uniforme":
        cross = lambda a, b: crossover_uniforme(a, b)
    else:
        raise ValueError("tipo_crossover deve ser um de: 'um_ponto', 'dois_pontos', 'uniforme'.")

    while len(nov_pop) < pop_tamanho:
        p1 = selecionar_torneio(pop, fits, k=torneio_k)
        p2 = selecionar_torneio(pop, fits, k=torneio_k)

        if random.random() < pc:
            c1, c2 = cross(p1, p2)
        else:
            c1, c2 = p1[:], p2[:]

        mutar_bitflip(c1, pm=pm)
        if len(nov_pop) + 1 < pop_tamanho:
            mutar_bitflip(c2, pm=pm)

        nov_pop.append(c1)
        if len(nov_pop) < pop_tamanho:
            nov_pop.append(c2)

    return nov_pop

def executar_ag(
    dim: int = 20,
    pop_tamanho: int = 50,
    geracoes: int = 500,
    torneio_k: int = 3,
    pc: float = 0.80,
    pm: float = 0.02,
    tipo_crossover: str = "um_ponto",
    seed: int = None
) -> Tuple[Individuo, int]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pop = popular(pop_tamanho, dim)
    melhor_ind, melhor_fitness = None, -1

    for _ in range(geracoes):
        fits = [fitness(ind, dim) for ind in pop]
        gen_melhor_idx = int(np.argmax(fits))
        if fits[gen_melhor_idx] > melhor_fitness:
            melhor_fitness = fits[gen_melhor_idx]
            melhor_ind = pop[gen_melhor_idx][:]

        pop = evoluir(pop, dim, torneio_k, pc, pm, tipo_crossover)

    fits_final = [fitness(ind, dim) for ind in pop]
    melhor_idx_final = int(np.argmax(fits_final))
    melhor_ind_final = pop[melhor_idx_final][:]
    melhor_fitness_final = fits_final[melhor_idx_final]

    return melhor_ind_final, melhor_fitness_final

def executar_30(tipo_crossover: str) -> Tuple[float, float, List[int]]:
    resultados = []
    for run in range(30):
        seed = 12345 + run
        _,  fit_final = executar_ag(
            dim=20,
            pop_tamanho=50,
            geracoes=500,
            torneio_k=3,
            pc=0.80,
            pm=0.02,
            tipo_crossover=tipo_crossover,
            seed=seed
        )
        resultados.append(fit_final)
    resultados = np.array(resultados, dtype=float)
    return float(resultados.mean()), float(resultados.std(ddof=1)), resultados.tolist()

if __name__ == "__main__":
    cfgrcs = ["um_ponto", "dois_pontos", "uniforme"]
    sumario = {}

    for cfg in cfgrcs:
        mean, std, vals = executar_30(cfg)
        sumario[cfg] = (mean, std, vals)

    for cfg in cfgrcs:
        mean, std, _ = sumario[cfg]
        print(f"{cfg:>10s} -> media = {mean:.2f} | desvio-padrao = {std:.2f}")

    for cfg in cfgrcs:
        _, _, vals = sumario[cfg]
        np.savetxt(f"ag_{cfg}.txt", np.array(vals, dtype=int), fmt="%d")

    melhor_ind, melhor_fitness = executar_ag(tipo_crossover="dois_pontos", seed=777)
    val, peso, ok = knapsack(melhor_ind, dim=20)
    print("Melhor individuo:", melhor_ind)
    print(f"Valor={val} | Peso={peso} | Valido={ok} | Fitness={melhor_fitness}")