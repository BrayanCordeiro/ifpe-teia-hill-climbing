import numpy as np
import matplotlib.pyplot as plt

results_um = np.loadtxt("ag_um_ponto.txt")
results_dois = np.loadtxt("ag_dois_pontos.txt")
results_uniforme = np.loadtxt("ag_uniforme.txt")

plt.figure(figsize=(9, 5))

x = np.arange(1, results_um.size + 1)
plt.plot(x, results_um, label="GA - Um Ponto")

x = np.arange(1, results_dois.size + 1)
plt.plot(x, results_dois, label="GA - Dois Pontos")

x = np.arange(1, results_uniforme.size + 1)
plt.plot(x, results_uniforme, label="GA - Uniforme")

plt.title("Convergência")
plt.xlabel("Iteracao")
plt.ylabel("Melhor")

plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()