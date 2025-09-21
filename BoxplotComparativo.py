import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

results = np.loadtxt("resultados_finais.txt")
results2 = np.loadtxt("resultados_finais_stochastic.txt")

resultados = results.tolist()
algoritmo  = ['Hill Climbing'] * len(resultados)
algoritmo2  = ['STOCHASTIC Hill Climbing'] * len(results2)

df_result = pd.DataFrame({'fitness': resultados, 'algoritmo' : algoritmo})
df_result2 = pd.DataFrame({'fitness': results2, 'algoritmo' : algoritmo2})

df_final = pd.concat([df_result, df_result2])
df_final = df_final.reset_index(drop=True)

sns.boxplot(df_final, x="algoritmo", y = 'fitness', hue = "algoritmo")
sns.swarmplot(df_final, x="algoritmo", y = 'fitness', hue = "algoritmo")

plt.show()