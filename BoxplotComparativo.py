import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

results = np.loadtxt("resultados_finais.txt")
results2 = np.loadtxt("resultados_finais_stochastic.txt")
results3 = np.loadtxt("ag_um_ponto.txt")
results4 = np.loadtxt("ag_dois_pontos.txt")
results5 = np.loadtxt("ag_uniforme.txt")

resultados = results.tolist()
algoritmo  = ['Hill Climbing'] * len(resultados)
algoritmo2  = ['STOCHASTIC Hill Climbing'] * len(results2)
algoritmo3 = ['AG - Um Ponto'] * len(results3)
algoritmo4 = ['AG - Dois Pontos'] * len(results4)
algoritmo5 = ['AG - Uniforme'] * len(results5)

df_result = pd.DataFrame({'fitness': resultados, 'algoritmo' : algoritmo})
df_result2 = pd.DataFrame({'fitness': results2, 'algoritmo' : algoritmo2})
df_result3 = pd.DataFrame({'fitness': results3, 'algoritmo' : algoritmo3})
df_result4 = pd.DataFrame({'fitness': results4, 'algoritmo' : algoritmo4})
df_result5 = pd.DataFrame({'fitness': results5, 'algoritmo' : algoritmo5})

df_final = pd.concat([df_result, df_result2, df_result3, df_result4, df_result5])
df_final = df_final.reset_index(drop=True)

sns.boxplot(df_final, x="algoritmo", y = 'fitness', hue = "algoritmo")
sns.swarmplot(df_final, x="algoritmo", y = 'fitness', hue = "algoritmo")

plt.show()