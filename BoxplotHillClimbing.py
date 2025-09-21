import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

results = np.loadtxt("resultados_finais.txt")

resultados = results.tolist()
algoritmo  = ['Hill Climbing'] * len(resultados)

df_result = pd.DataFrame({'fitness': resultados, 'algoritmo' : algoritmo})

sns.boxplot(df_result, x="algoritmo", y = 'fitness')
sns.swarmplot(df_result, x="algoritmo", y = 'fitness')

plt.show()