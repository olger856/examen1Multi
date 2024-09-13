import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Datos proporcionados en la tabla
data = {
    'sexo': [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    'edad': [9, 10, 9, 9, 9, 10, 10, 10, 9, 9, 10, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10],
    'divertid': [6, 2, 7, 4, 1, 6, 5, 7, 2, 3, 1, 5, 2, 4, 6, 3, 4, 3, 3, 4, 2, 5, 4, 3, 1],
    'pidocomp': [7, 1, 6, 4, 2, 6, 6, 7, 3, 3, 2, 5, 2, 4, 4, 4, 7, 2, 4, 3, 2, 7, 2, 5, 2],
    'aprendom': [3, 4, 3, 4, 2, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 5, 2, 7, 4, 2, 5, 2, 4, 2],
    'excur': [3, 4, 3, 6, 2, 4, 3, 4, 3, 3, 3, 4, 4, 7, 4, 4, 5, 6, 7, 4, 2, 5, 7, 5, 3],
    'quitatie': [4, 4, 3, 5, 2, 4, 3, 4, 4, 6, 5, 4, 5, 6, 2, 5, 4, 7, 6, 7, 3, 4, 7, 4, 2],
    'nomeint': [2, 3, 1, 3, 6, 3, 3, 3, 4, 5, 5, 2, 4, 6, 5, 4, 2, 4, 4, 6, 2, 4, 7, 4, 4],
    'gustovis': [1, 0, 1, 1, 0, 1, 1, 1, 0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 3, 1, 0, 0]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Escalado de las variables (normalización)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Aplicación del algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Mostrar los resultados
print("Centroides:")
print(kmeans.cluster_centers_)

print("\nAsignación a conglomerados:")
print(df[['sexo', 'edad', 'divertid', 'pidocomp', 'aprendom', 'excur', 'quitatie', 'nomeint', 'gustovis', 'cluster']])

# Visualización
plt.scatter(df['edad'], df['divertid'], c=df['cluster'], cmap='viridis')
plt.title("Clustering basado en Edad y Diversión")
plt.xlabel("Edad")
plt.ylabel("Diversión")
plt.show()