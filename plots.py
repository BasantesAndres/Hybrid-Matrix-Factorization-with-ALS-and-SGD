import pandas as pd
import matplotlib.pyplot as plt

# Cargar los resultados CSV generados por el programa
topn_metrics = pd.read_csv("results/ml100k/topn_metrics.csv")  # Cambia la ruta según el dataset

# Filtra las métricas de ALS y Híbrido
als_data = topn_metrics[topn_metrics['model'] == 'ALS']
hybrid_data = topn_metrics[topn_metrics['model'] == 'HYBRID']

# Comparación de Recall@10 y NDCG@10
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de Recall@10
ax[0].bar(als_data['dataset'], als_data['recall'], width=0.4, label='ALS', align='center', alpha=0.7)
ax[0].bar(hybrid_data['dataset'], hybrid_data['recall'], width=0.4, label='HYBRID', align='edge', alpha=0.7)
ax[0].set_title('Recall@10 Comparison')
ax[0].set_ylabel('Recall@10')
ax[0].legend()

# Gráfico de NDCG@10
ax[1].bar(als_data['dataset'], als_data['ndcg'], width=0.4, label='ALS', align='center', alpha=0.7)
ax[1].bar(hybrid_data['dataset'], hybrid_data['ndcg'], width=0.4, label='HYBRID', align='edge', alpha=0.7)
ax[1].set_title('NDCG@10 Comparison')
ax[1].set_ylabel('NDCG@10')
ax[1].legend()

plt.tight_layout()
plt.show()
