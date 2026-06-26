import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Estilo IEEE
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300

# Función para extraer dataset de la ruta
def get_dataset_from_path(filepath):
    parts = filepath.split(os.sep)
    if len(parts) >= 2:
        return parts[1]
    return "unknown"

# Buscar archivos
topn_files = glob.glob('results/**/topn_metrics.csv', recursive=True)
error_files = glob.glob('results/**/error_metrics.csv', recursive=True)

print(f"✅ Encontrados {len(topn_files)} archivos topn y {len(error_files)} archivos error")

# Cargar topn
dfs_topn = []
for f in topn_files:
    df = pd.read_csv(f)
    df['dataset'] = get_dataset_from_path(f)
    dfs_topn.append(df)
df_topn = pd.concat(dfs_topn, ignore_index=True)

# Cargar error
dfs_err = []
for f in error_files:
    df = pd.read_csv(f)
    df['dataset'] = get_dataset_from_path(f)
    dfs_err.append(df)
df_err = pd.concat(dfs_err, ignore_index=True)

# Filtrar solo TEST
df_topn = df_topn[df_topn['split'] == 'TEST']
df_err = df_err[df_err['split'] == 'TEST']

# Mostrar datos cargados
print("\n📊 Datos de error cargados:")
print(df_err[['model', 'dataset', 'rmse', 'mae']])
print("\n📊 Datos de topn cargados:")
print(df_topn[['model', 'dataset', 'recall', 'ndcg', 'map']])

# Orden
datasets = sorted(df_err['dataset'].unique())
models = ['als', 'sgd', 'hybrid']
model_labels = ['ALS', 'SGD', 'Hybrid']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, Naranja, Verde

# ---- Función para crear una imagen individual ----
def plot_metric(data, metric_name, ylabel, title, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, model in enumerate(models):
        vals = []
        for d in datasets:
            val = data[(data['dataset'] == d) & (data['model'] == model)][metric_name].values
            vals.append(val[0] if len(val) > 0 else 0)
        bars = ax.bar(x + i*width, vals, width, label=model_labels[i], color=colors[i])
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(vals) if max(vals) > 0 else 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Guardado: {filename} y {filename.replace('.png', '.pdf')}")
    plt.close()

# ---- Generar 4 imágenes separadas ----
# 1. RMSE
plot_metric(df_err, 'rmse', 'RMSE', '(a) RMSE', 'fig_rmse.png', ylim=(0.85, 1.20))

# 2. MAE
plot_metric(df_err, 'mae', 'MAE', '(b) MAE', 'fig_mae.png', ylim=(0.50, 0.95))

# 3. Recall@10
plot_metric(df_topn, 'recall', 'Recall@10', '(c) Recall@10', 'fig_recall.png', ylim=(0.0, 0.45))

# 4. NDCG@10
plot_metric(df_topn, 'ndcg', 'NDCG@10', '(d) NDCG@10', 'fig_ndcg.png', ylim=(0.0, 0.45))

print("\n✅ ¡Todas las figuras generadas exitosamente!")
print("Archivos generados: fig_rmse.png, fig_mae.png, fig_recall.png, fig_ndcg.png")