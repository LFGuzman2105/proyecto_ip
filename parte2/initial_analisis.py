import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Leer los datos
df = pd.read_csv('datos_subway.csv')

print("="*70)
print("ANÁLISIS DE PROCESO - SUBWAY")
print("="*70)
print("\n1. DATOS RECOLECTADOS:")
print(f"   - Total de observaciones: {len(df)}")
print(f"   - Período de medición: {len(df)} órdenes")

# Función para convertir tiempo MM:SS a segundos
def time_to_seconds(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

# Convertir todas las columnas de tiempo a segundos
time_columns = ['Tiempo ordenar', 'Tiempo preparar pan', 'Tiempo horno', 
                'Tiempo vegetales', 'Tiempo caja']

for col in time_columns:
    df[col + ' (seg)'] = df[col].apply(time_to_seconds)

# Calcular tiempo total por orden
df['Tiempo Total (seg)'] = df[[col + ' (seg)' for col in time_columns]].sum(axis=1)

# Convertir de vuelta a formato MM:SS para visualización
def seconds_to_time(seconds):
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}:{secs:02d}"

df['Tiempo Total'] = df['Tiempo Total (seg)'].apply(seconds_to_time)

print("\n" + "="*70)
print("2. INDICADORES CLAVE DE DESEMPEÑO (KPIs)")
print("="*70)

# KPIs de TIEMPO
print("\n📊 KPIs DE TIEMPO:")
print("-" * 70)

for col in time_columns:
    col_seg = col + ' (seg)'
    mean_time = df[col_seg].mean()
    std_time = df[col_seg].std()
    min_time = df[col_seg].min()
    max_time = df[col_seg].max()
    
    print(f"\n{col}:")
    print(f"  • Promedio: {seconds_to_time(int(mean_time))} ({mean_time:.1f} seg)")
    print(f"  • Desv. Est: {std_time:.1f} seg")
    print(f"  • Rango: {seconds_to_time(min_time)} - {seconds_to_time(max_time)}")

# Tiempo total
print(f"\n{'Tiempo Total de Proceso'}:")
print(f"  • Promedio: {seconds_to_time(int(df['Tiempo Total (seg)'].mean()))} ({df['Tiempo Total (seg)'].mean():.1f} seg)")
print(f"  • Desv. Est: {df['Tiempo Total (seg)'].std():.1f} seg")
print(f"  • Mínimo: {df['Tiempo Total'].min()}")
print(f"  • Máximo: {df['Tiempo Total'].max()}")

# KPIs de CANTIDAD
print("\n" + "-" * 70)
print("📊 KPIs DE CANTIDAD:")
print("-" * 70)
print(f"\n  • Total de panes: {df['Cantidad de panes'].sum()}")
print(f"  • Promedio por orden: {df['Cantidad de panes'].mean():.2f} panes")
print(f"  • Órdenes de 1 pan: {(df['Cantidad de panes'] == 1).sum()} ({(df['Cantidad de panes'] == 1).sum()/len(df)*100:.1f}%)")
print(f"  • Órdenes de 2 panes: {(df['Cantidad de panes'] == 2).sum()} ({(df['Cantidad de panes'] == 2).sum()/len(df)*100:.1f}%)")

# KPI de productividad
print("\n" + "-" * 70)
print("📊 KPIs DE PRODUCTIVIDAD:")
print("-" * 70)
tiempo_promedio_por_pan = df['Tiempo Total (seg)'].sum() / df['Cantidad de panes'].sum()
print(f"\n  • Tiempo promedio por pan: {seconds_to_time(int(tiempo_promedio_por_pan))} ({tiempo_promedio_por_pan:.1f} seg)")

# Análisis por cantidad de panes
print("\n" + "-" * 70)
print("📊 ANÁLISIS POR CANTIDAD DE PANES:")
print("-" * 70)
for cantidad in sorted(df['Cantidad de panes'].unique()):
    subset = df[df['Cantidad de panes'] == cantidad]
    print(f"\nÓrdenes de {cantidad} pan(es) - {len(subset)} observaciones:")
    print(f"  • Tiempo promedio total: {seconds_to_time(int(subset['Tiempo Total (seg)'].mean()))}")
    print(f"  • Tiempo por pan: {seconds_to_time(int(subset['Tiempo Total (seg)'].mean() / cantidad))}")

# VARIABILIDAD DEL PROCESO
print("\n" + "="*70)
print("3. ANÁLISIS DE VARIABILIDAD")
print("="*70)

cv_dict = {}
for col in time_columns:
    col_seg = col + ' (seg)'
    cv = (df[col_seg].std() / df[col_seg].mean()) * 100
    cv_dict[col] = cv
    print(f"\n{col}:")
    print(f"  • Coeficiente de Variación: {cv:.1f}%")
    if cv < 20:
        print(f"  • Evaluación: ✓ Proceso estable")
    elif cv < 35:
        print(f"  • Evaluación: ⚠ Variabilidad moderada")
    else:
        print(f"  • Evaluación: ✗ Alta variabilidad - requiere atención")

# IDENTIFICAR CUELLOS DE BOTELLA
print("\n" + "="*70)
print("4. IDENTIFICACIÓN DE CUELLOS DE BOTELLA")
print("="*70)

tiempo_promedio_estaciones = {col: df[col + ' (seg)'].mean() for col in time_columns}
tiempo_promedio_estaciones_sorted = sorted(tiempo_promedio_estaciones.items(), 
                                          key=lambda x: x[1], reverse=True)

print("\nEstaciones ordenadas por tiempo promedio:")
for i, (estacion, tiempo) in enumerate(tiempo_promedio_estaciones_sorted, 1):
    porcentaje = (tiempo / df['Tiempo Total (seg)'].mean()) * 100
    print(f"{i}. {estacion}: {seconds_to_time(int(tiempo))} ({porcentaje:.1f}% del tiempo total)")

# VISUALIZACIONES
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Análisis de Proceso - Subway', fontsize=16, fontweight='bold')

# 1. Tiempo promedio por estación
ax1 = axes[0, 0]
estaciones = [col.replace('Tiempo ', '') for col in time_columns]
tiempos = [df[col + ' (seg)'].mean() for col in time_columns]
bars = ax1.bar(estaciones, tiempos, color='steelblue', alpha=0.7)
ax1.set_ylabel('Tiempo (segundos)')
ax1.set_title('Tiempo Promedio por Estación')
ax1.tick_params(axis='x', rotation=45)
for i, (bar, tiempo) in enumerate(zip(bars, tiempos)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             seconds_to_time(int(tiempo)), ha='center', fontsize=9)

# 2. Distribución de tiempo total
ax2 = axes[0, 1]
ax2.hist(df['Tiempo Total (seg)'], bins=10, color='coral', alpha=0.7, edgecolor='black')
ax2.axvline(df['Tiempo Total (seg)'].mean(), color='red', linestyle='--', 
            linewidth=2, label='Media')
ax2.set_xlabel('Tiempo Total (segundos)')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Distribución de Tiempo Total')
ax2.legend()

# 3. Box plot de tiempos por estación
ax3 = axes[0, 2]
data_boxplot = [df[col + ' (seg)'].values for col in time_columns]
bp = ax3.boxplot(data_boxplot, labels=estaciones, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
    patch.set_alpha(0.7)
ax3.set_ylabel('Tiempo (segundos)')
ax3.set_title('Variabilidad por Estación')
ax3.tick_params(axis='x', rotation=45)

# 4. Tiempo total vs cantidad de panes
ax4 = axes[1, 0]
for cantidad in sorted(df['Cantidad de panes'].unique()):
    subset = df[df['Cantidad de panes'] == cantidad]
    ax4.scatter(subset.index, subset['Tiempo Total (seg)'], 
               label=f'{cantidad} pan(es)', s=100, alpha=0.6)
ax4.set_xlabel('Número de Orden')
ax4.set_ylabel('Tiempo Total (segundos)')
ax4.set_title('Tiempo Total por Orden')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Coeficiente de variación
ax5 = axes[1, 1]
cv_values = [cv_dict[col] for col in time_columns]
bars = ax5.bar(estaciones, cv_values, color='orange', alpha=0.7)
ax5.axhline(20, color='green', linestyle='--', label='Límite estable (20%)')
ax5.axhline(35, color='red', linestyle='--', label='Límite alto (35%)')
ax5.set_ylabel('Coeficiente de Variación (%)')
ax5.set_title('Variabilidad del Proceso (CV%)')
ax5.tick_params(axis='x', rotation=45)
ax5.legend()

# 6. Contribución porcentual al tiempo total
ax6 = axes[1, 2]
contribuciones = [(tiempo / sum(tiempos)) * 100 for tiempo in tiempos]
wedges, texts, autotexts = ax6.pie(contribuciones, labels=estaciones, autopct='%1.1f%%',
                                     startangle=90, colors=sns.color_palette("Set3"))
ax6.set_title('Contribución al Tiempo Total')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("5. RECOMENDACIONES")
print("="*70)

print("\nBasado en el análisis de los KPIs:")

# Identificar estación más lenta
estacion_mas_lenta = tiempo_promedio_estaciones_sorted[0][0]
print(f"\n✓ Prioridad 1: Optimizar '{estacion_mas_lenta}'")
print(f"  Es la estación que más tiempo consume en el proceso")

# Identificar alta variabilidad
estaciones_alta_variabilidad = [col for col, cv in cv_dict.items() if cv > 35]
if estaciones_alta_variabilidad:
    print(f"\n✓ Prioridad 2: Reducir variabilidad en:")
    for estacion in estaciones_alta_variabilidad:
        print(f"  - {estacion} (CV: {cv_dict[estacion]:.1f}%)")
    print(f"  Estandarizar procedimientos y capacitar al personal")

print("\n✓ Prioridad 3: Considerar capacitación cruzada")
print("  Para balancear la carga de trabajo entre estaciones")

print("\n" + "="*70)