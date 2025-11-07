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
print("ANÁLISIS DE TIEMPOS - PROCESO SUBWAY (4 ESTACIONES)")
print("="*70)
print("\n1. DATOS RECOLECTADOS:")
print(f"   - Total de observaciones: {len(df)}")
print(f"   - Período de medición: 2 horas")
print(f"   - Proceso de 4 estaciones sin tiempos de espera")

# Función para convertir tiempo MM:SS a segundos
def time_to_seconds(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

# Convertir todas las columnas de tiempo a segundos
time_columns = ['Tiempo pedido de pan y carne', 'Tiempo horno', 
                'Tiempo vegetales', 'Tiempo caja']

for col in time_columns:
    df[col + ' (seg)'] = df[col].apply(time_to_seconds)

# Convertir tiempo total también
df['Tiempo total (seg)'] = df['Tiempo total'].apply(time_to_seconds)

# Función para convertir de vuelta a formato MM:SS para visualización
def seconds_to_time(seconds):
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}:{secs:02d}"

print("\n" + "="*70)
print("2. INDICADORES CLAVE DE DESEMPEÑO (KPIs) - TIEMPOS")
print("="*70)

# KPIs de TIEMPO
print("\n📊 KPIs DE TIEMPO POR ESTACIÓN:")
print("-" * 70)

for col in time_columns:
    col_seg = col + ' (seg)'
    mean_time = df[col_seg].mean()
    std_time = df[col_seg].std()
    min_time = df[col_seg].min()
    max_time = df[col_seg].max()
    
    print(f"\n{col}:")
    print(f"  • Promedio: {seconds_to_time(int(mean_time))} ({mean_time.round()} seg)")
    print(f"  • Desv. Est: {std_time:.1f} seg")
    print(f"  • Rango: {seconds_to_time(min_time)} - {seconds_to_time(max_time)}")

# Tiempo total
print(f"\n{'Tiempo Total de Proceso'}:")
print(f"  • Promedio: {seconds_to_time(int(df['Tiempo total (seg)'].mean()))} ({df['Tiempo total (seg)'].mean().round()} seg)")
print(f"  • Desv. Est: {df['Tiempo total (seg)'].std():.1f} seg")
print(f"  • Mínimo: {df['Tiempo total'].min()} ({df['Tiempo total (seg)'].min().round()} seg)")
print(f"  • Máximo: {df['Tiempo total'].max()} ({df['Tiempo total (seg)'].max().round()} seg)")

# KPI de productividad temporal
print("\n" + "-" * 70)
print("📊 KPIs DE PRODUCTIVIDAD TEMPORAL:")
print("-" * 70)
tiempo_promedio_por_pan = df['Tiempo total (seg)'].sum() / df['Cantidad de panes'].sum()
ordenes_por_hora = (len(df) / (2 * 60 * 60)) * 3600  # 2 horas de observación
panes_por_hora = (df['Cantidad de panes'].sum() / (2 * 60 * 60)) * 3600

print(f"\n  • Tiempo promedio por pan: {seconds_to_time(int(tiempo_promedio_por_pan))} ({tiempo_promedio_por_pan.round()} seg)")
print(f"  • Órdenes por hora: {ordenes_por_hora/60:.1f} órdenes/hora")
print(f"  • Panes por hora: {panes_por_hora/60:.1f} panes/hora")
print(f"  • Throughput del proceso: {len(df)/2:.1f} órdenes/hora")

# Análisis por cantidad de panes (solo tiempos)
print("\n" + "-" * 70)
print("📊 ANÁLISIS DE TIEMPOS POR CANTIDAD DE PANES:")
print("-" * 70)
for cantidad in sorted(df['Cantidad de panes'].unique()):
    subset = df[df['Cantidad de panes'] == cantidad]
    print(f"\nÓrdenes de {cantidad} pan(es) - {len(subset)} observaciones:")
    print(f"  • Tiempo promedio total: {seconds_to_time(int(subset['Tiempo total (seg)'].mean()))}, ({subset['Tiempo total (seg)'].mean().round()} seg)")
    print(f"  • Tiempo por pan: {seconds_to_time(int(subset['Tiempo total (seg)'].mean() / cantidad))}, ({subset['Tiempo total (seg)'].mean().round()} seg)")

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
    porcentaje = (tiempo / df['Tiempo total (seg)'].mean()) * 100
    print(f"{i}. {estacion}: {seconds_to_time(int(tiempo))} ({porcentaje:.1f}% del tiempo total)")

# Análisis de eficiencia por estación
print("\n" + "-" * 70)
print("ANÁLISIS DE EFICIENCIA TEMPORAL POR ESTACIÓN:")
print("-" * 70)

estaciones_info = {
    'Tiempo pedido de pan y carne': 'Estación 1: Pedido y preparación inicial',
    'Tiempo horno': 'Estación 2: Horneado del pan',
    'Tiempo vegetales': 'Estación 3: Adición de vegetales',
    'Tiempo caja': 'Estación 4: Cobro y entrega final'
}

for estacion, descripcion in estaciones_info.items():
    tiempo_promedio = df[estacion + ' (seg)'].mean()
    print(f"\n{descripcion}:")
    print(f"  • Tiempo promedio: {seconds_to_time(int(tiempo_promedio))}")
    print(f"  • Contribución al proceso: {(tiempo_promedio/df['Tiempo total (seg)'].mean())*100:.1f}%")

# VISUALIZACIONES - Solo enfocadas en tiempos
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Análisis de Tiempos - Proceso Subway (4 Estaciones)', fontsize=16, fontweight='bold')

# 1. Tiempo promedio por estación
ax1 = axes[0, 0]
estaciones = [col.replace('Tiempo ', '').replace(' de pan y carne', ' (pan/carne)') for col in time_columns]
tiempos = [df[col + ' (seg)'].mean() for col in time_columns]
bars = ax1.bar(estaciones, tiempos, color='steelblue', alpha=0.7)
ax1.set_ylabel('Tiempo (segundos)')
ax1.set_title('Tiempo Promedio por Estación')
ax1.tick_params(axis='x', rotation=45)
for i, (bar, tiempo) in enumerate(zip(bars, tiempos)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             seconds_to_time(int(tiempo)), ha='center', fontsize=9)

# 2. Box plot de tiempos por estación
ax3 = axes[0, 1]
data_boxplot = [df[col + ' (seg)'].values for col in time_columns]
bp = ax3.boxplot(data_boxplot, labels=estaciones, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
    patch.set_alpha(0.7)
ax3.set_ylabel('Tiempo (segundos)')
ax3.set_title('Variabilidad por Estación')
ax3.tick_params(axis='x', rotation=45)

# 3. Tiempo total vs cantidad de panes
ax4 = axes[0, 2]
for cantidad in sorted(df['Cantidad de panes'].unique()):
    subset = df[df['Cantidad de panes'] == cantidad]
    ax4.scatter(subset.index, subset['Tiempo total (seg)'], 
               label=f'{cantidad} pan(es)', s=100, alpha=0.6)
ax4.set_xlabel('Número de Orden')
ax4.set_ylabel('Tiempo Total (segundos)')
ax4.set_title('Tiempo Total por Orden')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 4. Coeficiente de variación
ax5 = axes[1, 0]
cv_values = [cv_dict[col] for col in time_columns]
bars = ax5.bar(estaciones, cv_values, color='orange', alpha=0.7)
ax5.axhline(20, color='green', linestyle='--', label='Límite estable (20%)')
ax5.axhline(35, color='red', linestyle='--', label='Límite alto (35%)')
ax5.set_ylabel('Coeficiente de Variación (%)')
ax5.set_title('Variabilidad del Proceso (CV%)')
ax5.tick_params(axis='x', rotation=45)
ax5.legend()

# 5. Contribución porcentual al tiempo total
ax6 = axes[1, 1]
contribuciones = [(tiempo / sum(tiempos)) * 100 for tiempo in tiempos]
wedges, texts, autotexts = ax6.pie(contribuciones, labels=estaciones, autopct='%1.1f%%',
                                     startangle=90, colors=sns.color_palette("Set3"))
ax6.set_title('Contribución al Tiempo Total')

axNone = axes[1, 2]
axNone.axis('off')

plt.tight_layout()
plt.savefig('initial_analysis_charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("5. RECOMENDACIONES BASADAS EN ANÁLISIS DE TIEMPOS")
print("="*70)

print("\nBasado en el análisis de tiempos del proceso de 4 estaciones:")

# Identificar estación más lenta
estacion_mas_lenta = tiempo_promedio_estaciones_sorted[0][0]
print(f"\n✓ Prioridad 1: Optimizar '{estacion_mas_lenta}'")
print(f"  Es la estación que más tiempo consume en el proceso")
print(f"  Representa el {(tiempo_promedio_estaciones_sorted[0][1]/df['Tiempo total (seg)'].mean())*100:.1f}% del tiempo total")

# Identificar alta variabilidad
estaciones_alta_variabilidad = [col for col, cv in cv_dict.items() if cv > 35]
if estaciones_alta_variabilidad:
    print(f"\n✓ Prioridad 2: Reducir variabilidad en:")
    for estacion in estaciones_alta_variabilidad:
        print(f"  - {estacion} (CV: {cv_dict[estacion]:.1f}%)")
    print(f"  Estandarizar procedimientos y capacitar al personal")

print("\n✓ Prioridad 3: Mejoras operativas de tiempo")
print("  - Optimizar flujo de trabajo en estaciones lentas")
print("  - Balancear la carga de trabajo entre estaciones")
print("  - Reducir tiempos de setup entre órdenes")

print(f"\n✓ Prioridad 4: Metas de productividad temporal")
print(f"  - Tiempo actual por pan: {seconds_to_time(int(tiempo_promedio_por_pan))}")
print(f"  - Meta sugerida: reducir a 2:30 minutos por pan")
print(f"  - Esto incrementaría el throughput en {((tiempo_promedio_por_pan/150)-1)*100:.1f}%")

print("\n" + "="*70)
print("RESUMEN EJECUTIVO - ANÁLISIS DE TIEMPOS")
print("="*70)
print(f"• Proceso actual: {len(df)} órdenes en 2 horas")
print(f"• Throughput: {len(df)/2:.1f} órdenes/hora")
print(f"• Tiempo promedio por orden: {seconds_to_time(int(df['Tiempo total (seg)'].mean()))}, ({df['Tiempo total (seg)'].mean().round()} seg)")
print(f"• Estación crítica: {estacion_mas_lenta}")
print(f"• Productividad: {(df['Cantidad de panes'].sum()/2):.1f} panes/hora")
print(f"• Tiempo por pan: {seconds_to_time(int(tiempo_promedio_por_pan))}, ({tiempo_promedio_por_pan.round()} seg)")
print("="*70)