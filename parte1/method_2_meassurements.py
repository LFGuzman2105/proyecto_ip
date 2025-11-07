import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Cargar los datos del segundo dataset
df = pd.read_csv('datos2.csv')

# Limpiar espacios en blanco
df.columns = df.columns.str.strip()
df['Lectura'] = df['Lectura'].str.strip()

# Convertir fecha a formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')

print("="*70)
print("ANÁLISIS DE VELOCIDAD DE LECTURA - MÉTODO 2")
print("="*70)
print(f"\nPeríodo de medición: {df['Fecha'].min().strftime('%d/%m/%Y')} - {df['Fecha'].max().strftime('%d/%m/%Y')}")
print(f"Total de mediciones: {len(df)*2} ({len(df)} por persona)")
print(f"Número de lecturas diferentes: {df['Lectura'].nunique()}")
print("\n⚠️  Nota: Estas mediciones se realizaron escuchando música")

# ==========================================
# ESTADÍSTICAS GENERALES POR PERSONA
# ==========================================
print("\n" + "="*70)
print("1. ESTADÍSTICAS DESCRIPTIVAS POR PERSONA")
print("="*70)

stats_luis = df['Palabras por Minuto (Luis)'].describe()
stats_byron = df['Palabras por Minuto (Byron)'].describe()

comparison = pd.DataFrame({
    'Luis': stats_luis,
    'Byron': stats_byron,
    'Diferencia': stats_luis - stats_byron
})

print("\n", comparison.round(2))

# Coeficiente de variación (para medir consistencia)
cv_luis = (df['Palabras por Minuto (Luis)'].std() / df['Palabras por Minuto (Luis)'].mean()) * 100
cv_byron = (df['Palabras por Minuto (Byron)'].std() / df['Palabras por Minuto (Byron)'].mean()) * 100

print(f"\nCoeficiente de Variación:")
print(f"  Luis:  {cv_luis:.2f}% {'(más consistente)' if cv_luis < cv_byron else ''}")
print(f"  Byron: {cv_byron:.2f}% {'(más consistente)' if cv_byron < cv_luis else ''}")

# ==========================================
# ESTADÍSTICAS POR LECTURA
# ==========================================
print("\n" + "="*70)
print("2. PROMEDIOS POR TIPO DE LECTURA")
print("="*70)

lecturas_stats = df.groupby('Lectura').agg({
    'Palabras por Minuto (Luis)': ['count', 'mean', 'std'],
    'Palabras por Minuto (Byron)': ['mean', 'std']
}).round(2)

lecturas_stats.columns = ['N_Mediciones', 'Luis_Promedio', 'Luis_DesvEst', 'Byron_Promedio', 'Byron_DesvEst']

print("\n", lecturas_stats.sort_values('Luis_Promedio', ascending=False))

# ==========================================
# EVOLUCIÓN TEMPORAL
# ==========================================
print("\n" + "="*70)
print("3. EVOLUCIÓN TEMPORAL (Promedios por día)")
print("="*70)

evolucion = df.groupby('Fecha').agg({
    'Palabras por Minuto (Luis)': 'mean',
    'Palabras por Minuto (Byron)': 'mean'
}).round(1)

evolucion.columns = ['Luis', 'Byron']

print("\n", evolucion)

# Calcular tendencia (primera parte vs segunda parte de la semana)
mitad = len(evolucion) // 2
primera_parte_luis = evolucion['Luis'].iloc[:mitad].mean()
segunda_parte_luis = evolucion['Luis'].iloc[mitad:].mean()
primera_parte_byron = evolucion['Byron'].iloc[:mitad].mean()
segunda_parte_byron = evolucion['Byron'].iloc[mitad:].mean()

print(f"\nTendencia (Primera vs Segunda Mitad de la Semana):")
print(f"  Luis:  {primera_parte_luis:.1f} → {segunda_parte_luis:.1f} ppm ({segunda_parte_luis - primera_parte_luis:+.1f})")
print(f"  Byron: {primera_parte_byron:.1f} → {segunda_parte_byron:.1f} ppm ({segunda_parte_byron - primera_parte_byron:+.1f})")

# ==========================================
# ANÁLISIS DE VARIABILIDAD
# ==========================================
print("\n" + "="*70)
print("4. ANÁLISIS DE VARIABILIDAD")
print("="*70)

# Variabilidad intra-día (diferencia entre las 2 mediciones del mismo día)
variabilidad = df.groupby('Fecha').agg({
    'Palabras por Minuto (Luis)': lambda x: abs(x.iloc[0] - x.iloc[1]) if len(x) == 2 else 0,
    'Palabras por Minuto (Byron)': lambda x: abs(x.iloc[0] - x.iloc[1]) if len(x) == 2 else 0
})

print(f"\nVariabilidad promedio entre mediciones del mismo día:")
print(f"  Luis:  {variabilidad['Palabras por Minuto (Luis)'].mean():.1f} ppm")
print(f"  Byron: {variabilidad['Palabras por Minuto (Byron)'].mean():.1f} ppm")

# ==========================================
# COMPARACIÓN ENTRE LECTORES
# ==========================================
print("\n" + "="*70)
print("5. COMPARACIÓN ENTRE LECTORES")
print("="*70)

luis_gana = (df['Palabras por Minuto (Luis)'] > df['Palabras por Minuto (Byron)']).sum()
byron_gana = (df['Palabras por Minuto (Byron)'] > df['Palabras por Minuto (Luis)']).sum()
empates = (df['Palabras por Minuto (Luis)'] == df['Palabras por Minuto (Byron)']).sum()

print(f"\nMediciones donde cada uno fue más rápido:")
print(f"  Luis:  {luis_gana} mediciones ({luis_gana/len(df)*100:.1f}%)")
print(f"  Byron: {byron_gana} mediciones ({byron_gana/len(df)*100:.1f}%)")
print(f"  Empates: {empates} mediciones")

diferencia_promedio = (df['Palabras por Minuto (Luis)'] - df['Palabras por Minuto (Byron)']).mean()
print(f"\nDiferencia promedio: {diferencia_promedio:+.1f} ppm")
print(f"  {'Luis lee más rápido en promedio' if diferencia_promedio > 0 else 'Byron lee más rápido en promedio'}")

# ==========================================
# MEJORES Y PEORES RENDIMIENTOS
# ==========================================
print("\n" + "="*70)
print("6. MEJORES Y PEORES RENDIMIENTOS")
print("="*70)

mejor_luis = df.loc[df['Palabras por Minuto (Luis)'].idxmax()]
mejor_byron = df.loc[df['Palabras por Minuto (Byron)'].idxmax()]
peor_luis = df.loc[df['Palabras por Minuto (Luis)'].idxmin()]
peor_byron = df.loc[df['Palabras por Minuto (Byron)'].idxmin()]

print(f"\nMejor rendimiento de Luis:")
print(f"  {mejor_luis['Palabras por Minuto (Luis)']} ppm - {mejor_luis['Lectura']} ({mejor_luis['Fecha'].strftime('%d/%m/%Y')})")

print(f"\nMejor rendimiento de Byron:")
print(f"  {mejor_byron['Palabras por Minuto (Byron)']} ppm - {mejor_byron['Lectura']} ({mejor_byron['Fecha'].strftime('%d/%m/%Y')})")

print(f"\nPeor rendimiento de Luis:")
print(f"  {peor_luis['Palabras por Minuto (Luis)']} ppm - {peor_luis['Lectura']} ({peor_luis['Fecha'].strftime('%d/%m/%Y')})")

print(f"\nPeor rendimiento de Byron:")
print(f"  {peor_byron['Palabras por Minuto (Byron)']} ppm - {peor_byron['Lectura']} ({peor_byron['Fecha'].strftime('%d/%m/%Y')})")

# ==========================================
# RESUMEN EJECUTIVO
# ==========================================
print("\n" + "="*70)
print("7. RESUMEN EJECUTIVO - MÉTODO 2")
print("="*70)

print(f"""
Velocidad promedio general:
  • Luis:  {df['Palabras por Minuto (Luis)'].mean():.1f} ± {df['Palabras por Minuto (Luis)'].std():.1f} ppm
  • Byron: {df['Palabras por Minuto (Byron)'].mean():.1f} ± {df['Palabras por Minuto (Byron)'].std():.1f} ppm

Lectura más rápida para ambos:
  • {lecturas_stats[['Luis_Promedio', 'Byron_Promedio']].mean(axis=1).idxmax()}

Lectura más lenta para ambos:
  • {lecturas_stats[['Luis_Promedio', 'Byron_Promedio']].mean(axis=1).idxmin()}

Tendencia durante la semana:
  • {'Ambos lectores mejoraron en la segunda mitad de la semana' if segunda_parte_luis > primera_parte_luis and segunda_parte_byron > primera_parte_byron else 'El rendimiento varió durante la semana'}

Observación:
  • Estas mediciones se realizaron escuchando música, lo que puede haber
    afectado la velocidad y concentración de lectura.
""")

print("="*70)

# ==========================================
# GRÁFICOS DE CONTROL DE SHEWHART
# ==========================================
print("\n" + "="*70)
print("8. GRÁFICOS DE CONTROL DE SHEWHART")
print("="*70)

# Calcular X-bar y R por fecha
xbar_luis = df.groupby('Fecha')['Palabras por Minuto (Luis)'].mean()
xbar_byron = df.groupby('Fecha')['Palabras por Minuto (Byron)'].mean()

r_luis = df.groupby('Fecha')['Palabras por Minuto (Luis)'].apply(lambda x: x.max() - x.min())
r_byron = df.groupby('Fecha')['Palabras por Minuto (Byron)'].apply(lambda x: x.max() - x.min())

# Constantes para n=2 (2 mediciones por día)
A2 = 1.880
D3 = 0
D4 = 3.267

# Calcular límites de control para Luis
xbar_mean_luis = xbar_luis.mean()
r_mean_luis = r_luis.mean()
UCL_xbar_luis = xbar_mean_luis + A2 * r_mean_luis
LCL_xbar_luis = xbar_mean_luis - A2 * r_mean_luis
UCL_r_luis = D4 * r_mean_luis
LCL_r_luis = D3 * r_mean_luis

# Calcular límites de control para Byron
xbar_mean_byron = xbar_byron.mean()
r_mean_byron = r_byron.mean()
UCL_xbar_byron = xbar_mean_byron + A2 * r_mean_byron
LCL_xbar_byron = xbar_mean_byron - A2 * r_mean_byron
UCL_r_byron = D4 * r_mean_byron
LCL_r_byron = D3 * r_mean_byron

print("\nLímites de Control - Luis:")
print(f"  X-bar: {xbar_mean_luis:.2f} ppm (UCL: {UCL_xbar_luis:.2f}, LCL: {LCL_xbar_luis:.2f})")
print(f"  R:     {r_mean_luis:.2f} ppm (UCL: {UCL_r_luis:.2f})")

print("\nLímites de Control - Byron:")
print(f"  X-bar: {xbar_mean_byron:.2f} ppm (UCL: {UCL_xbar_byron:.2f}, LCL: {LCL_xbar_byron:.2f})")
print(f"  R:     {r_mean_byron:.2f} ppm (UCL: {UCL_r_byron:.2f})")

# Detectar puntos fuera de control
out_luis_xbar = []
out_byron_xbar = []
out_luis_r = []
out_byron_r = []

for fecha, valor in xbar_luis.items():
    if valor > UCL_xbar_luis or valor < LCL_xbar_luis:
        out_luis_xbar.append((fecha, valor))

for fecha, valor in xbar_byron.items():
    if valor > UCL_xbar_byron or valor < LCL_xbar_byron:
        out_byron_xbar.append((fecha, valor))

for fecha, valor in r_luis.items():
    if valor > UCL_r_luis:
        out_luis_r.append((fecha, valor))

for fecha, valor in r_byron.items():
    if valor > UCL_r_byron:
        out_byron_r.append((fecha, valor))

print(f"\nPuntos fuera de control:")
print(f"  Luis X-bar:  {len(out_luis_xbar)} puntos")
print(f"  Byron X-bar: {len(out_byron_xbar)} puntos")
print(f"  Luis R:      {len(out_luis_r)} puntos")
print(f"  Byron R:     {len(out_byron_r)} puntos")

if len(out_luis_xbar) + len(out_byron_xbar) + len(out_luis_r) + len(out_byron_r) == 0:
    print("\n✅ Proceso bajo control estadístico")
else:
    print("\n⚠️  Se detectaron causas especiales de variación")

# ==========================================
# CREAR GRÁFICOS
# ==========================================
print("\nGenerando gráficos de control...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Gráficos de Control de Shewhart - Método 2', fontsize=16, fontweight='bold')

# X-bar chart para Luis
ax1 = axes[0, 0]
x_pos = range(len(xbar_luis))
ax1.plot(x_pos, xbar_luis.values, 'o-', color='blue', linewidth=2, markersize=8, label='X-bar')
ax1.axhline(xbar_mean_luis, color='green', linestyle='-', linewidth=2, label=f'Media ({xbar_mean_luis:.1f})')
ax1.axhline(UCL_xbar_luis, color='red', linestyle='--', linewidth=2, label=f'UCL ({UCL_xbar_luis:.1f})')
ax1.axhline(LCL_xbar_luis, color='red', linestyle='--', linewidth=2, label=f'LCL ({LCL_xbar_luis:.1f})')

# Marcar puntos fuera de control
for fecha, valor in out_luis_xbar:
    idx = list(xbar_luis.index).index(fecha)
    ax1.plot(idx, valor, 'ro', markersize=12, markerfacecolor='none', markeredgewidth=2)

ax1.set_title('X-bar Chart - Luis', fontsize=12, fontweight='bold')
ax1.set_xlabel('Día de medición')
ax1.set_ylabel('Promedio de Palabras por Minuto')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{i+1}' for i in x_pos])

# R chart para Luis
ax2 = axes[0, 1]
ax2.plot(x_pos, r_luis.values, 'o-', color='blue', linewidth=2, markersize=8, label='R')
ax2.axhline(r_mean_luis, color='green', linestyle='-', linewidth=2, label=f'R Media ({r_mean_luis:.1f})')
ax2.axhline(UCL_r_luis, color='red', linestyle='--', linewidth=2, label=f'UCL ({UCL_r_luis:.1f})')

# Marcar puntos fuera de control
for fecha, valor in out_luis_r:
    idx = list(r_luis.index).index(fecha)
    ax2.plot(idx, valor, 'ro', markersize=12, markerfacecolor='none', markeredgewidth=2)

ax2.set_title('R Chart - Luis', fontsize=12, fontweight='bold')
ax2.set_xlabel('Día de medición')
ax2.set_ylabel('Rango (Max - Min)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{i+1}' for i in x_pos])

# X-bar chart para Byron
ax3 = axes[1, 0]
ax3.plot(x_pos, xbar_byron.values, 'o-', color='orange', linewidth=2, markersize=8, label='X-bar')
ax3.axhline(xbar_mean_byron, color='green', linestyle='-', linewidth=2, label=f'Media ({xbar_mean_byron:.1f})')
ax3.axhline(UCL_xbar_byron, color='red', linestyle='--', linewidth=2, label=f'UCL ({UCL_xbar_byron:.1f})')
ax3.axhline(LCL_xbar_byron, color='red', linestyle='--', linewidth=2, label=f'LCL ({LCL_xbar_byron:.1f})')

# Marcar puntos fuera de control
for fecha, valor in out_byron_xbar:
    idx = list(xbar_byron.index).index(fecha)
    ax3.plot(idx, valor, 'ro', markersize=12, markerfacecolor='none', markeredgewidth=2)

ax3.set_title('X-bar Chart - Byron', fontsize=12, fontweight='bold')
ax3.set_xlabel('Día de medición')
ax3.set_ylabel('Promedio de Palabras por Minuto')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'{i+1}' for i in x_pos])

# R chart para Byron
ax4 = axes[1, 1]
ax4.plot(x_pos, r_byron.values, 'o-', color='orange', linewidth=2, markersize=8, label='R')
ax4.axhline(r_mean_byron, color='green', linestyle='-', linewidth=2, label=f'R Media ({r_mean_byron:.1f})')
ax4.axhline(UCL_r_byron, color='red', linestyle='--', linewidth=2, label=f'UCL ({UCL_r_byron:.1f})')

# Marcar puntos fuera de control
for fecha, valor in out_byron_r:
    idx = list(r_byron.index).index(fecha)
    ax4.plot(idx, valor, 'ro', markersize=12, markerfacecolor='none', markeredgewidth=2)

ax4.set_title('R Chart - Byron', fontsize=12, fontweight='bold')
ax4.set_xlabel('Día de medición')
ax4.set_ylabel('Rango (Max - Min)')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'{i+1}' for i in x_pos])

plt.tight_layout()
plt.savefig('method_2_shewhart_charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)