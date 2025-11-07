import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Cargar los tres datasets
df1 = pd.read_csv('datos1.csv')
df2 = pd.read_csv('datos2.csv')
df3 = pd.read_csv('datos3.csv')

# Limpiar y preparar datos
for df in [df1, df2, df3]:
    df.columns = df.columns.str.strip()
    df['Lectura'] = df['Lectura'].str.strip()
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')

# Agregar identificador de método
df1['Método'] = 'Baseline'
df2['Método'] = 'Método 2'
df3['Método'] = 'Método 3'

# Combinar todos los datasets
df_all = pd.concat([df1, df2, df3], ignore_index=True)
df_all = df_all.sort_values('Fecha').reset_index(drop=True)

print("="*80)
print("GRÁFICOS DE CONTROL DE SHEWHART - ANÁLISIS DE VARIACIÓN")
print("="*80)

# ==========================================
# FUNCIÓN PARA CALCULAR X-BAR Y R
# ==========================================
def calculate_xbar_r(data, subgroup_col='Fecha'):
    """
    Calcula X-bar (promedio) y R (rango) para cada subgrupo
    """
    grouped = data.groupby(subgroup_col)
    
    xbar = grouped.mean()
    r = grouped.apply(lambda x: x.max() - x.min())
    n = grouped.size()
    
    return xbar, r, n

# ==========================================
# CALCULAR LÍMITES DE CONTROL PARA BASELINE
# ==========================================
def calculate_control_limits(xbar, r, n):
    """
    Calcula límites de control para gráficos X-bar y R
    Constantes para n=2 (2 mediciones por día)
    """
    # Constantes para n=2
    A2 = 1.880
    D3 = 0
    D4 = 3.267
    
    # X-bar chart
    xbar_mean = xbar.mean()
    r_mean = r.mean()
    
    UCL_xbar = xbar_mean + A2 * r_mean
    LCL_xbar = xbar_mean - A2 * r_mean
    
    # R chart
    UCL_r = D4 * r_mean
    LCL_r = D3 * r_mean
    
    return {
        'xbar_mean': xbar_mean,
        'UCL_xbar': UCL_xbar,
        'LCL_xbar': LCL_xbar,
        'r_mean': r_mean,
        'UCL_r': UCL_r,
        'LCL_r': LCL_r
    }

# ==========================================
# ANÁLISIS PARA LUIS
# ==========================================
print("\n" + "="*80)
print("ANÁLISIS DE CONTROL PARA LUIS")
print("="*80)

# Calcular X-bar y R por fecha para cada método
xbar_luis_1, r_luis_1, n_luis_1 = calculate_xbar_r(df1[['Fecha', 'Palabras por Minuto (Luis)']], 'Fecha')
xbar_luis_2, r_luis_2, n_luis_2 = calculate_xbar_r(df2[['Fecha', 'Palabras por Minuto (Luis)']], 'Fecha')
xbar_luis_3, r_luis_3, n_luis_3 = calculate_xbar_r(df3[['Fecha', 'Palabras por Minuto (Luis)']], 'Fecha')

# Usar baseline para establecer límites de control
limits_luis = calculate_control_limits(
    xbar_luis_1['Palabras por Minuto (Luis)'], 
    r_luis_1['Palabras por Minuto (Luis)'], 
    2
)

print("\nLÍMITES DE CONTROL BASADOS EN BASELINE (Luis):")
print(f"  X-bar promedio: {limits_luis['xbar_mean']:.2f} ppm")
print(f"  UCL (X-bar):    {limits_luis['UCL_xbar']:.2f} ppm")
print(f"  LCL (X-bar):    {limits_luis['LCL_xbar']:.2f} ppm")
print(f"  R promedio:     {limits_luis['r_mean']:.2f} ppm")
print(f"  UCL (R):        {limits_luis['UCL_r']:.2f} ppm")

# Combinar todos los X-bar para Luis
xbar_luis_all = pd.concat([
    xbar_luis_1['Palabras por Minuto (Luis)'].to_frame().assign(Método='Baseline'),
    xbar_luis_2['Palabras por Minuto (Luis)'].to_frame().assign(Método='Método 2'),
    xbar_luis_3['Palabras por Minuto (Luis)'].to_frame().assign(Método='Método 3')
])

# Detectar puntos fuera de control
out_of_control_luis = []
for idx, row in xbar_luis_all.iterrows():
    value = row['Palabras por Minuto (Luis)']
    metodo = row['Método']
    if value > limits_luis['UCL_xbar'] or value < limits_luis['LCL_xbar']:
        out_of_control_luis.append({
            'Fecha': idx,
            'Valor': value,
            'Método': metodo,
            'Tipo': 'Por encima de UCL' if value > limits_luis['UCL_xbar'] else 'Por debajo de LCL'
        })

print(f"\n🔴 PUNTOS FUERA DE CONTROL (Luis): {len(out_of_control_luis)}")
for point in out_of_control_luis:
    print(f"  {point['Fecha'].strftime('%d/%m/%Y')} - {point['Método']}: {point['Valor']:.1f} ppm ({point['Tipo']})")

# ==========================================
# ANÁLISIS PARA BYRON
# ==========================================
print("\n" + "="*80)
print("ANÁLISIS DE CONTROL PARA BYRON")
print("="*80)

# Calcular X-bar y R por fecha para cada método
xbar_byron_1, r_byron_1, n_byron_1 = calculate_xbar_r(df1[['Fecha', 'Palabras por Minuto (Byron)']], 'Fecha')
xbar_byron_2, r_byron_2, n_byron_2 = calculate_xbar_r(df2[['Fecha', 'Palabras por Minuto (Byron)']], 'Fecha')
xbar_byron_3, r_byron_3, n_byron_3 = calculate_xbar_r(df3[['Fecha', 'Palabras por Minuto (Byron)']], 'Fecha')

# Usar baseline para establecer límites de control
limits_byron = calculate_control_limits(
    xbar_byron_1['Palabras por Minuto (Byron)'], 
    r_byron_1['Palabras por Minuto (Byron)'], 
    2
)

print("\nLÍMITES DE CONTROL BASADOS EN BASELINE (Byron):")
print(f"  X-bar promedio: {limits_byron['xbar_mean']:.2f} ppm")
print(f"  UCL (X-bar):    {limits_byron['UCL_xbar']:.2f} ppm")
print(f"  LCL (X-bar):    {limits_byron['LCL_xbar']:.2f} ppm")
print(f"  R promedio:     {limits_byron['r_mean']:.2f} ppm")
print(f"  UCL (R):        {limits_byron['UCL_r']:.2f} ppm")

# Combinar todos los X-bar para Byron
xbar_byron_all = pd.concat([
    xbar_byron_1['Palabras por Minuto (Byron)'].to_frame().assign(Método='Baseline'),
    xbar_byron_2['Palabras por Minuto (Byron)'].to_frame().assign(Método='Método 2'),
    xbar_byron_3['Palabras por Minuto (Byron)'].to_frame().assign(Método='Método 3')
])

# Detectar puntos fuera de control
out_of_control_byron = []
for idx, row in xbar_byron_all.iterrows():
    value = row['Palabras por Minuto (Byron)']
    metodo = row['Método']
    if value > limits_byron['UCL_xbar'] or value < limits_byron['LCL_xbar']:
        out_of_control_byron.append({
            'Fecha': idx,
            'Valor': value,
            'Método': metodo,
            'Tipo': 'Por encima de UCL' if value > limits_byron['UCL_xbar'] else 'Por debajo de LCL'
        })

print(f"\n🔴 PUNTOS FUERA DE CONTROL (Byron): {len(out_of_control_byron)}")
for point in out_of_control_byron:
    print(f"  {point['Fecha'].strftime('%d/%m/%Y')} - {point['Método']}: {point['Valor']:.1f} ppm ({point['Tipo']})")

# ==========================================
# CREAR GRÁFICOS
# ==========================================
print("\n" + "="*80)
print("GENERANDO GRÁFICOS DE CONTROL...")
print("="*80)

# Figura con 4 subplots (X-bar y R para ambos)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Gráficos de Control de Shewhart - X-bar y R Charts', fontsize=16, fontweight='bold')

# Preparar datos para plotting
dates_all = pd.concat([
    xbar_luis_1.index.to_series().reset_index(drop=True),
    xbar_luis_2.index.to_series().reset_index(drop=True),
    xbar_luis_3.index.to_series().reset_index(drop=True)
])

# X-bar chart para Luis
ax1 = axes[0, 0]
x_pos = list(range(len(xbar_luis_all)))

# Dividir por método para colores
baseline_mask = xbar_luis_all['Método'] == 'Baseline'
musica_mask = xbar_luis_all['Método'] == 'Método 2'
optimo_mask = xbar_luis_all['Método'] == 'Método 3'

ax1.plot(np.array(x_pos)[baseline_mask], xbar_luis_all[baseline_mask]['Palabras por Minuto (Luis)'], 
         'o-', color='blue', label='Baseline', linewidth=2, markersize=6)
ax1.plot(np.array(x_pos)[musica_mask], xbar_luis_all[musica_mask]['Palabras por Minuto (Luis)'], 
         'o-', color='orange', label='Método 2', linewidth=2, markersize=6)
ax1.plot(np.array(x_pos)[optimo_mask], xbar_luis_all[optimo_mask]['Palabras por Minuto (Luis)'], 
         'o-', color='green', label='Método 3', linewidth=2, markersize=6)

ax1.axhline(limits_luis['xbar_mean'], color='black', linestyle='-', linewidth=2, label='Media')
ax1.axhline(limits_luis['UCL_xbar'], color='red', linestyle='--', linewidth=2, label='UCL')
ax1.axhline(limits_luis['LCL_xbar'], color='red', linestyle='--', linewidth=2, label='LCL')

# Marcar separación entre métodos
ax1.axvline(len(xbar_luis_1) - 0.5, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(len(xbar_luis_1) + len(xbar_luis_2) - 0.5, color='gray', linestyle=':', alpha=0.5)

ax1.set_title('X-bar Chart - Luis', fontsize=12, fontweight='bold')
ax1.set_xlabel('Día de medición')
ax1.set_ylabel('Promedio de Palabras por Minuto')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# R chart para Luis
ax2 = axes[0, 1]
r_luis_all = pd.concat([
    r_luis_1['Palabras por Minuto (Luis)'].to_frame().assign(Método='Baseline'),
    r_luis_2['Palabras por Minuto (Luis)'].to_frame().assign(Método='Método 2'),
    r_luis_3['Palabras por Minuto (Luis)'].to_frame().assign(Método='Método 3')
])

baseline_mask_r = r_luis_all['Método'] == 'Baseline'
musica_mask_r = r_luis_all['Método'] == 'Método 2'
optimo_mask_r = r_luis_all['Método'] == 'Método 3'

ax2.plot(np.array(x_pos)[baseline_mask_r], r_luis_all[baseline_mask_r]['Palabras por Minuto (Luis)'], 
         'o-', color='blue', label='Baseline', linewidth=2, markersize=6)
ax2.plot(np.array(x_pos)[musica_mask_r], r_luis_all[musica_mask_r]['Palabras por Minuto (Luis)'], 
         'o-', color='orange', label='Método 2', linewidth=2, markersize=6)
ax2.plot(np.array(x_pos)[optimo_mask_r], r_luis_all[optimo_mask_r]['Palabras por Minuto (Luis)'], 
         'o-', color='green', label='Método 3', linewidth=2, markersize=6)

ax2.axhline(limits_luis['r_mean'], color='black', linestyle='-', linewidth=2, label='R Media')
ax2.axhline(limits_luis['UCL_r'], color='red', linestyle='--', linewidth=2, label='UCL')

# Marcar separación entre métodos
ax2.axvline(len(r_luis_1) - 0.5, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(len(r_luis_1) + len(r_luis_2) - 0.5, color='gray', linestyle=':', alpha=0.5)

ax2.set_title('R Chart - Luis', fontsize=12, fontweight='bold')
ax2.set_xlabel('Día de medición')
ax2.set_ylabel('Rango (Max - Min)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# X-bar chart para Byron
ax3 = axes[1, 0]
ax3.plot(np.array(x_pos)[baseline_mask], xbar_byron_all[baseline_mask]['Palabras por Minuto (Byron)'], 
         'o-', color='blue', label='Baseline', linewidth=2, markersize=6)
ax3.plot(np.array(x_pos)[musica_mask], xbar_byron_all[musica_mask]['Palabras por Minuto (Byron)'], 
         'o-', color='orange', label='Método 2', linewidth=2, markersize=6)
ax3.plot(np.array(x_pos)[optimo_mask], xbar_byron_all[optimo_mask]['Palabras por Minuto (Byron)'], 
         'o-', color='green', label='Método 3', linewidth=2, markersize=6)

ax3.axhline(limits_byron['xbar_mean'], color='black', linestyle='-', linewidth=2, label='Media')
ax3.axhline(limits_byron['UCL_xbar'], color='red', linestyle='--', linewidth=2, label='UCL')
ax3.axhline(limits_byron['LCL_xbar'], color='red', linestyle='--', linewidth=2, label='LCL')

# Marcar separación entre métodos
ax3.axvline(len(xbar_byron_1) - 0.5, color='gray', linestyle=':', alpha=0.5)
ax3.axvline(len(xbar_byron_1) + len(xbar_byron_2) - 0.5, color='gray', linestyle=':', alpha=0.5)

ax3.set_title('X-bar Chart - Byron', fontsize=12, fontweight='bold')
ax3.set_xlabel('Día de medición')
ax3.set_ylabel('Promedio de Palabras por Minuto')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# R chart para Byron
ax4 = axes[1, 1]
r_byron_all = pd.concat([
    r_byron_1['Palabras por Minuto (Byron)'].to_frame().assign(Método='Baseline'),
    r_byron_2['Palabras por Minuto (Byron)'].to_frame().assign(Método='Método 2'),
    r_byron_3['Palabras por Minuto (Byron)'].to_frame().assign(Método='Método 3')
])

ax4.plot(np.array(x_pos)[baseline_mask_r], r_byron_all[baseline_mask_r]['Palabras por Minuto (Byron)'], 
         'o-', color='blue', label='Baseline', linewidth=2, markersize=6)
ax4.plot(np.array(x_pos)[musica_mask_r], r_byron_all[musica_mask_r]['Palabras por Minuto (Byron)'], 
         'o-', color='orange', label='Método 2', linewidth=2, markersize=6)
ax4.plot(np.array(x_pos)[optimo_mask_r], r_byron_all[optimo_mask_r]['Palabras por Minuto (Byron)'], 
         'o-', color='green', label='Método 3', linewidth=2, markersize=6)

ax4.axhline(limits_byron['r_mean'], color='black', linestyle='-', linewidth=2, label='R Media')
ax4.axhline(limits_byron['UCL_r'], color='red', linestyle='--', linewidth=2, label='UCL')

# Marcar separación entre métodos
ax4.axvline(len(r_byron_1) - 0.5, color='gray', linestyle=':', alpha=0.5)
ax4.axvline(len(r_byron_1) + len(r_byron_2) - 0.5, color='gray', linestyle=':', alpha=0.5)

ax4.set_title('R Chart - Byron', fontsize=12, fontweight='bold')
ax4.set_xlabel('Día de medición')
ax4.set_ylabel('Rango (Max - Min)')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shewhart_control_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# EVALUACIÓN DE CAUSAS ESPECIALES
# ==========================================
print("\n" + "="*80)
print("EVALUACIÓN DE CAUSAS ESPECIALES DE VARIACIÓN")
print("="*80)

print(f"""
📊 RESUMEN DE ANÁLISIS:

Luis:
  • Puntos fuera de control: {len(out_of_control_luis)}
  • {'✅ Proceso bajo control estadístico' if len(out_of_control_luis) == 0 else '⚠️  Causas especiales detectadas'}

Byron:
  • Puntos fuera de control: {len(out_of_control_byron)}
  • {'✅ Proceso bajo control estadístico' if len(out_of_control_byron) == 0 else '⚠️  Causas especiales detectadas'}

🔍 INTERPRETACIÓN:
""")

# Analizar tendencias por método
for metodo in ['Baseline', 'Método 2', 'Método 3']:
    print(f"\n{metodo}:")
    luis_puntos = [p for p in out_of_control_luis if p['Método'] == metodo]
    byron_puntos = [p for p in out_of_control_byron if p['Método'] == metodo]
    
    if len(luis_puntos) > 0 or len(byron_puntos) > 0:
        print(f"  ⚠️  Variación especial detectada")
        if len(luis_puntos) > 0:
            print(f"     Luis: {len(luis_puntos)} puntos fuera de control")
        if len(byron_puntos) > 0:
            print(f"     Byron: {len(byron_puntos)} puntos fuera de control")
    else:
        print(f"  ✅ Sin variación especial")

print(f"""
💡 CONCLUSIONES:

1. {'Los cambios de método SÍ causaron variación especial' if (len(out_of_control_luis) > 0 or len(out_of_control_byron) > 0) else 'Los cambios NO causaron variación especial significativa'}

2. {'El método Método 3 muestra mejoras efectivas (puntos por encima de UCL)' if any(p['Método'] == 'Método 3' and p['Tipo'] == 'Por encima de UCL' for p in out_of_control_luis + out_of_control_byron) else 'No se detectaron mejoras significativas fuera de los límites'}

3. {'La música afectó negativamente el rendimiento (puntos por debajo de LCL)' if any(p['Método'] == 'Método 2' and p['Tipo'] == 'Por debajo de LCL' for p in out_of_control_luis + out_of_control_byron) else 'La música no causó deterioro significativo'}

4. Los cambios implementados {'son efectivos y deben mantenerse' if any(p['Método'] == 'Método 3' and p['Tipo'] == 'Por encima de UCL' for p in out_of_control_luis + out_of_control_byron) else 'requieren evaluación adicional'}
""")

print("="*80)