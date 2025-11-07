import pandas as pd
import numpy as np
from datetime import datetime

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

print("="*80)
print("ANÁLISIS COMPARATIVO: IMPACTO DE MÉTODOS EN VELOCIDAD DE LECTURA")
print("="*80)

print("\n📊 RESUMEN DE DATOS:")
print(f"  • Baseline:     {len(df1)} mediciones ({df1['Fecha'].min().strftime('%d/%m/%Y')} - {df1['Fecha'].max().strftime('%d/%m/%Y')})")
print(f"  • Método 2:   {len(df2)} mediciones ({df2['Fecha'].min().strftime('%d/%m/%Y')} - {df2['Fecha'].max().strftime('%d/%m/%Y')})")
print(f"  • Método 3:       {len(df3)} mediciones ({df3['Fecha'].min().strftime('%d/%m/%Y')} - {df3['Fecha'].max().strftime('%d/%m/%Y')})")

# ==========================================
# COMPARACIÓN DE PROMEDIOS GENERALES
# ==========================================
print("\n" + "="*80)
print("1. COMPARACIÓN DE VELOCIDAD PROMEDIO POR MÉTODO")
print("="*80)

metodos_stats = pd.DataFrame({
    'Luis': [
        df1['Palabras por Minuto (Luis)'].mean(),
        df2['Palabras por Minuto (Luis)'].mean(),
        df3['Palabras por Minuto (Luis)'].mean()
    ],
    'Byron': [
        df1['Palabras por Minuto (Byron)'].mean(),
        df2['Palabras por Minuto (Byron)'].mean(),
        df3['Palabras por Minuto (Byron)'].mean()
    ]
}, index=['Baseline', 'Método 2', 'Método 3'])

metodos_stats['Promedio_Ambos'] = metodos_stats.mean(axis=1)
metodos_stats['Desv_Est_Luis'] = [
    df1['Palabras por Minuto (Luis)'].std(),
    df2['Palabras por Minuto (Luis)'].std(),
    df3['Palabras por Minuto (Luis)'].std()
]
metodos_stats['Desv_Est_Byron'] = [
    df1['Palabras por Minuto (Byron)'].std(),
    df2['Palabras por Minuto (Byron)'].std(),
    df3['Palabras por Minuto (Byron)'].std()
]

print("\n", metodos_stats.round(2))

# Calcular mejoras porcentuales
baseline_luis = df1['Palabras por Minuto (Luis)'].mean()
baseline_byron = df1['Palabras por Minuto (Byron)'].mean()

print("\n📈 MEJORA RESPECTO AL BASELINE:")
print(f"\nLuis:")
print(f"  Método 2:    {((df2['Palabras por Minuto (Luis)'].mean() - baseline_luis) / baseline_luis * 100):+.2f}%")
print(f"  Método 3: {((df3['Palabras por Minuto (Luis)'].mean() - baseline_luis) / baseline_luis * 100):+.2f}%")

print(f"\nByron:")
print(f"  Método 2:    {((df2['Palabras por Minuto (Byron)'].mean() - baseline_byron) / baseline_byron * 100):+.2f}%")
print(f"  Método 3: {((df3['Palabras por Minuto (Byron)'].mean() - baseline_byron) / baseline_byron * 100):+.2f}%")

# ==========================================
# COMPARACIÓN DE CONSISTENCIA
# ==========================================
print("\n" + "="*80)
print("2. ANÁLISIS DE CONSISTENCIA (Coeficiente de Variación)")
print("="*80)

cv_data = pd.DataFrame({
    'Luis_CV': [
        (df1['Palabras por Minuto (Luis)'].std() / df1['Palabras por Minuto (Luis)'].mean()) * 100,
        (df2['Palabras por Minuto (Luis)'].std() / df2['Palabras por Minuto (Luis)'].mean()) * 100,
        (df3['Palabras por Minuto (Luis)'].std() / df3['Palabras por Minuto (Luis)'].mean()) * 100
    ],
    'Byron_CV': [
        (df1['Palabras por Minuto (Byron)'].std() / df1['Palabras por Minuto (Byron)'].mean()) * 100,
        (df2['Palabras por Minuto (Byron)'].std() / df2['Palabras por Minuto (Byron)'].mean()) * 100,
        (df3['Palabras por Minuto (Byron)'].std() / df3['Palabras por Minuto (Byron)'].mean()) * 100
    ]
}, index=['Baseline', 'Método 2', 'Método 3'])

print("\n", cv_data.round(2))
print("\n⚠️  Nota: Menor CV = Mayor consistencia")

# ==========================================
# COMPARACIÓN POR TIPO DE LECTURA
# ==========================================
print("\n" + "="*80)
print("3. COMPARACIÓN POR TIPO DE LECTURA")
print("="*80)

for lectura in df1['Lectura'].unique():
    print(f"\n📖 {lectura}")
    print("-" * 70)
    
    stats_lectura = pd.DataFrame({
        'Luis_Baseline': [df1[df1['Lectura'] == lectura]['Palabras por Minuto (Luis)'].mean()],
        'Luis_Método_2': [df2[df2['Lectura'] == lectura]['Palabras por Minuto (Luis)'].mean()],
        'Luis_Método_3': [df3[df3['Lectura'] == lectura]['Palabras por Minuto (Luis)'].mean()],
        'Byron_Baseline': [df1[df1['Lectura'] == lectura]['Palabras por Minuto (Byron)'].mean()],
        'Byron_Método_2': [df2[df2['Lectura'] == lectura]['Palabras por Minuto (Byron)'].mean()],
        'Byron_Método_3': [df3[df3['Lectura'] == lectura]['Palabras por Minuto (Byron)'].mean()]
    })
    
    print(stats_lectura.round(1).to_string(index=False))

# ==========================================
# ANÁLISIS DE RANGOS Y EXTREMOS
# ==========================================
print("\n" + "="*80)
print("4. COMPARACIÓN DE RANGOS (Min - Max)")
print("="*80)

rangos = pd.DataFrame({
    'Luis_Min': [
        df1['Palabras por Minuto (Luis)'].min(),
        df2['Palabras por Minuto (Luis)'].min(),
        df3['Palabras por Minuto (Luis)'].min()
    ],
    'Luis_Max': [
        df1['Palabras por Minuto (Luis)'].max(),
        df2['Palabras por Minuto (Luis)'].max(),
        df3['Palabras por Minuto (Luis)'].max()
    ],
    'Byron_Min': [
        df1['Palabras por Minuto (Byron)'].min(),
        df2['Palabras por Minuto (Byron)'].min(),
        df3['Palabras por Minuto (Byron)'].min()
    ],
    'Byron_Max': [
        df1['Palabras por Minuto (Byron)'].max(),
        df2['Palabras por Minuto (Byron)'].max(),
        df3['Palabras por Minuto (Byron)'].max()
    ]
}, index=['Baseline', 'Método 2', 'Método 3'])

rangos['Luis_Rango'] = rangos['Luis_Max'] - rangos['Luis_Min']
rangos['Byron_Rango'] = rangos['Byron_Max'] - rangos['Byron_Min']

print("\n", rangos)

# ==========================================
# VENTAJA COMPETITIVA
# ==========================================
print("\n" + "="*80)
print("5. ANÁLISIS DE VENTAJA COMPETITIVA")
print("="*80)

for i, (df, nombre) in enumerate([(df1, 'Baseline'), (df2, 'Método 2'), (df3, 'Método 3')]):
    luis_gana = (df['Palabras por Minuto (Luis)'] > df['Palabras por Minuto (Byron)']).sum()
    byron_gana = (df['Palabras por Minuto (Byron)'] > df['Palabras por Minuto (Luis)']).sum()
    total = len(df)
    
    print(f"\n{nombre}:")
    print(f"  Luis ganó:  {luis_gana}/{total} mediciones ({luis_gana/total*100:.1f}%)")
    print(f"  Byron ganó: {byron_gana}/{total} mediciones ({byron_gana/total*100:.1f}%)")

# ==========================================
# RESUMEN EJECUTIVO Y RECOMENDACIONES
# ==========================================
print("\n" + "="*80)
print("6. RESUMEN EJECUTIVO Y CONCLUSIONES")
print("="*80)

mejor_metodo_luis = metodos_stats['Luis'].idxmax()
mejor_metodo_byron = metodos_stats['Byron'].idxmax()
mejor_metodo_general = metodos_stats['Promedio_Ambos'].idxmax()

mejora_luis = ((metodos_stats.loc[mejor_metodo_luis, 'Luis'] - metodos_stats.loc['Baseline', 'Luis']) / metodos_stats.loc['Baseline', 'Luis'] * 100)
mejora_byron = ((metodos_stats.loc[mejor_metodo_byron, 'Byron'] - metodos_stats.loc['Baseline', 'Byron']) / metodos_stats.loc['Baseline', 'Byron'] * 100)

print(f"""
🎯 MÉTODO MÁS EFECTIVO:
  • General: {mejor_metodo_general}
  • Para Luis: {mejor_metodo_luis} ({mejora_luis:+.1f}% vs baseline)
  • Para Byron: {mejor_metodo_byron} ({mejora_byron:+.1f}% vs baseline)

📊 VELOCIDADES PROMEDIO GENERALES:
  • Baseline:    {metodos_stats.loc['Baseline', 'Promedio_Ambos']:.1f} ppm
  • Método 2:  {metodos_stats.loc['Método 2', 'Promedio_Ambos']:.1f} ppm
  • Método 3:      {metodos_stats.loc['Método 3', 'Promedio_Ambos']:.1f} ppm

🔍 OBSERVACIONES CLAVE:
  1. El método Método 3 {'aumentó' if metodos_stats.loc['Método 3', 'Promedio_Ambos'] > metodos_stats.loc['Baseline', 'Promedio_Ambos'] else 'no mejoró'} significativamente la velocidad de lectura
  2. La música {'tuvo un impacto negativo' if metodos_stats.loc['Método 2', 'Promedio_Ambos'] < metodos_stats.loc['Baseline', 'Promedio_Ambos'] else 'mejoró'} en el rendimiento
  3. La consistencia {'mejoró' if cv_data.loc['Método 3'].mean() < cv_data.loc['Baseline'].mean() else 'varió'} con condiciones óptimas

💡 RECOMENDACIONES:
  • Utilizar el método "{mejor_metodo_general}" para máxima velocidad
  • {'Evitar' if metodos_stats.loc['Método 2', 'Promedio_Ambos'] < metodos_stats.loc['Baseline', 'Promedio_Ambos'] else 'Considerar'} música durante la lectura
  • El ambiente controlado y lectura en voz alta, emplear técnicas para evitar redundancia y subvocalización {'son beneficiosos' if metodos_stats.loc['Método 3', 'Promedio_Ambos'] > metodos_stats.loc['Baseline', 'Promedio_Ambos'] else 'no mostraron mejoras significativas'}
""")

print("="*80)