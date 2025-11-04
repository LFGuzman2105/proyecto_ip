import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("="*80)
print("SIMULACIÓN DE PROCESO SUBWAY CON SimPy")
print("="*80)

# ============================================================================
# 1. CARGAR Y PROCESAR DATOS REALES
# ============================================================================
df = pd.read_csv('datos_subway.csv')

def time_to_seconds(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

# Convertir tiempos a segundos
time_columns = ['Tiempo ordenar', 'Tiempo preparar pan', 'Tiempo horno', 
                'Tiempo vegetales', 'Tiempo caja']

for col in time_columns:
    df[col + ' (seg)'] = df[col].apply(time_to_seconds)

# Calcular estadísticas para la simulación
stats_data = {}
for col in time_columns:
    col_seg = col + ' (seg)'
    stats_data[col] = {
        'mean': df[col_seg].mean(),
        'std': df[col_seg].std(),
        'min': df[col_seg].min(),
        'max': df[col_seg].max()
    }

# Distribución de cantidad de panes
prob_1_pan = (df['Cantidad de panes'] == 1).sum() / len(df)
prob_2_panes = (df['Cantidad de panes'] == 2).sum() / len(df)

print("\n1. DATOS OBSERVADOS CARGADOS:")
print(f"   - Total observaciones: {len(df)}")
print(f"   - Probabilidad 1 pan: {prob_1_pan:.2%}")
print(f"   - Probabilidad 2 panes: {prob_2_panes:.2%}")

# ============================================================================
# 2. CONFIGURACIÓN DE LA SIMULACIÓN
# ============================================================================

class SubwayConfig:
    """Configuración del sistema Subway"""
    # Recursos (según observación)
    NUM_EMPLEADOS_ORDEN = 1      # Empleado 1: Tomar orden, preparar pan, meter al horno
    NUM_EMPLEADOS_HORNO_VEG = 1  # Empleado 2: Preparar pan, meter/sacar horno, vegetales
    NUM_EMPLEADOS_VEG = 1        # Empleado 3: Sacar del horno, vegetales
    NUM_EMPLEADOS_CAJA = 1       # Empleado 4: Caja
    NUM_HORNOS = 2               # 2 hornos disponibles
    NUM_CAJAS = 1                # 1 caja registradora
    
    # Tiempos de llegada
    TIEMPO_ENTRE_LLEGADAS_MEAN = 120  # segundos (ajustable)
    
    # Tiempo de simulación
    TIEMPO_SIMULACION = 8 * 3600  # 8 horas en segundos

config = SubwayConfig()

print("\n2. CONFIGURACIÓN DE RECURSOS:")
print(f"   - Empleados estación orden/preparar: {config.NUM_EMPLEADOS_ORDEN}")
print(f"   - Empleados estación horno/vegetales: {config.NUM_EMPLEADOS_HORNO_VEG}")
print(f"   - Empleados estación vegetales: {config.NUM_EMPLEADOS_VEG}")
print(f"   - Empleados caja: {config.NUM_EMPLEADOS_CAJA}")
print(f"   - Hornos disponibles: {config.NUM_HORNOS}")
print(f"   - Cajas registradoras: {config.NUM_CAJAS}")

# ============================================================================
# 3. MODELO DE SIMULACIÓN
# ============================================================================

class SubwaySimulation:
    def __init__(self, env, config, stats_data, prob_1_pan, mejoras=None):
        self.env = env
        self.config = config
        self.stats_data = stats_data
        self.prob_1_pan = prob_1_pan
        self.mejoras = mejoras or {}
        
        # Recursos del sistema
        self.empleado_orden = simpy.Resource(env, capacity=config.NUM_EMPLEADOS_ORDEN)
        self.empleado_horno_veg = simpy.Resource(env, capacity=config.NUM_EMPLEADOS_HORNO_VEG)
        self.empleado_veg = simpy.Resource(env, capacity=config.NUM_EMPLEADOS_VEG)
        self.empleado_caja = simpy.Resource(env, capacity=config.NUM_EMPLEADOS_CAJA)
        self.hornos = simpy.Resource(env, capacity=config.NUM_HORNOS)
        
        # Métricas
        self.ordenes_completadas = []
        self.tiempos_espera = []
        self.tiempos_proceso = []
        self.utilizacion_recursos = {
            'orden': [],
            'horno_veg': [],
            'veg': [],
            'caja': [],
            'hornos': []
        }
    
    def generar_tiempo(self, estacion):
        """Genera tiempo basado en distribución normal de datos observados"""
        mean = self.stats_data[estacion]['mean']
        std = self.stats_data[estacion]['std']
        min_val = self.stats_data[estacion]['min']
        max_val = self.stats_data[estacion]['max']
        
        # Aplicar mejoras si existen
        factor_mejora = self.mejoras.get(estacion, 1.0)
        mean = mean * factor_mejora
        
        # Generar tiempo con distribución normal truncada
        tiempo = np.random.normal(mean, std)
        tiempo = max(min_val, min(tiempo, max_val))
        return max(1, tiempo)  # Mínimo 1 segundo
    
    def proceso_cliente(self, nombre, num_panes):
        """Simula el proceso completo de un cliente"""
        llegada = self.env.now
        tiempos = {'llegada': llegada}
        
        # 1. ORDENAR (Empleado 1)
        with self.empleado_orden.request() as req:
            yield req
            inicio_orden = self.env.now
            tiempos['espera_orden'] = inicio_orden - llegada
            tiempo_ordenar = self.generar_tiempo('Tiempo ordenar')
            yield self.env.timeout(tiempo_ordenar)
            tiempos['tiempo_ordenar'] = tiempo_ordenar
        
        # 2. PREPARAR PAN (puede ser Empleado 1 o 2)
        # Simulamos que cualquiera puede hacerlo
        empleado_preparar = self.empleado_orden if np.random.random() < 0.5 else self.empleado_horno_veg
        with empleado_preparar.request() as req:
            yield req
            inicio_preparar = self.env.now
            tiempos['espera_preparar'] = inicio_preparar - (llegada + tiempos['tiempo_ordenar'])
            tiempo_preparar = self.generar_tiempo('Tiempo preparar pan')
            yield self.env.timeout(tiempo_preparar)
            tiempos['tiempo_preparar'] = tiempo_preparar
        
        # 3. HORNO (Requiere empleado + horno físico)
        empleado_horno = self.empleado_horno_veg
        with empleado_horno.request() as req_emp, self.hornos.request() as req_horno:
            yield req_emp & req_horno
            inicio_horno = self.env.now
            tiempo_horno = self.generar_tiempo('Tiempo horno')
            yield self.env.timeout(tiempo_horno)
            tiempos['tiempo_horno'] = tiempo_horno
        
        # 4. VEGETALES (Empleado 2 o 3)
        empleado_veg = self.empleado_veg if np.random.random() < 0.5 else self.empleado_horno_veg
        with empleado_veg.request() as req:
            yield req
            inicio_veg = self.env.now
            tiempo_veg = self.generar_tiempo('Tiempo vegetales')
            # Si no lleva vegetales (tiempo 0 en datos), skip
            if tiempo_veg > 5:  # Solo si es significativo
                yield self.env.timeout(tiempo_veg)
            tiempos['tiempo_vegetales'] = tiempo_veg if tiempo_veg > 5 else 0
        
        # 5. CAJA (Empleado 4)
        with self.empleado_caja.request() as req:
            yield req
            inicio_caja = self.env.now
            tiempo_caja = self.generar_tiempo('Tiempo caja')
            yield self.env.timeout(tiempo_caja)
            tiempos['tiempo_caja'] = tiempo_caja
        
        # Calcular métricas
        salida = self.env.now
        tiempo_total = salida - llegada
        tiempo_proceso_puro = (tiempos['tiempo_ordenar'] + tiempos['tiempo_preparar'] + 
                               tiempos['tiempo_horno'] + tiempos['tiempo_vegetales'] + 
                               tiempos['tiempo_caja'])
        tiempo_espera_total = tiempo_total - tiempo_proceso_puro
        
        self.ordenes_completadas.append({
            'nombre': nombre,
            'num_panes': num_panes,
            'llegada': llegada,
            'salida': salida,
            'tiempo_total': tiempo_total,
            'tiempo_proceso': tiempo_proceso_puro,
            'tiempo_espera': tiempo_espera_total,
            **tiempos
        })
        
        self.tiempos_proceso.append(tiempo_total)
        self.tiempos_espera.append(tiempo_espera_total)
    
    def generador_clientes(self):
        """Genera llegada de clientes"""
        num_cliente = 1
        while True:
            # Determinar cantidad de panes
            num_panes = 1 if np.random.random() < self.prob_1_pan else 2
            
            # Crear proceso de cliente
            self.env.process(self.proceso_cliente(f"Cliente_{num_cliente}", num_panes))
            
            # Tiempo hasta siguiente cliente (distribución exponencial)
            tiempo_entre_llegadas = np.random.exponential(self.config.TIEMPO_ENTRE_LLEGADAS_MEAN)
            yield self.env.timeout(tiempo_entre_llegadas)
            
            num_cliente += 1
    
    def monitor_utilizacion(self):
        """Monitorea utilización de recursos"""
        while True:
            self.utilizacion_recursos['orden'].append(
                self.empleado_orden.count / self.empleado_orden.capacity
            )
            self.utilizacion_recursos['horno_veg'].append(
                self.empleado_horno_veg.count / self.empleado_horno_veg.capacity
            )
            self.utilizacion_recursos['veg'].append(
                self.empleado_veg.count / self.empleado_veg.capacity
            )
            self.utilizacion_recursos['caja'].append(
                self.empleado_caja.count / self.empleado_caja.capacity
            )
            self.utilizacion_recursos['hornos'].append(
                self.hornos.count / self.hornos.capacity
            )
            yield self.env.timeout(60)  # Cada minuto

# ============================================================================
# 4. EJECUTAR SIMULACIÓN ORIGINAL
# ============================================================================

print("\n" + "="*80)
print("3. EJECUTANDO SIMULACIÓN DEL PROCESO ORIGINAL...")
print("="*80)

# Simulación original
env_original = simpy.Environment()
sim_original = SubwaySimulation(env_original, config, stats_data, prob_1_pan)

env_original.process(sim_original.generador_clientes())
env_original.process(sim_original.monitor_utilizacion())
env_original.run(until=config.TIEMPO_SIMULACION)

# Resultados simulación original
df_sim_original = pd.DataFrame(sim_original.ordenes_completadas)

print(f"\n   ✓ Simulación completada")
print(f"   - Órdenes procesadas: {len(df_sim_original)}")
print(f"   - Tiempo promedio total: {df_sim_original['tiempo_total'].mean():.1f} seg")
print(f"   - Tiempo promedio de espera: {df_sim_original['tiempo_espera'].mean():.1f} seg")
print(f"   - Tiempo promedio de proceso: {df_sim_original['tiempo_proceso'].mean():.1f} seg")

# ============================================================================
# 5. SIMULACIÓN CON MEJORAS
# ============================================================================

print("\n" + "="*80)
print("4. EJECUTANDO SIMULACIÓN CON MEJORAS PROPUESTAS...")
print("="*80)

# Definir mejoras (reducción de tiempos)
mejoras = {
    'Tiempo ordenar': 0.85,      # 15% más rápido con mejor sistema
    'Tiempo preparar pan': 0.90,  # 10% más rápido con práctica
    'Tiempo vegetales': 0.80,     # 20% más rápido organizando mejor
    'Tiempo caja': 0.85,          # 15% más rápido con sistema mejorado
}

print("\n   MEJORAS APLICADAS:")
for estacion, factor in mejoras.items():
    reduccion = (1 - factor) * 100
    print(f"   - {estacion}: {reduccion:.0f}% reducción")

# Simulación con mejoras
env_mejorado = simpy.Environment()
sim_mejorado = SubwaySimulation(env_mejorado, config, stats_data, prob_1_pan, mejoras)

env_mejorado.process(sim_mejorado.generador_clientes())
env_mejorado.process(sim_mejorado.monitor_utilizacion())
env_mejorado.run(until=config.TIEMPO_SIMULACION)

# Resultados simulación mejorada
df_sim_mejorado = pd.DataFrame(sim_mejorado.ordenes_completadas)

print(f"\n   ✓ Simulación con mejoras completada")
print(f"   - Órdenes procesadas: {len(df_sim_mejorado)}")
print(f"   - Tiempo promedio total: {df_sim_mejorado['tiempo_total'].mean():.1f} seg")
print(f"   - Tiempo promedio de espera: {df_sim_mejorado['tiempo_espera'].mean():.1f} seg")
print(f"   - Tiempo promedio de proceso: {df_sim_mejorado['tiempo_proceso'].mean():.1f} seg")

# ============================================================================
# 6. VALIDACIÓN: COMPARAR SIMULACIÓN VS DATOS REALES
# ============================================================================

print("\n" + "="*80)
print("5. VALIDACIÓN: COMPARACIÓN SIMULACIÓN vs DATOS REALES")
print("="*80)

# Calcular tiempo total real
df['Tiempo Total (seg)'] = df[[col + ' (seg)' for col in time_columns]].sum(axis=1)

# Estadísticas comparativas
print("\nTIEMPO TOTAL DEL PROCESO:")
print(f"   Datos Reales:        {df['Tiempo Total (seg)'].mean():.1f} ± {df['Tiempo Total (seg)'].std():.1f} seg")
print(f"   Simulación Original: {df_sim_original['tiempo_total'].mean():.1f} ± {df_sim_original['tiempo_total'].std():.1f} seg")
print(f"   Simulación Mejorada: {df_sim_mejorado['tiempo_total'].mean():.1f} ± {df_sim_mejorado['tiempo_total'].std():.1f} seg")

# Test estadístico (t-test)
t_stat, p_value = stats.ttest_ind(df['Tiempo Total (seg)'], df_sim_original['tiempo_total'])
print(f"\n   Test t (Real vs Simulación): t={t_stat:.3f}, p={p_value:.3f}")
if p_value > 0.05:
    print(f"   ✓ La simulación replica bien el proceso real (p > 0.05)")
else:
    print(f"   ⚠ Hay diferencias significativas (p < 0.05)")

# Comparar mejoras
mejora_tiempo = ((df_sim_original['tiempo_total'].mean() - 
                  df_sim_mejorado['tiempo_total'].mean()) / 
                 df_sim_original['tiempo_total'].mean() * 100)
print(f"\n   IMPACTO DE MEJORAS: {mejora_tiempo:.1f}% reducción en tiempo total")

# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Comparación de distribuciones de tiempo total
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['Tiempo Total (seg)'], bins=15, alpha=0.5, label='Datos Reales', 
         color='blue', density=True)
ax1.hist(df_sim_original['tiempo_total'], bins=15, alpha=0.5, 
         label='Simulación Original', color='green', density=True)
ax1.hist(df_sim_mejorado['tiempo_total'], bins=15, alpha=0.5, 
         label='Simulación Mejorada', color='orange', density=True)
ax1.set_xlabel('Tiempo Total (segundos)')
ax1.set_ylabel('Densidad')
ax1.set_title('Validación: Distribución de Tiempos Totales')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Box plot comparativo
ax2 = fig.add_subplot(gs[0, 2])
data_box = [df['Tiempo Total (seg)'], df_sim_original['tiempo_total'], 
            df_sim_mejorado['tiempo_total']]
bp = ax2.boxplot(data_box, labels=['Real', 'Sim\nOriginal', 'Sim\nMejorada'],
                 patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_ylabel('Tiempo Total (seg)')
ax2.set_title('Comparación de Variabilidad')
ax2.grid(True, alpha=0.3)

# 3. Tiempo de proceso vs espera (Simulación Original)
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(df_sim_original['tiempo_proceso'], df_sim_original['tiempo_espera'],
           alpha=0.6, s=50)
ax3.set_xlabel('Tiempo de Proceso (seg)')
ax3.set_ylabel('Tiempo de Espera (seg)')
ax3.set_title('Original: Proceso vs Espera')
ax3.grid(True, alpha=0.3)

# 4. Tiempo de proceso vs espera (Simulación Mejorada)
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(df_sim_mejorado['tiempo_proceso'], df_sim_mejorado['tiempo_espera'],
           alpha=0.6, s=50, color='orange')
ax4.set_xlabel('Tiempo de Proceso (seg)')
ax4.set_ylabel('Tiempo de Espera (seg)')
ax4.set_title('Mejorada: Proceso vs Espera')
ax4.grid(True, alpha=0.3)

# 5. Utilización de recursos (Original)
ax5 = fig.add_subplot(gs[1, 2])
util_original = {k: np.mean(v) * 100 for k, v in sim_original.utilizacion_recursos.items()}
ax5.bar(util_original.keys(), util_original.values(), color='steelblue', alpha=0.7)
ax5.set_ylabel('Utilización (%)')
ax5.set_title('Utilización de Recursos (Original)')
ax5.tick_params(axis='x', rotation=45)
ax5.axhline(80, color='red', linestyle='--', label='Límite ideal 80%')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Tiempos por estación (Comparación)
ax6 = fig.add_subplot(gs[2, :])
estaciones = ['tiempo_ordenar', 'tiempo_preparar', 'tiempo_horno', 
              'tiempo_vegetales', 'tiempo_caja']
nombres_est = ['Ordenar', 'Preparar Pan', 'Horno', 'Vegetales', 'Caja']

x = np.arange(len(nombres_est))
width = 0.25

tiempos_reales = [df[col + ' (seg)'].mean() for col in time_columns]
tiempos_sim_orig = [df_sim_original[col].mean() for col in estaciones]
tiempos_sim_mej = [df_sim_mejorado[col].mean() for col in estaciones]

ax6.bar(x - width, tiempos_reales, width, label='Datos Reales', color='blue', alpha=0.7)
ax6.bar(x, tiempos_sim_orig, width, label='Sim. Original', color='green', alpha=0.7)
ax6.bar(x + width, tiempos_sim_mej, width, label='Sim. Mejorada', color='orange', alpha=0.7)

ax6.set_xlabel('Estación')
ax6.set_ylabel('Tiempo Promedio (seg)')
ax6.set_title('Comparación de Tiempos por Estación')
ax6.set_xticks(x)
ax6.set_xticklabels(nombres_est)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('SIMULACIÓN Y VALIDACIÓN DEL PROCESO SUBWAY', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('discrete_event_simulation_model_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. RESUMEN EJECUTIVO
# ============================================================================

print("\n" + "="*80)
print("6. RESUMEN EJECUTIVO")
print("="*80)

print("\n📊 VALIDACIÓN DEL MODELO:")
error_porcentual = abs(df['Tiempo Total (seg)'].mean() - 
                       df_sim_original['tiempo_total'].mean()) / df['Tiempo Total (seg)'].mean() * 100
print(f"   - Error del modelo: {error_porcentual:.1f}%")
if error_porcentual < 10:
    print(f"   ✓ Modelo validado (error < 10%)")
else:
    print(f"   ⚠ Modelo requiere calibración (error > 10%)")

print("\n💡 IMPACTO DE MEJORAS PROPUESTAS:")
print(f"   - Reducción tiempo total: {mejora_tiempo:.1f}%")
print(f"   - Ahorro por orden: {df_sim_original['tiempo_total'].mean() - df_sim_mejorado['tiempo_total'].mean():.1f} seg")
print(f"   - Órdenes adicionales/día: {len(df_sim_mejorado) - len(df_sim_original)} órdenes")

capacidad_adicional = ((len(df_sim_mejorado) - len(df_sim_original)) / 
                       len(df_sim_original) * 100)
print(f"   - Incremento capacidad: {capacidad_adicional:.1f}%")

print("\n🎯 RECOMENDACIONES:")
print("   1. Implementar mejoras en estaciones de alta variabilidad")
print("   2. Capacitar personal para reducir tiempos de ordenar y caja")
print("   3. Optimizar organización de vegetales")
print("   4. Monitorear utilización de recursos continuamente")

print("\n" + "="*80)