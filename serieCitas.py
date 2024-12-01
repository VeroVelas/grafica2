import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Crear una carpeta para guardar las gráficas
GRAPH_PATH = "./graficas"
os.makedirs(GRAPH_PATH, exist_ok=True)

# Cargar los datos
data = pd.read_csv('dataset_citas_mascotas.csv')
data['Fecha'] = pd.to_datetime(data['Fecha'])
data.set_index('Fecha', inplace=True)

# Consolidar datos diarios
citas_diarias = data.resample('D').sum()['Cantidad de Citas']

# Validaciones de los datos
if citas_diarias.isnull().any():
    print("Datos faltantes detectados. Imputando valores...")
    citas_diarias = citas_diarias.fillna(0)

if not pd.api.types.is_numeric_dtype(citas_diarias):
    raise ValueError("La columna 'Cantidad de Citas' debe ser numérica.")

# Filtrar los datos para el mes anterior y el actual
start_date = citas_diarias.index[-1] - pd.DateOffset(months=1)
end_date = citas_diarias.index[-1]
filtered_data = citas_diarias[start_date:end_date]

# ---- Modelo SARIMA ---- #
sarima_model = SARIMAX(filtered_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))
sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.get_forecast(steps=30).predicted_mean

# Calcular el porcentaje de aumento
last_month_avg = filtered_data.mean()  # Promedio del mes actual y anterior
next_month_avg_sarima = sarima_forecast.mean()  # Promedio del siguiente mes (SARIMA)
percentage_increase_sarima = ((next_month_avg_sarima - last_month_avg) / last_month_avg) * 100

# ---- Gráfica de Línea para Vista de Usuario ---- #
line_chart_path = os.path.join(GRAPH_PATH, "grafica_linea_citas_usuario.png")
plt.figure(figsize=(14, 8))

# Línea de datos históricos
plt.plot(filtered_data.index, filtered_data, label="Citas Reales", color="blue", linewidth=2, marker='o')

# Línea proyectada (SARIMA)
plt.plot(sarima_forecast.index, sarima_forecast, label="Proyección Próximo Mes", color="red", linestyle="--", linewidth=2)

# Títulos y detalles
plt.title("Citas Diarias y Proyección", fontsize=18, fontweight='bold')
plt.xlabel("Fecha", fontsize=14)
plt.ylabel("Cantidad de Citas", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar aumento o disminución en la gráfica
if percentage_increase_sarima > 0:
    trend_message = f"Aumento proyectado: {percentage_increase_sarima:.2f}%"
    trend_color = "green"
else:
    trend_message = f"Disminución proyectada: {abs(percentage_increase_sarima):.2f}%"
    trend_color = "red"

plt.text(0.02, 0.92, trend_message, transform=plt.gca().transAxes, fontsize=12, color=trend_color, fontweight='bold')

# Guardar y cerrar la gráfica
plt.tight_layout()
plt.savefig(line_chart_path)
plt.close()

# ---- Gráfica de Pastel: Distribución de Animales ---- #
pie_chart_path = os.path.join(GRAPH_PATH, "grafica_pastel_tipo_animal.png")
tipo_animal_totales = data['Tipo de Animal'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(tipo_animal_totales, labels=tipo_animal_totales.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribución por Tipo de Animal", fontsize=16)
plt.legend(tipo_animal_totales.index, title="Tipos de Animales", loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(pie_chart_path)
plt.close()

# Mensajes de confirmación
print(f"Gráfica de línea guardada en: {line_chart_path}")
print(f"Gráfica de pastel guardada en: {pie_chart_path}")
