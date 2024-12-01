import pandas as pd
import numpy as np
import random

# Generar datos sintéticos para un dataset
np.random.seed(42)
random.seed(42)

# Configuración de fechas
fechas = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

# Generar columnas
data = {
    "Fecha": fechas,
    "Tipo de Animal": random.choices(["Perro", "Gato"], k=len(fechas)),
    "Cantidad de Citas": np.random.poisson(lam=5, size=len(fechas)),
}

# Crear DataFrame
df = pd.DataFrame(data)

# Guardar el dataset en un archivo CSV
output_file = "dataset_citas_mascotas.csv"
df.to_csv(output_file, index=False)

print(f"Dataset generado y guardado en {output_file}")
