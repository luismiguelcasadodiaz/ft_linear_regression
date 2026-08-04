"""
Comparación de regresión lineal: OLS vs Gradient Descent
----------------------------------------------------------
Lee data.csv (columnas: km, price) y dibuja dos scatter plots lado a lado,
cada uno con la recta de regresión calculada por un método distinto,
mostrando el tiempo de ejecución de cada método en el título.
"""

import csv
import time

import matplotlib.pyplot as plt

from determination_coef import calculate_r_squared
from ols import calculate_ols
from gd import calculate_gd


# ---------------------------------------------------------------------------
# Lectura de datos
# ---------------------------------------------------------------------------

def leer_datos(ruta="data.csv"):
    puntos_x = []  # kilómetros
    puntos_y = []  # precio

    with open(ruta, newline="") as f:
        reader = csv.reader(f)
        cabecera = next(reader)  # saltamos la línea de cabecera
        for fila in reader:
            if not fila:
                continue
            km, precio = fila[0], fila[1]
            puntos_x.append(float(km))
            puntos_y.append(float(precio))

    return puntos_x, puntos_y


# ---------------------------------------------------------------------------
# Dibujado
# ---------------------------------------------------------------------------

def dibujar_comparacion(ax, puntos_x, puntos_y, funcion_calculo, nombre_metodo):
    inicio = time.perf_counter()
    interseccion, pendiente = funcion_calculo(puntos_x, puntos_y)
    duracion_ms = (time.perf_counter() - inicio) * 1000

    r_2 = calculate_r_squared(puntos_x, puntos_y, pendiente, interseccion)

    ax.scatter(puntos_x, puntos_y, color="tab:blue", s=25, alpha=0.7)
    ax.axline(
        (0, interseccion), slope=pendiente, color="tab:red", linewidth=1.5
    )
    ax.text(
        0.55, 0.97,
        f"y = {pendiente:.4f}x + {interseccion:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.text(
        0.05, 0.07,
        f"R2 = {r_2:.6f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )    
    ax.set_xlim(left=0)

    ax.set_title(f"{nombre_metodo}\n{duracion_ms:.3f} ms")
    ax.set_xlabel("Kilómetros")
    ax.set_ylabel("Precio")
    ax.grid(True, linestyle="--", alpha=0.4)

    return interseccion, pendiente, duracion_ms


def main():
    puntos_x, puntos_y = leer_datos("data.csv")

    fig, (ax_ols, ax_gd) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    dibujar_comparacion(ax_ols, puntos_x, puntos_y, calculate_ols, "OLS")
    dibujar_comparacion(ax_gd, puntos_x, puntos_y, calculate_gd, "Gradient Descent")

    fig.suptitle("Comparación de regresión lineal: OLS vs Gradient Descent")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
