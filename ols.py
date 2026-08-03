"""
This method is called least squares (or more fully, ordinary least squares, often abbreviated OLS).

The name comes from the core idea: it finds the line that minimizes the sum of the squared differences (the "squares") between the actual data points and the predicted values on the line — hence "least" squares, since we're minimizing that sum.

A few related terms you might come across:

Simple linear regression — refers to this specific case with one predictor variable (
x
x) and one outcome (
y
y), fitted with least squares.
Method of least squares — the general mathematical technique (invented independently by Legendre and Gauss around 1805), which also applies to more complex models (multiple regression, polynomial fitting, etc.).
Ordinary least squares (OLS) — the standard version, as opposed to variants like weighted least squares or generalized least squares that handle special cases (e.g., unequal variance in the errors).
"""
def calculate_ols(puntos_x, puntos_y)-> tuple[float, float]:
    """
    Calcula la pendiente y el corte con el eje y de la recta de regresión lineal
    usando el método de mínimos cuadrados (OLS).

    Parámetros:
        puntos_x -- lista de coordenadas x de los puntos
        puntos_y -- lista de coordenadas y de los puntos

    Retorna:
        Una tupla (pendiente, corte_y) donde:
            pendiente -- pendiente de la recta (m)
            corte_y   -- punto de corte con el eje y (b)
    """
    n = len(puntos_x)
    if n <= 1:
        raise ValueError("No hay puntos para calcular la regresión.")

    sum_x = sum(puntos_x)
    sum_y = sum(puntos_y)
    sum_xy = sum(x * y for x, y in zip(puntos_x, puntos_y))
    sum_x_squared = sum(x ** 2 for x in puntos_x)

    # Fórmulas para la pendiente (m) y el corte con el eje y (b)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    intercept_y = (sum_y - slope * sum_x) / n

    return slope, intercept_y