import math
"""
This method is called gradient descent.

The name comes from the core idea: it finds the line that minimizes the sum of the squared differences (the "squares") between the actual data points and the predicted values on the line — hence "least" squares, since we're minimizing that sum.

A few related terms you might come across:

Simple linear regression — refers to this specific case with one predictor variable (
(x) and one outcome (y), fitted with least squares.
Method of least squares — the general mathematical technique (invented independently by Legendre and Gauss around 1805), which also applies to more complex models (multiple regression, polynomial fitting, etc.).
Ordinary least squares (OLS) — the standard version, as opposed to variants like weighted least squares or generalized least squares that handle special cases (e.g., unequal variance in the errors).
"""
def calculate_gd(puntos_x, puntos_y)-> tuple[float, float]:
    """
    Calculates the slope and y-intercept of the linear regression line
    using the gradient descent method.

    Parameters:
        puntos_x -- list of x coordinates of the points
        puntos_y -- list of y coordinates of the points

    Returns:
        A tuple (slope, intercept_y) where:
            slope -- slope of the line (m)
            intercept_y   -- point of intersection with the y-axis (b)
    """
    x_min, x_max = min(puntos_x), max(puntos_x)
    x_range = x_max - x_min
    if x_range == 0:
        raise ValueError("All x values are the same. Cannot perform gradient descent.")

    y_min, y_max = min(puntos_y), max(puntos_y)
    y_range = max(puntos_y) - min(puntos_y)
    if y_range == 0:
        raise ValueError("All y values are the same. Cannot perform gradient descent.")

    x_normalized = [(x - x_min) / (x_max - x_min) for x in puntos_x]
    y_normalized = [(y - y_min) / (y_max - y_min) for y in puntos_y]

    n = len(puntos_x)
    if n <= 1:
        raise ValueError("No hay puntos para calcular la regresión.")
    slope_n, intercept_y_n = 0.0, 0.0
    slope_n_1, intercept_y_n_1 = 0.0, 0.0
    learning_rate = 0.1
    for _ in range(10000):  # Number of iterations
        y_pred = [slope_n * x + intercept_y_n for x in x_normalized]
        error = [y - y_hat for y, y_hat in zip(y_normalized, y_pred)]
        slope_gradient = (-1/n) * sum(x * e for x, e in zip(x_normalized, error))
        intercept_gradient = (-1/n) * sum(error)
        slope_n_1 -= learning_rate * slope_gradient
        intercept_y_n_1 -= learning_rate * intercept_gradient
        print(f"Iteration {_}: slope = {slope_n_1}, intercept_y = {intercept_y_n}, error = {abs(slope_n_1 - slope_n)}")
        if abs(slope_n_1 - slope_n) < 1e-6:
            break
        if slope_n_1 == float('inf') or intercept_y_n_1 == float('inf'):
            print("Error: The gradient descent algorithm diverged.")
            slope_n_1, intercept_y_n_1 = slope_n, intercept_y_n  # Reset to previous values
            break
        slope_n = slope_n_1
        intercept_y_n = intercept_y_n_1
    slop_denormalized = slope_n_1 * (y_range / x_range)
    intercept_y_denormalized = intercept_y_n_1 * y_range + y_min - slop_denormalized * x_min
    return intercept_y_denormalized, slop_denormalized

    