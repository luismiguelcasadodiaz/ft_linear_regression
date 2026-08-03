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
    n = len(puntos_x)
    if n <= 1:
        raise ValueError("No hay puntos para calcular la regresión.")
    slope_n, intercept_y_n = 0.0, 0.0
    slope_n_1, intercept_y_n_1 = 0.0, 0.0
    learning_rate = 0.01
    for _ in range(1000):  # Number of iterations
        y_pred = [slope_n * x + intercept_y_n for x in puntos_x]
        print(f"Iteration {_}: slope = {slope_n}, intercept_y = {intercept_y_n}, y_pred = {y_pred}")
        error = [y - y_hat for y, y_hat in zip(puntos_y, y_pred)]
        slope_gradient = (-2/n) * sum(x * e for x, e in zip(puntos_x, error))
        intercept_gradient = (-2/n) * sum(error)
        slope_n_1 -= learning_rate * slope_gradient
        intercept_y_n_1 -= learning_rate * intercept_gradient
        if math.isclose(slope_n_1, slope_n, abs_tol=1e-9):
            break
        slope_n = slope_n_1
        intercept_y_n = intercept_y_n_1
    return slope_n_1, intercept_y_n_1