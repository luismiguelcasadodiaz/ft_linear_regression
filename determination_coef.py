

def r_square(y_true, y_pred)->float:
    """
    Calculate the coefficient of determination (R^2) for a linear regression model.

    Parameters:
    y_true (list): The true values of the dependent variable.
    y_pred (list): The predicted values from the regression model.

    Returns:
    float: The R^2 value, which indicates how well the model explains the variability of the response data.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")

    ss_res = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((y_t - mean_y) ** 2 for y_t in y_true)

    if ss_tot == 0:
        raise ValueError("Total sum of squares (ss_tot) is zero; cannot compute R^2.")

    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def calculate_r_squared(puntos_x, puntos_y, slope, intercept_y)->float:
    """
    Calculate the R^2 value for a linear regression model given the slope and intercept.

    Parameters:
    puntos_x (list): The x coordinates of the data points.
    puntos_y (list): The y coordinates of the data points.
    slope (float): The slope of the regression line.
    intercept_y (float): The y-intercept of the regression line.

    Returns:
    float: The R^2 value, which indicates how well the model explains the variability of the response data.
    """
    y_pred = [slope * x + intercept_y for x in puntos_x]
    return r_square(puntos_y, y_pred)