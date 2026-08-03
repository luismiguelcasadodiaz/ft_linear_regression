"""
Regression line trainer  using Gradient Descent (GD)
Reads points from a CSV file 'data.csv', calculates the regression line using GD,
and saves the resulting thetas to a CSV file 'thetas.csv.

The CSV file should contain two columns: the first column for the x values 
(mileage) and the second column for the y values (price). 
The first row should contain the headers.

if the CSV file is not formatted correctly, the program will raise an error.
if the CSV file is missing or contains invalid data, the program will raise an error.

The program will read the points from the CSV file, calculate the regression line
using calculate_gd function, and save the resulting thetas (intercept and slope)
to 'thetas.csv'.


"""
import csv
from gd import calculate_gd
from ols import calculate_ols

points_file = "data.csv"
try:
    with open(points_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        puntos_x = []
        puntos_y = []
        for row in reader:
            puntos_x.append(float(row[0]))
            puntos_y.append(float(row[1]))
except FileNotFoundError:
    print(f"Error: The file '{points_file}' was not found.")
    exit(1)
except ValueError:
    print(f"Error: The file '{points_file}' contains invalid data.")
    exit(1) 

theta_1, theta_0 = calculate_ols(puntos_x, puntos_y)
print(f"Ordinary Least Squares Calculated thetas: theta_0 (intercept) = {theta_0}, theta_1 (slope) = {theta_1}")   

theta_1, theta_0 = calculate_gd(puntos_x, puntos_y)
print(f"Gradient Descent Calculated thetas: theta_0 (intercept) = {theta_0}, theta_1 (slope) = {theta_1}")   
thetas_file = "thetas.csv"
try:
    with open(thetas_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([theta_0, theta_1])
except Exception as e:
    print(f"Error: Could not write to the file '{thetas_file}'. {e}")
    exit(1)