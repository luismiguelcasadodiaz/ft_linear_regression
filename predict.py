"""
    Reads thetas from a CSV file, Uses them to make predictions based on user 
    input.
    
    The user is prompted to enter a value for mileage, and the program 
    calculates the corresponding price using the linear regression model defined
    by the thetas. The predicted price is then displayed to the user.

    The thetas are read from a CSV file named 'thetas.csv', which is expected to
    contain two values: the intercept (theta_0) and the slope (theta_1). 
    
    The program will continue to prompt the user for mileage values until the 
    user decides to exit by entering q or Q. If the user enters an invalid input
    (non-numeric), an error message will be displayed, and the user will be 
    prompted again. 

    Example of thetas.csv content:
    0.0,0.0
    This indicates that the linear regression model predicts a price of 0.0 for
    a mileage of 0.0, and the price increases by 0.0 for each additional unit of
    mileage. 

    Note: The program assumes that the thetas.csv file is formatted correctly and
    contains valid numeric values. If the file is missing or contains invalid data,
    the program will raise an error.    

    Usage:
    1. Ensure that the 'thetas.csv' file is present in the same directory as this script.
    2. Run the script. It will read the thetas from the CSV file and prompt you to enter mileage values.
    3. Enter a mileage value to get the predicted price based on the linear regression model.
    4. To exit the program, enter 'q' or 'Q' when prompted for mileage.
    Example:
    $ python predict.py
    Enter mileage (or 'q' to quit): 15000
    Predicted price: 30000.0
    Enter mileage (or 'q' to quit): 20000
    Predicted price: 40000.0
    Enter mileage (or 'q' to quit): q
    Exiting the program.

"""
import csv
import sys

data_file = "thetas.csv"
try:
    with open(data_file, mode='r') as file:
        reader = csv.reader(file)
        thetas = next(reader)  # Read the first line
        theta_0 = float(thetas[0])  # Intercept
        theta_1 = float(thetas[1])  # Slope
except FileNotFoundError:
    print(f"Error: The file '{data_file}' was not found.")
    sys.exit(1)
except ValueError:
    print(f"Error: The file '{data_file}' contains invalid data.")
    sys.exit(1)

mileage = input("Enter mileage (or 'q' to quit): ")
while mileage.lower() != 'q':
    try:
        mileage_value = float(mileage)
        predicted_price = theta_0 + theta_1 * mileage_value
        print(f"Predicted price: {predicted_price}")
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")
    
    mileage = input("Enter mileage (or 'q' to quit): ")     

