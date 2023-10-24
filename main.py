import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

### Variables

# Size of polynomial used in the regression model
degrees = 6

###


# Updated dataset downloaded from https://finance.yahoo.com/quote/TSLA/history?p=TSLA
dataset = "TSLA.csv"

# Add dataset into dataframe
df = pd.read_csv(dataset)

# Convert date to integer
df["Date_ordinal"] = pd.to_datetime(df['Date']).apply(lambda date: date.toordinal())

# Grab x and y values from dataset
x = df["Date_ordinal"].values
y = df["Close"].values

# Create train-test-split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Reshape array from 1D to 2D
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Create array of evenly spaced numbers between lowest and highest value of x_test
x_new = np.linspace(min(x_test), max(x_test), x_train.size).reshape(x_train.size, 1)

# Create predictions with n degrees polynomial (defined in variables)
lr = LinearRegression()
poly_features = PolynomialFeatures(degree=degrees, include_bias=False)
std_scaler = StandardScaler()
polynomial_regression = Pipeline([
    ("poly_features", poly_features),
    ("std_scaler", std_scaler),
    ("lin_reg", lr),
])
polynomial_regression.fit(x.reshape(-1, 1), y.reshape(-1, 1))
y_predict = polynomial_regression.predict(x_new)

# Plot graph
plt.xlabel("Date (converted to integer)")
plt.ylabel("Value [$]")
plt.plot(x_new, y_predict, "r", label="Prediction")
plt.plot(x, y, "g.", label="Real values")
plt.legend(loc="upper left")
plt.show()

# Calculate mean squared error value
print("MSE = " + str(mean_squared_error(y_train, y_predict)))
