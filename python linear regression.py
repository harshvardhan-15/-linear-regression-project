# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Features (e.g., hours studied)
y = 3.5 * X + np.random.randn(100, 1) * 2  # Labels (e.g., scores)

# Convert to Pandas DataFrame for visualization
data = pd.DataFrame({"Hours_Studied": X.flatten(), "Scores": y.flatten()})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize the results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.title("Linear Regression Example")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.show()

