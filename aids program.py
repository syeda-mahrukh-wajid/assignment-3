import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('aids1.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('death due to aids (Training set)')
plt.xlabel('Years')
plt.ylabel('death')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('death due to aids(Test set)')
plt.xlabel('Years')
plt.ylabel('death')
plt.show()

print(regressor.predict([[1996]]))