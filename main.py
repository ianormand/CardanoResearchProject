import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv(r"C:\Users\ianor\Downloads\ADA-USD.csv")

print(data)

predict = "Adj Close"
date = "Date"

#make a new data set without the thing i want to predict
X = np.array(data.drop([predict, date, "Close"], 1))
Y = np.array(data[predict])


#divide into 4 arrays each using 10% of the data so it wont memorize anything
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

print("x_train shape: ", X_train.shape)
print("y_train.shape: ", Y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", Y_test.shape)


linear = linear_model.LinearRegression()

linear.fit(X_train, Y_train)
#accuracy is calculated with .score() which finds the coefficient of determination aka the r^2 value
acc = linear.score(X_test, Y_test)
print("\naccuracy: ", acc, "\n")

print('Coefficient: \n', linear.coef_)
print('\nIntercept (b): \n', linear.intercept_, '\n')

predictions = linear.predict(X_test)
print("predicions.length", len(predictions))

for i in range(len(predictions)):

    print(f"Prediction[{i}]: ", predictions[i], f"X_test[{i}]",  X_test[i], f"Y_test[{i}]", Y_test[i], "\n",
          "daily range = ", X_test.transpose()[1][i] - X_test.transpose()[2][i], "\n", "miss = ",
          predictions[i] - Y_test[i])


plotting = True
while plotting:

    #plot of True prices over time
    # plot = "Date"
    # plt.scatter(data[plot][-37:], predictions[-37:], color="green")
    # plt.scatter(data[plot][-37:], data["Adj Close"][-37:], color="red")
    # plt.legend(loc=4)
    # plt.xlabel(plot)
    # plt.ylabel("predictions = green")
    # plt.show()

    # #plot of True prices over time
    # plot = "Date"
    # plt.scatter(data[plot], data["Adj Close"])
    # plt.legend(loc=4)
    # plt.xlabel(plot)
    # plt.ylabel("Adj Close")
    # plt.show()
    if input('If you want to see the graph again enter "Graph": ').lower() == "graph":
        plotting = True
    else:
        continue


