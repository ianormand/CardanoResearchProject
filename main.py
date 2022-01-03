from os import write
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt

# data taken from yahoo finance. 10/14/2020 through 10/11/2021
data = pd.read_csv(r"C:\Users\ianor\Downloads\ADA-USD.csv")

print(data)
print(data.columns)
predict = "Nxt Close"
date = "Date"

# make a new data set without the thing i want to predict, axis = 1
X = np.array(data.drop([predict, date, "Close"], 1))
Y = np.array(data[predict])


# divide into 4 arrays each using only 10% of the data so it wont memorize anything
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

print("x_train shape: ", X_train.shape)
print("y_train.shape: ", Y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", Y_test.shape)


linear = linear_model.LinearRegression()
print(linear)

linear.fit(X_train, Y_train)
# accuracy is calculated with .score() which finds the coefficient of determination aka the r^2 value
# r^2 = True
r_squared = linear.score(X_test, Y_test)


print("\nr_squared(r^2): ", r_squared, "\n")

print('Coefficient: \n', linear.coef_)
print('\nIntercept (b): \n', linear.intercept_, '\n')

predictions = linear.predict(X_test)

# print all the different dates' daily range, prediction, and actual value
missPercents = []
day_range_percents = []

#@TODO turn into a function, run 30 times and average the values to lower variablilty from different trials

for i in range(len(predictions)):
    # set up variables
    day_range = X_test.transpose()[1][i] - X_test.transpose()[2][i]
    day_range_percent = day_range / Y_test[i] * 100
    miss = abs(predictions[i] - Y_test[i])
    rangeMiss = miss/day_range * 100
    predYtest = predictions[i] / Y_test[i] * 100

    # print out all relevant metrics for each individual day
    print(f"Prediction[{i}]: ", predictions[i],
          f"X_test[{i}]",  X_test[i],
          f"Y_test[{i}]", Y_test[i],
          "\n",
          "daily range = ", day_range,
          'or', day_range_percent, '%',
          "\n",
          "miss = ", miss, 'or', 100 - predYtest, '%',
          "\nPrediction/Y_test: ", predYtest, '%')

    missPercents.append(miss)
    day_range_percents.append(day_range_percent)


# get sum of |predictions - y_test|
allMisses = 0
allRanges = 0
for i in range(len(missPercents)):
    allMisses = missPercents[i] + allMisses
    allRanges = day_range_percents[i] + allRanges

average_miss = (allMisses / len(predictions)) * 100
print("\naverage miss", average_miss, '%')
# mean absolute error
mean_absolute_error_percentage = (100 - allMisses / len(predictions) * 100)
print("Mean Absolute Error (MAE): ", mean_absolute_error_percentage, '%', '\n')

sum_daily_ranges = allRanges / len(day_range_percents)
print(f'Average daily range: {sum_daily_ranges}\n')

with open('first30.txt', 'a') as f:
    writeline_averages = '\nAverage Miss: {}\nMean Absolute Error (MAE): {}\nAverage daily range: {}'
    f.writelines(writeline_averages.format(average_miss, mean_absolute_error_percentage, sum_daily_ranges))
    # f.writelines('\nAverage Miss: ', average_miss)
    # f.writelines("\nMean Absolute Error (MAE): ", mean_absolute_error_percentage, '%')
    # f.writelines('\nAverage daily range: ', sum_daily_ranges)
        

while True:
    graph_input = input("See prediction/close(enter p) graph or year data graph(enter e): ")

    if graph_input.lower() == 'p':
        plt.style.use('seaborn-whitegrid')
        # import numpy as np

        fig = plt.figure()
        ax = plt.axes()

        list37 = []

        # print(data[date[-37:]])
        for i in range(37):
            list37.append(i)

        x = np.array(list37)

        y1 = np.array(Y_test)
        y2 = np.array(predictions)

        ax.plot(x, y1)

        plt.plot(x, y1, color='blue')
        plt.plot(x, y2, color='red')

        plt.show()

    elif graph_input.lower() == 'e':

        fig = plt.figure()
        ax = plt.axes()

        x = np.array(data[date])
        y = np.array(data[predict])

        ax.plot(x, y)

        plt.plot(x, y)

        # call graph to screen
        plt.show()
    elif graph_input == 'q':
        quit()
    else:
        continue
    continue



