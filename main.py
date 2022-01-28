from os import write
from csv import writer
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import r2_score

# data taken from yahoo finance. 10/14/2020 through 10/11/2021
data = pd.read_csv(r"C:\Users\ianor\Downloads\ADA-USD.csv")

print(data)
print(data.columns)
predict = "Nxt Close"
date = "Date"

# make a new data set without the thing i want to predict, axis = 1
X = np.array(data.drop([predict, date, "Close"], 1))
Y = np.array(data[predict])


# divide into 4 arrays, each using only 10% of the data, so it wont memorize anything
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

print("x_train shape: ", X_train.shape)
print("y_train.shape: ", Y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", Y_test.shape)


linear = linear_model.LinearRegression()
print(linear)

linear.fit(X_train, Y_train)
r_squared = linear.score(X_test, Y_test)

print("\nr_squared(r^2): ", r_squared, "\n")

print('Coefficient: \n', linear.coef_)
print('\nIntercept (b): \n', linear.intercept_, '\n')

predictions = linear.predict(X_test)

# print all the different dates' daily range, prediction, and actual value
missPercents = []
day_range_percents = []
r_squared_list = []

for i in range(len(predictions)):
    # set up variables
    day_range = X_test.transpose()[1][i] - X_test.transpose()[2][i]
    day_range_percent = day_range / Y_test[i] * 100
    miss = abs(predictions[i] - Y_test[i])
    rangeMiss = miss/day_range * 100
    predYtest = predictions[i] / Y_test[i] * 100
    r_squared = r2_score(Y_test, predictions)

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
    r_squared_list.append(r_squared)

# get sum of |predictions - y_test|
allMisses = sum(missPercents)
allRanges = sum(day_range_percents)
allR_squared = sum(r_squared_list)

average_miss = (allMisses / len(predictions)) * 100
print("\naverage miss", average_miss, '%')
average_r_squared = allR_squared / len(predictions)
print("Average of R^2 Values: ", average_r_squared)
sum_daily_ranges = allRanges / len(day_range_percents)
print(f'Average daily range: {sum_daily_ranges}\n')


header = ['Average Miss', 'Average r^2', 'Average daily range']

#ONLY USED FOR THE FIRST 30 TIMES

# with open('filtered_data.csv', 'a', newline='') as f:
#     writer = csv.writer(f)

#     # write the data
#     csv_data = [average_miss, average_r_squared, sum_daily_ranges]
#     writer.writerow(csv_data)

df2=pd.read_csv('filtered_data.csv')
print("average of ALL Average Misses: ", df2['Average Miss'].sum() / len(df2["Average Miss"]), "%")
print("average of ALL r^2: ", df2['Average r^2'].sum() / len(df2['Average r^2']), "%")
print("average of ALL Average Daily Ranges: ", df2['Average daily range'].sum() / len(df2['Average daily range']), "%")


while True:
    graph_input = input("See prediction/close(enter p) graph or year data graph(enter e): ")

    if graph_input.lower() == 'p':
        plt.style.use('seaborn-whitegrid')

        fig = plt.figure()
        ax = plt.axes()

        plt.title("Red = Machine Learning Model's Predictions; Blue = Real Time Prices")

        plt.xlabel("Time (Days)")
        plt.ylabel("Price (USD)")

        list37 = []

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

        plt.title("Unproccessed Input Data (364 days)")

        plt.xlabel("Time (Days)")
        plt.ylabel("Price (USD)")

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



