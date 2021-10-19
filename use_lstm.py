# Stacked LSTM  with memory
import math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# workaround for import
#model = Sequential()


# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('datasets/ftp_normal.txt', usecols=[1])
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scale = MinMaxScaler(feature_range=(0, 1))
dataset = scale.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.01)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
batch_size = 1

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("models/lstm_model.h5")
# make predictions
# trainPredict = model.predict(trainX, batch_size=batch_size)
trainPredict = reconstructed_model.predict(trainX, batch_size=batch_size)
# model.reset_states()
reconstructed_model.reset_states()
start_time = datetime.now()
testPredict = reconstructed_model.predict(testX, batch_size=batch_size)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print("testPredict=", testPredict[0])
print("testY=", testY)
# confusion_matrix(y_true, testPredict)
# print("Confusion matrix", confusion_matrix(y_true, testPredict))
# tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
# print("tn fp fn tp", tn, fp, fn, tp)
#
# print("Confusion matrix", confusion_matrix(testY, testPredict))
#
# invert predictions
trainPredict = scale.inverse_transform(trainPredict)
trainY = scale.inverse_transform([trainY])
testPredict = scale.inverse_transform(testPredict)
print('Test predict inverse transform', testPredict)
testY = scale.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % testScore)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan

testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

# plot baseline and predictions
f = plt.figure()
ground_truth, = plt.plot(scale.inverse_transform(dataset))
train_predict, = plt.plot(trainPredictPlot)
test_predict, = plt.plot(testPredictPlot)
plt.legend([ground_truth, train_predict, test_predict], ['Ground Truth', 'Train Predict', 'Test Predict'])
plt.show()
f.savefig("plot.pdf")
plt.close()
exit(0)
