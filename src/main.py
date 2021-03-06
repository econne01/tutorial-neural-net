from os import path
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
RANDOM_SEED = 7
numpy.random.seed(RANDOM_SEED)


def get_data():
    datafile = path.join(Path(".").resolve(), 'data/pima-indians-diabetes.data.csv')
    # load pima indians dataset
    dataset = numpy.loadtxt(datafile, delimiter=',')
    return dataset

def get_model():
    # create model
    model = Sequential()
    # Each layer of the model has parameters like
    # Dense(NUM_OF_NEURONS, ...)
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, training_set_inputs, training_set_outputs):
    # Fit the model
    model.fit(training_set_inputs, training_set_outputs, epochs=150, batch_size=20)

def evaluate_model(model, test_set_inputs, test_set_outputs):
    # evaluate the model
    scores = model.evaluate(test_set_inputs, test_set_outputs)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def predict(model, test_set_inputs, test_set_outputs):
    # predict the outputs by using the model
    predictions = model.predict(test_set_inputs)
    for i in range(len(predictions)):
        print("%d. Prediction: %.2f%%, Actual: %d" % (i, predictions[i][0], test_set_outputs[i]))

if __name__ == '__main__':
    print('BEGIN data processing')
    dataset = get_data()
    # split into input (X) and output (Y) variables
    # (aka, the input data (X) will be used to predict the dependent variables (Y))
    print('Dataset Length:', len(dataset))
    # Test our model on the final 10% of dataset entries
    training_split = int(len(dataset) * 0.9)
    trainingX = dataset[:training_split,0:8]
    trainingY = dataset[:training_split,8]

    model = get_model()
    train_model(model, trainingX, trainingY)

    testX = dataset[training_split:,0:8]
    testY = dataset[training_split:,8]

    evaluate_model(model, testX, testY)

    predict(model, testX, testY)

    print('END data processing')
