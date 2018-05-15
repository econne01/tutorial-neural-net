from os import path
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
RANDOM_SEED = 7
numpy.random.seed(RANDOM_SEED)


def load_data():
    datafile = path.join(Path(".").resolve(), 'data/pima-indians-diabetes.data.csv')
    # load pima indians dataset
    dataset = numpy.loadtxt(datafile, delimiter=',')
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]

if __name__ == '__main__':
    print('BEGIN data processing')
    load_data()

    print('END data processing')
