import numpy as np
from numpy import array
import pandas as pd

import os
import glob

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# enumerate adds a number like [0, [22, 35, 42], ... ] to each sample
# creates data and then looks for row with max length, then adds 0 triplet until each row is the same length as max
# length
def create_all_data():
    data = []
    temp = []
    max_len = 0
    label_names = [f for f in os.listdir("HMP_Dataset")]
    for label in label_names:
        file_list = glob.glob(os.path.join(os.getcwd(), "HMP_Dataset/" + label, "*.txt"))
        for file in file_list:
            with open(file) as f:
                for line in f:
                    line = line.split()
                    line = [int(i) for i in line]
                    temp.append(line)
                data.append(temp)
                temp = []
    for row in data:
        if len(row) > max_len:
            max_len = len(row)
    for index, row in enumerate(data):
        while len(row) != max_len:
            data[index].append([0, 0, 0])
    return data


def create_labels():
    labels = []
    label_names = [f for f in os.listdir("HMP_Dataset")]
    for label in label_names:
        file_list = glob.glob(os.path.join(os.getcwd(), "HMP_Dataset/" + label, "*.txt"))
        for num in range(len(file_list)):
            labels.append(label)
    return labels


# data is a list of labels, turns data into array called values
# LabelEncoder turns the 'string' labels into labels between 0 and n where n is number of labels
# fit_transform actually takes in array of strings and turns them into numbers
# after this, it reshapes the array so that there is now a row for each label
# OneHotEncoder and fit_transform then turns the number that represents the label in each row into a one hot encoding
def create_onehot_labels(labels):
    data = labels
    values = array(data)
    le = LabelEncoder()
    num_labels = le.fit_transform(values)
    num_labels = num_labels.reshape(len(num_labels), 1)
    enc = OneHotEncoder(sparse=False, categories='auto')
    onehot_labels = enc.fit_transform(num_labels)
    return onehot_labels


def stratify(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, stratify=labels)
    return x_train, x_test, y_train, y_test


def create_np_labels():
    np_labels = create_onehot_labels(create_labels())
    return np_labels


def create_np_data(one_d, two_d, three_d):
    pd_data = pd.DataFrame(create_all_data()).values
    np_data = np.zeros((one_d, two_d, three_d))
    for i in range(one_d):
        for j in range(two_d):
            for k in range(three_d):
                np_data[i, j, k] = pd_data[i, j][k]
    np_data = np.reshape(np_data, (one_d, (two_d*three_d)))
    return np_data


def create_np_csv(two_d):
    #two_d max is 9318
    np_data = create_np_data(850, two_d, 3)
    np_labels = create_np_labels()
    x_train, x_test, y_train, y_test = train_test_split(np_data, np_labels, test_size=0.2, shuffle=True, stratify=np_labels)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)
    np.savetxt('x_train', x_train, delimiter=',', fmt='%0.0f')
    np.savetxt('x_test', x_test, delimiter=',', fmt='%0.0f')
    np.savetxt('y_train', y_train, delimiter=',', fmt='%0.0f')
    np.savetxt('y_test', y_test, delimiter=',', fmt='%0.0f')
    np.savetxt('x_val', x_val, delimiter=',', fmt='%0.0f')
    np.savetxt('y_val', y_val, delimiter=',', fmt='%0.0f')


# need window and more knowledge on how cnn works before implementing this
def create_cnn(size, num_cnn_layers):
    num_filters = 32
    kernel = (3,3)
    max_neurons = 64
    model = Sequential()
    for i in range(1, num_cnn_layers+1):
        if i == 1:
            model.add(Conv2D(num_filters*i, kernel, input_shape=size, activation='relu', padding='same'))
        else:
            model.add(Conv2D(num_filters * i, kernel, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(int(max_neurons), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(int(max_neurons/2), activation='relu'))
    model.add(Dense(14, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_fnn(x_train, y_train, input_dim, epochs):
    #input_dim is 27954
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
    model.add(Dense(units=64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(units=14, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)
    return model


def cross_val_fnn(x_train, y_train):
    model_list = []
    accuracy = []
    highest_accuracy = 0
    average_accuracy = 0
    best_model = None
    n_folds = 10
    print("======================================================================================")
    for i in range(n_folds):
        print("Training on Fold: ", i + 1)
        x_t, x_val, y_t, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=np.random.randint(1, 1000, 1)[0])
        model = create_fnn(x_t, y_t, 27954, 50)
        score, accuracy_t = model.evaluate(x_t, y_t)
        print("Val Score: " +str(score))
        print("Val Accuracy: " +str(accuracy_t))
        print()
        model_list.append(model)
        accuracy.append(accuracy_t)
    for accuracies in accuracy:
        if accuracies > highest_accuracy:
            highest_accuracy = accuracies
    for accuracies in accuracy:
        average_accuracy += accuracies
    average_accuracy = average_accuracy / (len(accuracy))
    best_model = model_list[accuracy.index(highest_accuracy)]
    return best_model, highest_accuracy, average_accuracy


x_train = pd.read_csv('x_train', header=None).values
y_train = pd.read_csv('y_train', header=None).values
x_val = pd.read_csv('x_val', header=None).values
y_val = pd.read_csv('y_val', header=None).values
x_test = pd.read_csv('x_test', header=None).values
y_test = pd.read_csv('y_test', header=None).values

sess = tf.Session()

best_mod, high_acc, avg_acc = cross_val_fnn(x_train, y_train)
print(best_mod)
print("Highest Training Accuracy: "+str(high_acc))
print("Average Training Accuracy: "+str(avg_acc))
sess.close()
