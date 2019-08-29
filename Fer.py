# IMPORTS
from __future__ import division
import sys, os
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import seaborn as sns
import cv2
import warnings

warnings.filterwarnings("ignore")

IMG_width = 48
IMG_height = 48
num_labels = 7
batch_size = 64
epochs = 100


def processing_data(dataset_path):
    # read data
    data = pd.read_csv(dataset_path)

    # converting to list
    pixels = data['pixels'].tolist()

    # getting features for training
    X = []
    for i in pixels:
        x = [int(j) for j in i.split(' ')]
        x = np.asarray(x).reshape(IMG_width, IMG_height)
        X.append(x.astype('float64'))

    X = np.asarray(X)
    # expand dimension help CNNs
    X = np.expand_dims(X, -1)

    # getting emotions for training
    y = pd.get_dummies(data['emotion']).as_matrix()

    # storing data
    np.save('dataX', X)
    np.save('dataY', y)

    return X, y


def split_data(X, y):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    # splitting into training, validation and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # storing data
    np.save('dataX_train', X_train)
    np.save('dataY_train', y_train)
    np.save('dataX_test', X_test)
    np.save('dataY_test', y_test)

    return X_train, X_test, y_train, y_test

# model arhitecture
def get_model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(IMG_width, IMG_height, 1),
                     kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])
    return model


def save_model(file_json, file_weights):
    # saving the  model to be used later
    fer_json = model.to_json()
    with open(file_json, "w") as json_file:
        json_file.write(fer_json)
    model.save_weights(file_weights)
    print("Saved model to disk")


def load_model(path_json, path_weights):
    json_file = open(path_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_weights)
    print("Loaded model from disk")

    return loaded_model


def confusionMatrix(y_true, y_pred):
    confusionMatrix = confusion_matrix(y_true, y_pred)
    precision = confusionMatrix / confusionMatrix.sum(axis=1)

    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    title = 'Confusion matrix'
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusionMatrix, cmap="Blues", annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.xticks(rotation=90)
    plt.show()

    sns.heatmap(precision, cmap="Blues", annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.title("Precision Matrix", fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.xticks(rotation=90)
    plt.show()


def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save, reduce_lr_loss]


def k_fold(X_train, y_train):
    k = 5
    j = 0
    result = []

    print('Start KFold')
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=False)
    for train_idx, val_idx in kf.split(X_train):
        print('\nFold ', j)
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_valid_cv = X_train[val_idx]
        y_valid_cv = y_train[val_idx]

        name_weights = "final_model_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights=name_weights, patience_lr=10)

        model = get_model()
        model.fit(np.array(X_train_cv), np.array(y_train_cv),
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(np.array(X_valid_cv), np.array(y_valid_cv)),
                  shuffle=True,
                  callbacks=callbacks)

        j = j + 1
        print(model.evaluate(X_valid_cv, y_valid_cv))

    return model


def make_prediction(path, model):
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # loading image
    full_size_image = cv2.imread(path)

    print("Image Loaded")
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)
    print(faces)

    # detecting faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # predicting the emotion
        yhat = model.predict(cropped_img)
        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                    1, cv2.LINE_AA)
        print("Emotion: " + labels[int(np.argmax(yhat))])

    cv2.imshow('Emotion', full_size_image)
    cv2.waitKey()


X_1, y_1 = processing_data('./fer2013.csv')
X_2, y_2 = processing_data('./fer.csv')

X = np.concatenate((X_1, X_2))
y = np.concatenate((y_1, y_2))

X_train, X_test, y_train, y_test = split_data(X, y)

model = k_fold(X_train, y_train)

predy = np.argmax(model.predict(X_test), axis=1)
truey = np.argmax(y_test, axis=1)
print("Accuracy score = ", accuracy_score(truey, predy))

confusionMatrix(truey, predy)
