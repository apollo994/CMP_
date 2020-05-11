#!/usr/bin/env python3
#Fabio Zanarello, Sanger Institute, 2020

import sys
import argparse
from collections import defaultdict

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import matplotlib.pyplot as plt

import pickle

import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, auc


def convert_to_binary(y):
    pred_class = []
    pred_prob = []

    for i in range(len(y)):
        pred_prob.append(y[i][0])

        if y[i][0]>=0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)

    return np.asarray(pred_class), np.asarray(pred_prob)

def main():
    parser = argparse.ArgumentParser(description='My nice tool.')
    parser.add_argument('--exp', metavar='EXP', help='name of the experiment')


    args = parser.parse_args()

    ############################################################################
    #convolutional and pooling layers
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ############################################################################
    #fully connected layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    ############################################################################
    #compire the model

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    batch_size = 16


    ############################################################################
    #data upload

    # training with augmentation
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # test only rescale
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            args.exp+'/train/',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')

    # validation set is not shuffled
    validation_generator = validation_datagen.flow_from_directory(
            args.exp+'/validation/',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False)

    train_size = len(train_generator.filenames)
    validation_size = len(validation_generator.filenames)


    ############################################################################
    #model fit

    model.fit_generator(
        train_generator,
        steps_per_epoch= train_size // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps= validation_size // batch_size)


    ############################################################################
    #saving model

    #model arch and weigths
    model.save(f'{args.exp}/results/{args.exp}_model.h5')

    #history dictionary
    with open(f'{args.exp}/results/{args.exp}_history.pickle', 'wb') as file_pi:
            pickle.dump(model.history.history, file_pi)


    ############################################################################
    #training plots

    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,5), constrained_layout=True,)

    _ = fig.suptitle(f'{args.exp} trainig result')

    _ = ax1.set_title('Loss')
    _ = ax1.plot(model.history.history['loss'], label='train')
    _ = ax1.plot(model.history.history['val_loss'], label='test')
    _ = ax1.legend()

    _ = ax2.set_title('Accuracy')
    _ = ax2.plot(model.history.history['accuracy'], label='train')
    _ = ax2.plot(model.history.history['val_accuracy'], label='test')
    _ = ax2.legend()

    plt.savefig(f'{args.exp}/results/{args.exp}_trainig_result.png')


    ############################################################################
    #real vs predicted

    #real
    y_real = validation_generator.classes

    #predicted
    test_pred_prob = model.predict(validation_generator)
    y_pred_class, y_pred_prob = convert_to_binary(test_pred_prob)

    with open(f'{args.exp}/results/{args.exp}_real_pred.pickle', 'wb') as file_pi:
            pickle.dump((y_real,y_pred_class), file_pi)


    ############################################################################
    #prediction metrics

    precision = precision_score(y_real, y_pred_class)
    recall = recall_score(y_real, y_pred_class)

    fpr_th, tpr_th, _ = roc_curve(y_real, y_pred_prob)
    AUC_ROC = auc(fpr_th, tpr_th)

    precision_th, recall_th, _ = precision_recall_curve(y_real, y_pred_prob)
    PR_AUC = auc(recall_th, precision_th)

    with open(f'{args.exp}/results/{args.exp}_metrics.csv', 'w') as metrics:
        metrics.write('prec,rec,auROC,auPR\n')
        metrics.write(f'{precision},{recall},{AUC_ROC},{PR_AUC}\n')

    ############################################################################
    #prediction plots

    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,5), constrained_layout=True,)

    _ = fig.suptitle(f'{args.exp} predicions result')

    _ = ax1.set_title('ROC curve')
    _ = ax1.plot(fpr_th, tpr_th)
    _ = ax1.set_xlabel('False Positive Rate')
    _ = ax1.set_ylabel('True Positive Rate')


    _ = ax2.set_title('Precision-recall')
    _ = ax2.plot(recall_th, precision_th)
    _ = ax2.set_xlabel('Recall')
    _ = ax2.set_ylabel('Precision')


    plt.savefig(f'{args.exp}/results/{args.exp}_predictions_result.png')


if __name__ == "__main__":
  main()
