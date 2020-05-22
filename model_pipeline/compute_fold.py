#!/usr/bin/env python3
#Fabio Zanarello, Sanger Institute, 2020

import sys
import argparse
from collections import defaultdict

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import matplotlib.pyplot as plt

import pandas as pd

import pickle
import os

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
    parser.add_argument('--ep', metavar='EP', default=50 , type=int, help='numer of epoch')
    parser.add_argument('--pos', metavar='POS', help='positive label')
    parser.add_argument('--info', metavar='INFO_TAB', help='table containig img name and feature')


    args = parser.parse_args()

    folds = [dI for dI in os.listdir(args.exp) if os.path.isdir(os.path.join(args.exp,dI))]
    folds = [f for f in folds if f[0]=='f']
    folds = sorted(folds)

    histories = []
    reals = []
    preds_p = []
    preds_c = []

    ############################################################################
    #setting positive label

    lab_df = pd.read_csv(args.info)
    labels = list(set(lab_df.ft))
    labels_cp = labels

    print (labels)

    a = 0

    for i in labels:
        if args.pos == i:
            positive_label = i
            labels_cp.remove(positive_label)
            negative_label = labels_cp[0]

            print(f'Positive label= {positive_label}')
            print(f'Negative label= {negative_label}')

            a = 1

    if a != 1:
        print (f'POSITIVE LABEL NOT FOUND!!!')
        sys.exit()



    ############################################################################

    for fold in folds:
        print ('---------------------------------------------------------------')
        print (f'working on {fold}')

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

        METRICS = [
              keras.metrics.TruePositives(name='tp'),
              keras.metrics.FalsePositives(name='fp'),
              keras.metrics.TrueNegatives(name='tn'),
              keras.metrics.FalseNegatives(name='fn'),
              keras.metrics.BinaryAccuracy(name='accuracy'),
              keras.metrics.Precision(name='precision'),
              keras.metrics.Recall(name='recall'),
              keras.metrics.AUC(name='auc'),
        ]


        model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=METRICS)

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
                f'{args.exp}/{fold}/train/',
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='binary',
                classes = [negative_label, positive_label]
                )

        # validation set is not shuffled
        validation_generator = validation_datagen.flow_from_directory(
                f'{args.exp}/{fold}/validation/',
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False,
                classes = [negative_label, positive_label]
                )

        train_size = len(train_generator.filenames)
        validation_size = len(validation_generator.filenames)


        ############################################################################
        #model fit

        model.fit_generator(
            train_generator,
            steps_per_epoch= train_size // batch_size,
            epochs=args.ep,
            validation_data=validation_generator,
            validation_steps= validation_size // batch_size)


        ############################################################################
        #saving model

        #model arch and weigths
        model.save(f'{args.exp}/results/{args.exp}_{fold}_model.h5')

        ############################################################################
        #training plots

        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,5), constrained_layout=True,)

        _ = fig.suptitle(f'{args.exp} {fold} trainig result')

        _ = ax1.set_title('Loss')
        _ = ax1.plot(model.history.history['loss'], label='train')
        _ = ax1.plot(model.history.history['val_loss'], label='test')
        _ = ax1.legend()

        _ = ax2.set_title('Accuracy')
        _ = ax2.plot(model.history.history['accuracy'], label='train')
        _ = ax2.plot(model.history.history['val_accuracy'], label='test')
        _ = ax2.legend()

        plt.savefig(f'{args.exp}/results/{args.exp}_{fold}_trainig_result.png')

        ########################################################################
        #appending results lists

        #real
        y_real = validation_generator.classes
        reals.append(y_real)

        #preds
        test_pred_prob = model.predict(validation_generator)
        y_pred_class, y_pred_prob = convert_to_binary(test_pred_prob)
        preds_c.append(y_pred_class)
        preds_p.append(y_pred_prob)

    ############################################################################

    #saving lists

    with open(f'{args.exp}/results/{args.exp}_real.pk', 'wb') as real_pi:
                pickle.dump(reals, real_pi)

    with open(f'{args.exp}/results/{args.exp}_pred_p.pk', 'wb') as pred_p_pi:
                pickle.dump(preds_p, pred_p_pi)

    with open(f'{args.exp}/results/{args.exp}_pred_c.pk', 'wb') as pred_c_pi:
                pickle.dump(preds_c, pred_c_pi)




if __name__ == "__main__":
  main()
