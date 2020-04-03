import warnings 
warnings.filterwarnings("ignore")

#Infrastructure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import NotFittedError

#DNN
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_radam.training import RAdamOptimizer

#Calibration
from sklearn.neighbors import KNeighborsClassifier

#Data Handling
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import (
    check_array,
    NotFittedError,
)
from sklearn.utils.multiclass import check_classification_targets

#Utils
import numpy as np

class HonestDNN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        calibration_split = .33
    ):
        self.calibration_split = calibration_split
        self.transformer_fitted_ = False
        self.voter_fitted_ = False

    def check_transformer_fit_(self):
        '''
        raise a NotFittedError if the transformer isn't fit
        '''
        if not self.transformer_fitted_:
                msg = (
                        "This %(name)s instance's transformer is not fitted yet. "
                        "Call 'fit_transform' or 'fit' with appropriate arguments "
                        "before using this estimator."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})
                
    def check_voter_fit_(self):
        '''
        raise a NotFittedError if the voter isn't fit
        '''
        if not self.voter_fitted_:
                msg = (
                        "This %(name)s instance's voter is not fitted yet. "
                        "Call 'fit_voter' or 'fit' with appropriate arguments "
                        "before using this estimator."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})
                
    def fit_transformer(self, X, y, epochs = 10):
        #format y
        check_classification_targets(y)
        
        
        self.network = keras.Sequential()
        self.network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=np.shape(X)[1:]))
        self.network.add(layers.AveragePooling2D())
        self.network.add(layers.BatchNormalization())
        self.network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        self.network.add(layers.AveragePooling2D())
        self.network.add(layers.BatchNormalization())
        self.network.add(layers.Flatten())
        
        self.network.add(layers.Dense(units=120, activation='relu'))
        self.network.add(layers.BatchNormalization())
        self.network.add(layers.Dense(units=16, activation='relu'))
        self.network.add(layers.BatchNormalization())
        self.network.add(layers.Dense(units=len(np.unique(y)), activation = 'softmax'))

        self.network.compile(loss = 'categorical_crossentropy', metrics=['acc'], optimizer = RAdamOptimizer())
        self.network.fit(X, 
                    keras.utils.to_categorical(y), 
                    epochs = epochs, 
                    callbacks = [ModelCheckpoint("best_model.h5", save_best_only = True)], 
                    verbose = False,
                    validation_split = .2)
        self.network.load_weights("best_model.h5")

        self.encoder = keras.models.Model(inputs = self.network.inputs, outputs = self.network.layers[-3].output)
        
        #make sure to flag that we're fit
        self.transformer_fitted_ = True
        
    def fit_voter(self, X, y):
        #format y
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        
        self.knn = KNeighborsClassifier(int(np.log2(len(X))))
        self.knn.fit(X, y)
        
        #make sure to flag that we're fit
        self.voter_fitted_ = True

    def fit(self, X, y, epochs = 10):
        #format y
        check_classification_targets(y)

        #split into train and cal
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size = self.calibration_split)
        
        #fit the transformer
        self.fit_transformer(X_train, y_train, epochs = epochs)

        #fit the voter
        X_cal_transformed = self.transform(X_cal)
        self.fit_voter(X_cal_transformed, y_cal)
        
    def transform(self, X):
        self.check_transformer_fit_()
        return self.encoder.predict(X)
    
    def get_transformer(self):
        self.check_transformer_fit_()
        return self.encoder

    def vote(self, X_transformed):
        self.check_voter_fit_()
        return self.knn.predict_proba(X_transformed)
    
    def get_voter(self):
        self.check_voter_fit_()
        return self.knn
    
    def decide(self, X_voted):
        return self.classes_[np.argmax(X_voted, axis = -1)]

    def predict_proba(self, X):
        return self.vote(self.transform(X))
    
    def predict(self, X):
        return self.decide(self.predict_proba(X))