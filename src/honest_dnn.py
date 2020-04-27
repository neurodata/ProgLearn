import warnings 
warnings.filterwarnings("ignore")

#Infrastructure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import NotFittedError

#DNN
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#Calibration
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression

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
        calibration_split = .5,
        iso_split = 0.5,
        n_jobs = -1
    ):
        self.calibration_split = calibration_split
        self.iso_split = iso_split
        self.transformer_fitted_ = False
        self.voter_fitted_ = False
        self.n_jobs = n_jobs

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
                
    def fit_transformer(self, X, y, epochs = 100, lr = 1e-4):
        #format y
        check_classification_targets(y)
        
        self.network = keras.Sequential()
        self.network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=np.shape(X)[1:]))
        self.network.add(layers.AveragePooling2D())
        self.network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        self.network.add(layers.AveragePooling2D())
        self.network.add(layers.Flatten())
        
        self.network.add(layers.Dense(units=120, activation='relu'))
        self.network.add(layers.Dense(units=84, activation='relu'))
        self.network.add(layers.Dense(units=len(np.unique(y)), activation = 'softmax'))

        self.network.compile(loss = 'categorical_crossentropy', metrics=['acc'], optimizer = keras.optimizers.Adam(lr))
        self.network.fit(X, 
                    keras.utils.to_categorical(y), 
                    epochs = epochs, 
                    callbacks = [ModelCheckpoint("best_model.h5", 
                                                 save_best_only = True, 
                                                 verbose = False),
                                 EarlyStopping(patience = 5)
                                ], 
                    verbose = False,
                    validation_split = .15)
        self.network.load_weights("best_model.h5")

        self.encoder = keras.models.Model(inputs = self.network.inputs, outputs = self.network.layers[-2].output)
        
        #make sure to flag that we're fit
        self.transformer_fitted_ = True
        
    def fit_voter(self, X, y):
        #format y
        check_classification_targets(y)
        
        self.knn = KNeighborsClassifier(int(np.log2(len(X))), n_jobs = self.n_jobs)
        self.knn.fit(X, y)
        
        #make sure to flag that we're fit
        self.voter_fitted_ = True
        
    def _get_reliability(self, cdf_hat, competences):
        B_m = []
        B_m_hat = []
        num_bins = int(np.log2(len(cdf_hat)))
        for i in range(num_bins):
            prop = i*1./num_bins
            inds = np.where((cdf_hat >= prop) & (cdf_hat <= prop+1./num_bins))[0]
            if len(inds) > 0:
                B_m.append(len(np.where(competences[inds] == 1)[0])*1./len(inds))
                B_m_hat.append(np.mean(cdf_hat[inds]))
            else:
                B_m.append(prop)
                B_m_hat.append(prop + 0.5/num_bins)
        return B_m, B_m_hat
        
    def fit_iso(self, X, y):
        y_hat_unscaled = self.predict_proba_unscaled(X)
        self.class_conditional_isos = []
        for class_idx, class_val in enumerate(self.classes_):
            cdf_hat_D = y_hat_unscaled[:, class_idx]
            classness_D = (y == class_idx).astype("uint32")
            B_m_D, B_m_hat_D = self._get_reliability(cdf_hat_D, classness_D)
            class_conditional_iso = IsotonicRegression().fit(B_m_hat_D, B_m_D)
            self.class_conditional_isos.append(class_conditional_iso)

    def fit(self, X, y, epochs = 30, lr = 1e-4):
        #format y
        check_classification_targets(y)

        #split
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size = self.calibration_split)
        X_train, X_iso, y_train, y_iso = train_test_split(X, y, test_size = self.iso_split)
        
        #fit the transformer
        self.fit_transformer(X_train, y_train, epochs = epochs, lr = lr)

        #fit the voter
        X_cal_transformed = self.transform(X_cal)
        self.fit_voter(X_cal_transformed, y_cal)
        
        #fit the isotonic regressor
        self.fit_iso(X_iso, y_iso)
        
    def transform(self, X):
        self.check_transformer_fit_()
        return self.encoder.predict(X)
    
    def get_transformer(self):
        self.check_transformer_fit_()
        return self.encoder

    def vote(self, X_transformed):
        self.check_voter_fit_()
        return self.knn.predict_proba(X_transformed)
    
    def scale(self, y_hat_proba):
        y_hat_proba_scaled = np.zeros_like(y_hat_proba)
        for class_idx in range(len(y_hat_proba_scaled[0])):
            class_conditional_probas = y_hat_proba[:, class_idx]
            class_conditional_probas_scaled = self.class_conditional_isos[class_idx].predict(class_conditional_probas)
            y_hat_proba_scaled[:, class_idx] = class_conditional_probas_scaled
        return y_hat_proba_scaled 
    
    def get_voter(self):
        self.check_voter_fit_()
        return self.knn
    
    def decide(self, X_voted):
        return np.argmax(X_voted, axis = -1)

    def predict_proba_unscaled(self, X):
        return self.vote(self.transform(X))
    
    def predict_proba(self, X):
        return self.predict_proba_unscaled(X)
    
    def predict(self, X):
        return self.decide(self.predict_proba(X))