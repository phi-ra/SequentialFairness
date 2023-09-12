import numpy as np
from itertools import permutations

from statsmodels.distributions.empirical_distribution import ECDF
from .aux import EQF

class MultiWasserStein:
    """
    Baseclass to calculate sequential, multiple Wasserstein Barycenters

    Parameter
    ---------
    estimator: A base estimator that can call either predict() in the case of
        regression or predict_proba in the casse of classification
    sigma: Optional(default=0.001), The jitter value applied when calculating 
        the ECDF and EQF
    """
    def __init__(self, 
                 estimator, 
                 sigma=0.001) -> None:
        self.estimator = estimator
        self.sigma = sigma

        self.level = 0
        self.sensitive_values = {}
        self.weights = {}

        self.eqf_dict = {}
        self.ecdf_dict = {}

        self.calib_pred = {}
        self.pred = {}
        self.pred_0 = {}
        self.pred_1 = {}

        self.fair_calib ={}
        self.fair = {}
        self.fair_0 = {}
        self.fair_1 = {}
        
    def fit(self,
            X_calib,
            sensitive_name=None, 
            sensitive_idx=None,
            objective='regression') -> None:
        
        if self.level == 0:            
            if objective == 'regression':
                self.calib_pred[f'level_{self.level}'] = self.estimator.predict(X_calib)
                prediction_unlabeled = self.calib_pred[f'level_{self.level}']
            elif objective == 'classification':
                self.calib_pred[f'level_{self.level}'] = self.estimator.predict_proba(X_calib)
                prediction_unlabeled = self.calib_pred[f'level_{self.level}']
        else:
            prediction_unlabeled = self.fair_calib[f'level_{int(self.level-1)}']
            self.calib_pred[f'level_{self.level}'] = prediction_unlabeled

        sensitive_idx = self._check_args_fit(sensitive_name, sensitive_idx, X_calib)

        iw0 = np.where(X_calib.iloc[:,sensitive_idx]==0)[0]
        iw1 = np.where(X_calib.iloc[:,sensitive_idx]==1)[0]

        w0 = len(iw0)/X_calib.shape[0]
        self.weights[sensitive_idx] = np.array([w0, 1-w0])

        eps = np.random.uniform(-self.sigma,
                                self.sigma,
                                len(prediction_unlabeled))
        
        # Fit the ecdf and eqf objects
        self.eqf_dict[f'level_{self.level}'] = {}
        self.eqf_dict[f'level_{self.level}'][0] = EQF(prediction_unlabeled[iw0]+eps[iw0])
        self.eqf_dict[f'level_{self.level}'][1] = EQF(prediction_unlabeled[iw1]+eps[iw1])

        self.ecdf_dict[f'level_{self.level}'] = {}
        self.ecdf_dict[f'level_{self.level}'][0] = ECDF(prediction_unlabeled[iw0]+eps[iw0])
        self.ecdf_dict[f'level_{self.level}'][1] = ECDF(prediction_unlabeled[iw1]+eps[iw1])

        # Run fair prediction on calibration for next level
        self.fair_calib[f'level_{int(self.level)}'] =  self.transform(X=X_calib, 
                                                                      sensitive_idx=sensitive_idx,
                                                                      mode='calibration')

    def transform(self, 
                  X, 
                  sensitive_name=None, 
                  sensitive_idx=None,
                  mode='evaluation',
                  epsilon=0, 
                  objective='regression'):
        sensitive_idx = self._check_args_fit(sensitive_name, sensitive_idx, X)
        #self.epsilon[sensitive_idx] = epsilon

        if self.level == 0:
            if objective == 'regression':
                prediction = self.estimator.predict(X)

            elif objective == 'classification':
                prediction = self.estimator.predict_proba(X)
        else:
            if mode == 'calibration':
                prediction = self.fair_calib[f'level_{int(self.level-1)}']
            elif mode == 'evaluation':
                prediction = self.fair[f'level_{int(self.level-1)}']
            else:
                raise ValueError('Need to specify either evaluation or calibration')
            
        # Recalculate weights and split predictions
        iw0 = np.where(X.iloc[:,sensitive_idx]==0)[0]
        iw1 = np.where(X.iloc[:,sensitive_idx]==1)[0]

        pred_0 = prediction[iw0]
        pred_1 = prediction[iw1]

        # Initialize
        pred_fair_0 = np.zeros_like(pred_0)
        pred_fair_1 = np.zeros_like(pred_1)

        # Calculate
        eps = np.random.uniform(-self.sigma,
                                self.sigma,
                                len(prediction))

        # Run 
        pred_fair_0 += (self.weights[sensitive_idx][0] *
                        self.eqf_dict[f'level_{self.level}'][0](self.ecdf_dict[f'level_{self.level}'][0](pred_0 + eps[iw0])))
        pred_fair_0 += (self.weights[sensitive_idx][1] *
                        self.eqf_dict[f'level_{self.level}'][1](self.ecdf_dict[f'level_{self.level}'][0](pred_0 + eps[iw0])))
        
        pred_fair_1 += (self.weights[sensitive_idx][0] *
                        self.eqf_dict[f'level_{self.level}'][0](self.ecdf_dict[f'level_{self.level}'][1](pred_1 + eps[iw1])))
        pred_fair_1 += (self.weights[sensitive_idx][1] *
                        self.eqf_dict[f'level_{self.level}'][1](self.ecdf_dict[f'level_{self.level}'][1](pred_1 + eps[iw1])))
        
        # Recombine
        pred_fair = np.zeros_like(prediction)
        pred_fair[iw0] = pred_fair_0
        pred_fair[iw1] = pred_fair_1

        if mode == 'evaluation':
            print('saving mods')
            self.pred[f'level_{self.level}'] = prediction
            self.pred_0[f'level_{self.level}'] = pred_0
            self.pred_1[f'level_{self.level}'] = pred_1

            self.fair[f'level_{self.level}'] = pred_fair
            self.fair_0[f'level_{self.level}'] = pred_fair_0
            self.fair_1[f'level_{self.level}'] = pred_fair_1

        if mode == 'calibration':
            return pred_fair
        else:
            return (1-epsilon)*pred_fair + epsilon*prediction
        
    def _check_args_fit(self, sens_name, sens_idx, data):
        if sens_name is None and sens_idx is None:
            raise ValueError('Specify either idx or name')

        if sens_name is not None and sens_idx is not None:
            raise ValueError('Specify either idx or name, not both')
        
        if sens_name is not None:
            sens_idx = np.where(data.columns == sens_name)[0][0]

        return sens_idx
    

class WassersteinFairRegression():
    def __init__(self, method, sigma = 0.0001):
        self.sigma = sigma
        self.method = method
        self.ri = dict()
        self.weights = dict()
        self.values = dict()
        self.y_pred = dict()
        self.y_pred0 = dict()
        self.y_pred1 = dict()
        self.y_pred_fair = dict()
        self.y_pred_fair0 = dict()
        self.y_pred_fair1 = dict()
    
    def fit(self,
            X_calib,
            objective='regression',
            sensitive_name=None,
            sensitive_idx=None): # a_index, # ri: value between 0 (fair) and 1 (unfair)
        """
        ToDo: Make work with numpy arrays
        """
        # Run argcheck first
        sensitive_idx = self._check_args_fit(sensitive_name, sensitive_idx, X_calib)
    
        sens_val_0, sens_val_1 = set(X_calib.iloc[:,sensitive_idx])
        self.values[sensitive_idx] = [sens_val_0, sens_val_1]
    
        iw0 = np.where(X_calib.iloc[:,sensitive_idx]==sens_val_0)[0]
        iw1 = np.where(X_calib.iloc[:,sensitive_idx]==sens_val_1)[0]

        w0 = len(iw0)/X_calib.shape[0]
        self.weights[sensitive_idx] = np.array([w0, 1-w0])

        if objective == 'regression':
            y_pred_unlab = self.method.predict(X_calib)
        elif objective == 'classification':
            y_pred_unlab = self.method.predict_proba(X_calib)

        eps = np.random.uniform(-self.sigma, self.sigma, len(y_pred_unlab))
        # Fit the ecdf and eqf objects
        self.ecdf0 = ECDF(y_pred_unlab[iw0]+eps[iw0])
        self.ecdf1 = ECDF(y_pred_unlab[iw1]+eps[iw1])
        self.eqf0 = EQF(y_pred_unlab[iw0]+eps[iw0])
        self.eqf1 = EQF(y_pred_unlab[iw1]+eps[iw1])

    def predict(self,
                X,
                sensitive_name=None,
                sensitive_idx=None,
                epsilon = 0, 
                objective='regression'):
        
        sensitive_idx = self._check_args_fit(sensitive_name, sensitive_idx, X)

        self.ri[sensitive_idx] = epsilon

        if objective == 'regression':
            self.y_pred[sensitive_idx] = self.method.predict(X)
        elif objective == 'classification':
            self.y_pred[self.sensitive_idx] = self.method.predict_proba(X)

        iw0 = np.where(X.iloc[:,sensitive_idx]==self.values[sensitive_idx][0])[0]
        iw1 = np.where(X.iloc[:,sensitive_idx]==self.values[sensitive_idx][1])[0]

        self.y_pred0[sensitive_idx] = self.y_pred[sensitive_idx][iw0]
        self.y_pred1[sensitive_idx] = self.y_pred[sensitive_idx][iw1]
        self.y_pred_fair0[sensitive_idx] = np.zeros_like(self.y_pred0[sensitive_idx])
        self.y_pred_fair1[sensitive_idx] = np.zeros_like(self.y_pred1[sensitive_idx])
        
        eps = np.random.uniform(-self.sigma, self.sigma, len(self.y_pred[sensitive_idx]))
        self.y_pred_fair0[sensitive_idx] += self.weights[sensitive_idx][0] * self.eqf0(self.ecdf0(self.y_pred0[sensitive_idx]+eps[iw0]))
        self.y_pred_fair0[sensitive_idx] += self.weights[sensitive_idx][1] * self.eqf1(self.ecdf0(self.y_pred0[sensitive_idx]+eps[iw0]))
        self.y_pred_fair1[sensitive_idx] += self.weights[sensitive_idx][0] * self.eqf0(self.ecdf1(self.y_pred1[sensitive_idx]+eps[iw1]))
        self.y_pred_fair1[sensitive_idx] += self.weights[sensitive_idx][1] * self.eqf1(self.ecdf1(self.y_pred1[sensitive_idx]+eps[iw1]))
        
        # Recombine
        self.y_pred_fair[sensitive_idx] = np.zeros_like(self.y_pred[sensitive_idx])
        self.y_pred_fair[sensitive_idx][iw0] = self.y_pred_fair0[sensitive_idx]
        self.y_pred_fair[sensitive_idx][iw1] = self.y_pred_fair1[sensitive_idx]

        return (1-self.ri[sensitive_idx]) * self.y_pred_fair[sensitive_idx] + self.ri[sensitive_idx] * self.y_pred[sensitive_idx]
    
    def _check_args_fit(self, sens_name, sens_idx, data):
        if sens_name is None and sens_idx is None:
            raise ValueError('Specify either idx or name')

        if sens_name is not None and sens_idx is not None:
            raise ValueError('Specify either idx or name, not both')
        
        if sens_name is not None:
            sens_idx = np.where(data.columns == sens_name)[0][0]

        return sens_idx
