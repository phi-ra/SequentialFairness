import numpy as np
import statsmodels.api as sm
from itertools import permutations
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

from .wasserstein import MultiWasserStein

def unfairness(data1, data2):
    """
    compute the unfairness of two populations
    """
    x = np.linspace(min(min(data1), min(data2)), max(max(data1), max(data2)))
    ecdf1 = sm.distributions.ECDF(data1)(x)
    ecdf2 = sm.distributions.ECDF(data2)(x)
    unfair_value = np.max(np.abs(ecdf1-ecdf2))
    return unfair_value


def calculate_metrics(output_dict, 
                      y_test, 
                      objective='regression', 
                      threshold=None):
    
    metrics_dict = {}

    for key, value in output_dict.items():
        metrics_dict[key] = {}
        
        for level in value.keys():
            metrics_dict[key][level] = {}

            # First MSE
            prediction = output_dict[key][level]['prediction']

            if objective == 'regression':
                metrics_dict[key][level]['mse'] = mean_squared_error(y_test,
                                                                 prediction)
            elif objective == 'classification':
                predictions_ = np.where(prediction > threshold, 1, 0)

                metrics_dict[key][level]['accuracy'] = accuracy_score(y_test,
                                                                    predictions_)
                metrics_dict[key][level]['f1_score'] = f1_score(y_test,
                                                                    predictions_)

            
            unfair_tmp = 0
            for key_2, sens_feature_ in output_dict[key][level]['sensitive'].items():
                    
                id0 = np.where(sens_feature_ == 0)[0]
                id1 = np.where(sens_feature_ == 1)[0]

                pred_0 = prediction[id0]
                pred_1 = prediction[id1]

                unfair_tmp += unfairness(pred_0, pred_1)
                metrics_dict[key][level][f'unfairness_{key_2}']= unfairness(pred_0, pred_1)

            metrics_dict[key][level]['unfairness'] = unfair_tmp 

    return metrics_dict


def get_all_predictions(estimator, 
                        sensitive_feature_vector, 
                        data_dict):

    all_comb = permutations(sensitive_feature_vector, len(sensitive_feature_vector))

    output_dict = {}

    model_dict = {}
    for base_model in all_comb:
        level=0
        output_dict[base_model] = {}
        model_dict[base_model] = MultiWasserStein(estimator=estimator)

        for feature_ in base_model:
            model_dict[base_model].fit(X_calib=data_dict['X_calib'], 
                                       sensitive_name=feature_)
            response = (model_dict[base_model]
                                        .transform(X=data_dict['X_test'],
                                                   sensitive_name=feature_))
            output_dict[base_model][f'level_{level}'] = {}
            output_dict[base_model][f'level_{level}']['prediction'] = response

            sens_counter = 0
            output_dict[base_model][f'level_{level}']['sensitive'] = {}
            for subfeature_ in base_model:
                sens_tmp = data_dict['X_test'].loc[:,subfeature_]
                output_dict[base_model][f'level_{level}']['sensitive'][sens_counter] = sens_tmp
                sens_counter += 1
            
            model_dict[base_model].level += 1
            level += 1

    return output_dict
