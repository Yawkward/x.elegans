#!/usr/bin/env python
# coding: utf-8

# ## n_job for grid searchCV & randomforest!

# In[7]:


from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler as zscore # zscore
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso #LRlasso
from collections import OrderedDict
from joblib import dump, load #to save models in files
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import re
import json
import os



from sklearn.model_selection import GridSearchCV
def gridcv(X, y, model, param_grid, naimpute=False, prepy=True, scorer = 'neg_mean_squared_error', cv_meth = LeaveOneOut(), cv_n_jobs = 1):
    """
    Perform Cross-Validation (defaukt: LOOCV) with hyperparameter tuning using GridSearchCV.
    
    Parameters:
    ----------
    X : pandas DataFrame or numpy array
        The feature matrix.
        
    y : pandas Series or numpy array
        The target variable.
        
    model : scikit-learn estimator
        The machine learning model to be used, should be an uninitialized model instance 
        (e.g., Lasso(), not Lasso(alpha=1.0)).
        
    param_grid : dict
        Dictionary containing the hyperparameters to be tuned and their possible values. 
        The keys should be prefixed with 'regressor__' to work with the pipeline.
        
    naimpute : bool, optional (default=False)
        Toggle imputation for missing values. 
        Currently not implemented; will print a message and return 0 if set to True.
        
    prepy : bool, optional (default=True)
        Toggle preprocessing target variable 'y' by setting any negative values to zero.
        
    scorer : str, callable, or None, optional (default='neg_mean_squared_error')
        A string or a scorer callable object / function with signature scorer(estimator, X, y). 
        For valid scoring strings, see the scikit-learn documentation.
        
    cv_meth : cross-validation generator, optional (default=LeaveOneOut())
        A cross-validation splitting strategy. 
        Possible inputs for cv are integers to specify the number of folds in a (Stratified)KFold, 
        CV splitter, cross-validation generator iterators, or an iterable.
        
    Returns:
    -------
    overall_metric : dict
        Dictionary containing the overall metrics and other details from the GridSearchCV.
        
    out_model : GridSearchCV object
        Fitted GridSearchCV object.
        
    best_params : dict
        Dictionary containing the best hyperparameters found by GridSearchCV.

    Call:
    ------
    from sklearn.model_selection import KFold

    # set up KFold cross-validator
    kfold_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    param_grid = {
        'regressor__alpha': np.array(np.arange(0.0125, 0.0425, 0.0025)),
        'regressor__fit_intercept': [True, False]
    }
    print(param_grid)

    # Call the gridcv function with KFold as the cross-validation method
    lasso_fullkfold_scores, lasso_fullkfold_model, best_param = gridcv(
        X, 
        y,
        Lasso(max_iter=4000),
        param_grid,
        scorer='r2', 
        cv_meth=kfold_cv
    )
    dump(lasso_fullkfold_model, './models/lasso_fullkfold_model.pkl') # save the model as .pkl
    """

    # overall_metric = {'CV': cv_meth, 'scoring_metric': scorer} originally
    overall_metric = {'CV': str(cv_meth), 'scoring_metric': str(scorer)} # transformed to string because json dump scores later

    if prepy:
        y[y < 0] = 0
    
    if naimpute:
      print("not implemented")
      return 0


    pipeline = Pipeline([
        ('scaler', zscore()), 
        ('regressor', model)        # Regression model
    ])

    
    # declaring an Grid object
    # score : https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    out_model = GridSearchCV(pipeline, param_grid=param_grid, cv=cv_meth, scoring=scorer, n_jobs=cv_n_jobs).fit(X,y)
    # GridSearchCV need the regressor__ prefix for the pipiline object in the para_grid later when called

    best_pipeline = out_model.best_estimator_
    y_pred = best_pipeline.predict(X)

    overall_metric['correlation_true_pred'] = list(np.corrcoef(list(y), list(y_pred)))
    overall_metric['correlation_true_pred'][0] = list(overall_metric['correlation_true_pred'][0])
    overall_metric['correlation_true_pred'][1] = list(overall_metric['correlation_true_pred'][1])


    # LOOCV folds: split{i}_test_score (number of data points minus one) 
    overall_metric['fold_scores'] = [out_model.cv_results_[f'split{i}_test_score'][out_model.best_index_] for i in range(out_model.n_splits_)]
    best_params = out_model.best_params_


    # printing section
    print("best parameter from gridsearch>>\n", out_model.best_params_)
    print(overall_metric['CV'])
    print(overall_metric['scoring_metric'])
    print("correlation Matrix>>\n", overall_metric['correlation_true_pred'])
    print("scores for each fold>>\n",overall_metric['fold_scores'])

    if str(model).startswith("Lasso"):
        # access the 'regressor' step from the best pipeline and then its coefficients
        coefficients = best_pipeline.named_steps['regressor'].coef_
        overall_metric['non_zero_coefficients'] = coefficients[coefficients != 0]
        overall_metric['non_zero_coefficients'] = overall_metric['non_zero_coefficients'].tolist()
        overall_metric['non_zero_features'] = list(X.columns[np.where(coefficients != 0)[0]])
        print("non_zero_features>>\n",overall_metric['non_zero_features'])

    if str(model).startswith("RandomForestRegressor"):
        
        feature_names = X.columns
        feature_importances = best_pipeline.named_steps['regressor'].feature_importances_
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        overall_metric['feature_importances'] = OrderedDict(sorted_feature_importance)
        print("feature_importances>>\n",overall_metric['feature_importances'])
       

    return overall_metric, out_model, best_params




# ### REF 1
#         #if np.mean(scores['fold_scores']) > 0.3:
#         #    print(f"\n >> TRUE, mean fold scores {np.mean(scores['fold_scores'])} is bigger than tresh << \n")
#             # select feature based on cumulative importance
#         #    cumulative_importance = 0.0
#         #   selected_features = []
#         #    for feature, importance in scores['feature_importances'].items():
#         #        cumulative_importance += importance
#         #        selected_features.append(feature)
#         #        if cumulative_importance >= 0.95:
#         #           break
#         #    cv_results['selected_features'][ran_state] = selected_features
#         # cv_results['model'][ran_state] = model
# 
#     
#     # Determine common features selected on cumulative importance
#     #first_key = list(cv_results['selected_features'])[0]
#     #cv_results['common_features'] = set(cv_results['selected_features'][first_key])
# 
#     #for r in list(cv_results['selected_features'].keys())[1:]:
#     #    current_features = set(cv_results['selected_features'][r])
#     #    cv_results['common_features'] = cv_results['common_features'].intersection(current_features)
#     #cv_results['common_features'] = list(cv_results['common_features'])

# In[40]:


def convert_type(obj):
    """
    Converts an type of numpy into a python inherent type
    This function can be used in combination with json.dump:
    ----
    Usage Example:
    
    with open(f"{output_path}{output_prefix}_nXcv.json", 'w') as file:
       json.dump(cv_results, file, default=convert_type)
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# In[41]:


def nX_cross_validation(X, target, param_grid, scorer_estimate, output_prefix, random_states, output_path='./models/10xKfold/', n_splits=3, cv_n_jobs=1, regr_n_job=1):
    if os.path.exists(output_path):
        print(f"The path {output_path} exists.")
    else:
        print(f"The path {output_path} does not exist.")
        raise FileNotFoundError(f"The path {output_path} does not exist.")
    best_fold_mean = float('-inf')
    best_model = []

    #cv_results = {'random_state': [], 'scores': {}, 'mean_scores': [], 'selected_features': {}, 'common_features': {}, 'model': {}}
    cv_results = {'random_state': [], 'scores': {}, 'mean_scores': [], 'selected_features': {}, 'best_param': []}

    for ran_state in random_states:
        print(ran_state)
        kfold_cv = KFold(n_splits=n_splits, shuffle=True, random_state=ran_state)
        scores, model, best_param = gridcv(
            X, 
            target,
            RandomForestRegressor(n_jobs=regr_n_job),
            param_grid,
            prepy=False,
            scorer=scorer_estimate, 
            cv_meth=kfold_cv,
            cv_n_jobs=cv_n_jobs
        )
        cv_results['random_state'].append(ran_state)
        cv_results['scores'][ran_state] = scores
        cv_results['mean_scores'].append(np.mean(scores['fold_scores']))
        if best_fold_mean < np.mean(scores['fold_scores']):
            best_fold_mean = np.mean(scores['fold_scores'])
            cv_results['best_param'] = best_param, ran_state, np.mean(scores['fold_scores'])

    # REF 1

    print(f"best estimator>>\n found in split: {cv_results['best_param'][1]}\n param_grid: {cv_results['best_param'][0]}\n mean fold score {cv_results['best_param'][2]}")    
    regr = RandomForestRegressor(max_features=cv_results['best_param'][0]['regressor__max_features'], n_estimators=cv_results['best_param'][0]['regressor__n_estimators'], bootstrap=cv_results['best_param'][0]['regressor__bootstrap'], n_jobs=regr_n_job)
    best_model = regr.fit(X, target)
    feature_names = X.columns
    feature_importances = best_model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, feature_importances))
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_feature_importance = OrderedDict(sorted_feature_importance)
    # select feature based on cumulative importance
    cumulative_importance = 0.0
    selected_features = []
    for feature, importance in sorted_feature_importance.items():
        print(f"feaute, {feature},  import, {importance}")
        cumulative_importance += importance
        selected_features.append(feature)
        if cumulative_importance >= 0.95:
            break
    cv_results['selected_features'] = selected_features

    #save to json
    with open(f"{output_path}{output_prefix}_nXcv.json", 'w') as file:
       json.dump(cv_results, file, default=convert_type)
    file.close()
    
    return cv_results



# In[11]:


def to_valid_variable_name(name):
    # Replace special characters with underscores
    name = re.sub(r'\W|^(?=\d)', '_', name)
    # Reduce multiple consecutive underscores to one
    name = re.sub(r'_{2,}', '_', name)
    # Truncate length if necessary
    max_length = 30
    if len(name) > max_length:
        name = name[:max_length]
    # Ensure it doesn't start with a number
    if name[0].isdigit():
        name = "_" + name
    return name


# In[12]:


tr_mut = pd.read_csv("/home/t44p/PW_rawdata/tr_gc_mutual/tr_mut.csv", sep=",")
gcms_mut = pd.read_csv("/home/t44p/PW_rawdata/tr_gc_mutual/gcms_mut.csv", sep=",")
lcms_mut = pd.read_csv("/home/t44p/PW_rawdata/tr_gc_mutual/lcms_mut.csv", sep=",")

X = pd.read_csv("/home/t44p/PW_rawdata/tr_gc_mutual/tr_mut_transposed.csv", sep=",")
#



# 
# >explore a better grid on the cluster 

# In[20]:


kfold_cv = KFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    'regressor__n_estimators': np.array(np.arange(500, 1501, 1000)),
    'regressor__max_features': np.round(np.exp2(np.array(np.arange(7.2, 15.3, 3)))).astype(int),
    'regressor__bootstrap': [False, True]
}   
rfr_fullkfold_scores, rfr_fullkfold_model, rfr_best_param = gridcv(
    X.iloc[:,:100], 
    gcms_mut.iloc[59,1:],
    RandomForestRegressor(n_jobs=2),
    param_grid,
    scorer='r2', 
    cv_meth=kfold_cv,
    cv_n_jobs=2
)
#for key, value in rfr_fullkfold_scores.items():
#    print(f"{key} >>>>\n {value}\n\n")


# In[42]:


print(f"RFR START >>> {to_valid_variable_name(str(gcms_mut.iloc[59,0]))}\n\n")
tenX = [42, 43, 44]#, 45, 46, 47, 48, 49, 50, 51, 52]
out = './py/10xKfold/test_rfr/'
param_grid = {
    'regressor__n_estimators': np.array(np.arange(10, 15, 1)),
    'regressor__max_features': np.array(np.arange(7, 10, 1)),
    'regressor__bootstrap': [False, True]
}   
sucrose_10xKfold = nX_cross_validation(X.iloc[:,:100], gcms_mut.iloc[59,1:], param_grid, 'r2', to_valid_variable_name(str(gcms_mut.iloc[59,0])), tenX, output_path=out, cv_n_jobs=2, regr_n_job=2)


