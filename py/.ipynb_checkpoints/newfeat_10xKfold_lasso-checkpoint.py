#!/usr/bin/env python
# coding: utf-8

# # Lasso with reduced feature space
# - WGCNA was used to cluster the transcription data
# - account for multicolinearity

# In[1]:


from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler as zscore # zscore
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso #LRlasso
from joblib import dump, load #to save models in files
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import re
import json
from datetime import datetime
import os


# ### Piplines from the python scripts 

# In[2]:



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


    # access the 'regressor' step from the best pipeline and then its coefficients
    coefficients = best_pipeline.named_steps['regressor'].coef_
    overall_metric['non_zero_coefficients'] = coefficients[coefficients != 0]
    overall_metric['non_zero_coefficients'] = overall_metric['non_zero_coefficients'].tolist()
    overall_metric['non_zero_features'] = list(X.columns[np.where(coefficients != 0)[0]])

    # printing section
    print("best parameter from gridsearch>>\n", out_model.best_params_)
    print(overall_metric['CV'])
    print(overall_metric['scoring_metric'])
    print("correlation Matrix>>\n", overall_metric['correlation_true_pred'])
    print("non_zero_features>>\n",overall_metric['non_zero_features'])
    print("scores for each fold>>\n",overall_metric['fold_scores'])

    return overall_metric, out_model, best_params


# In[3]:



def nX_cross_validation(X, target, param_grid, scorer_estimate, output_prefix, random_states, output_path='./models/10xKfold/', n_splits=3, cv_n_jobs=1):
    if os.path.exists(output_path):
        print(f"The path {output_path} exists.")
    else:
        print(f"The path {output_path} does not exist.")
        raise FileNotFoundError(f"The path {output_path} does not exist.")
    best_fold_mean = float('-inf')
    best_model = []
    #cv_results = {'random_state': [], 'scores': {}, 'mean_scores': [], 'common_features': {}, 'model': {}}
    cv_results = {'random_state': [], 'scores': {}, 'mean_scores': [], 'selected_features': {}, 'best_param': []}
    for ran_state in random_states:
        print(ran_state)
        kfold_cv = KFold(n_splits=n_splits, shuffle=True, random_state=ran_state)
        scores, model, best_param = gridcv(
            X, 
            target,
            Lasso(max_iter=8000),
            param_grid,
            prepy=False,
            scorer=scorer_estimate, 
            cv_meth=kfold_cv,
            cv_n_jobs=cv_n_jobs
        )
        cv_results['random_state'].append(ran_state)
        cv_results['scores'][ran_state] = scores
        cv_results['mean_scores'].append(np.mean(scores['fold_scores']))
        if best_fold_mean == -100:
            best_fold_mean = np.mean(scores['fold_scores'])
            cv_results['best_param'] = best_param, ran_state, np.mean(scores['fold_scores'])
        elif best_fold_mean < np.mean(scores['fold_scores']):
            best_fold_mean = np.mean(scores['fold_scores'])
            cv_results['best_param'] = best_param, ran_state, np.mean(scores['fold_scores'])


        # cv_results['model'][ran_state] = model

    # Determine common features...
    #cv_results['common_features'] = set(cv_results['scores'][42]['non_zero_features'])
    #for r in cv_results['random_state'][1:]:
    #    current_features = set(cv_results['scores'][r]['non_zero_features'])
    #    cv_results['common_features'] = cv_results['common_features'].intersection(current_features)
    #cv_results['common_features'] = list(cv_results['common_features'])
    print(f"best estimator>>\n found in split: {cv_results['best_param'][1]}\n param_grid: {cv_results['best_param'][0]}\n mean fold score {cv_results['best_param'][2]}")    
    best_model = Lasso(alpha=cv_results['best_param'][0]['regressor__alpha'], fit_intercept=cv_results['best_param'][0]['regressor__fit_intercept']).fit(X, target)
    cv_results['selected_features'] = list(X.columns[np.where(best_model.coef_ != 0)[0]])
    #save to json
    with open(f"{output_path}{output_prefix}_nXcv.json", 'w') as file:
       json.dump(cv_results, file)
    file.close()

    return cv_results


# In[4]:


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


# In[5]:


#gcms_mut = pd.read_csv("/home/t44p/PW_rawdata/tr_gc_mutual/gcms_mut.csv", sep=",")
#lcms_mut = pd.read_csv("/home/t44p/PW_rawdata/tr_gc_mutual/lcms_mut.csv", sep=",")
gcms_mut = pd.read_csv("/work/yhesse/PW_rawdata/tr_gc_mutual/gcms_mut.csv", sep=",")
lcms_mut = pd.read_csv("/work/yhesse/PW_rawdata/tr_gc_mutual/lcms_mut.csv", sep=",")


# In[6]:


X = pd.read_csv("/work/yhesse/PW_rawdata/tr_gc_mutual/tr_wgcna_MEs.csv", sep=",", index_col=0)
#X.head()


# In[7]:


gcms_target_dict = {}
for target in gcms_mut['metabolite']:
    o = to_valid_variable_name(target)
    #print(f"{o} == \t {target}")
    gcms_target_dict[o] = str(target)


lcms_target_dict = {}
for target in lcms_mut['metabolite']:
    o = to_valid_variable_name(target)
    #print(f"{o} == \t {target}")
    lcms_target_dict[o] = str(target)
        


# In[47]:


#np.logspace(-4, 1, 20)
##old grid
#param_grid = {
#    'regressor__alpha': np.array(np.arange(0.0125, 0.0625, 0.0025)),
#    'regressor__fit_intercept': [True, False]
#}


# In[8]:


param_grid = {
    'regressor__alpha': np.logspace(-4, 1, 20),
    'regressor__fit_intercept': [True]
}
cpus = 12
tenX = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
print(f"parameter grid >>\n {param_grid}")


# In[10]:


out = '/work/yhesse/jobs/xele_ml/newfeat_lasso/gcms/'

print(f"parameter grid >>\n {param_grid}")

for i, (gcms_target, orig_str) in enumerate(gcms_target_dict.items()):
    now = datetime.now()
    print(f"\n>> START {gcms_target} {now.isoformat()} <<")
    tmp_10xKfold = nX_cross_validation(X.iloc[:,:], gcms_mut.iloc[i,1:], param_grid, 'r2', str(gcms_target), random_states=tenX, output_path=out, cv_n_jobs=cpus)
    print(f"\n>> DONE <<\n\n")

with open(f"{out}gcms_dict_nXcv.json", 'w') as file:
    json.dump(gcms_target_dict, file)


# In[9]:



out = '/work/yhesse/jobs/xele_ml/newfeat_lasso/lcms/'

for i, (lcms_target, orig_str) in enumerate(lcms_target_dict.items()):
    now = datetime.now()
    print(f"\n>> START {lcms_target} {now.isoformat()} <<")
    print(f"{lcms_mut.iloc[i,0]}\t{lcms_target}")
    tmp_10xKfold = nX_cross_validation(X.iloc[:,:], lcms_mut.iloc[i,1:], param_grid, 'r2', str(lcms_target), random_states=tenX, output_path=out, cv_n_jobs=cpus)
    print(f"\n>> DONE <<\n\n")


with open(f"{out}lcms_dict_nXcv.json", 'w') as file:
    json.dump(lcms_target_dict, file)
