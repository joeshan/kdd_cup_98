'''

Joe Shan
Sep 2017

Data and Introduction:
http://www.kdd.org/kdd-cup/view/kdd-cup-1998/Intro

Focus on campaign response

'''

### Initialization

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import TransformerMixin
import pandas_profiling as pprof
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import logit, probit, poisson, ols


def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()


def profiling_by_decile(X, pred_score):
    import matplotlib.pyplot as plt
    X_col = list(X.columns)
    col = ['decile',]
    col.extend(X_col)
    prof = pd.DataFrame(np.zeros((10,len(col))), columns=col)
    
    centile_point = []
    
    for N in range(10,100,10):
        centile_point.append(np.percentile(pred_score, 100-N))
        
    decile_index =[[s >= centile_point[0] for s in pred_score]
                   ,[centile_point[0] > s >= centile_point[1] for s in pred_score]
                   ,[centile_point[1] > s >= centile_point[2] for s in pred_score]
                   ,[centile_point[2] > s >= centile_point[3] for s in pred_score]
                   ,[centile_point[3] > s >= centile_point[4] for s in pred_score]
                   ,[centile_point[4] > s >= centile_point[5] for s in pred_score]
                   ,[centile_point[5] > s >= centile_point[6] for s in pred_score]
                   ,[centile_point[6] > s >= centile_point[7] for s in pred_score]
                   ,[centile_point[7] > s >= centile_point[8] for s in pred_score]
                   ,[s < centile_point[8] for s in pred_score]]
    
    for i in range(10):
        prof['decile'][i] = i+1
        for j in range(1,len(col)):
            prof[col[j]][i] = X[col[j]][decile_index[i]].mean()
    
    prof['decile'] = prof['decile'].astype(int)
    
    for j in range(1,len(col)):
        plt.bar(prof['decile'], prof[col[j]])
        plt.ylabel('Mean of %s' % col[j])
        plt.xlabel('decile')
        plt.title('Profile of %s by decile' % col[j])
        plt.show()
    
    return prof


def performance_by_decile(y, pred_score):
    perf = pd.DataFrame(np.zeros((10,6)),columns=['decile','size','resp', 'resp %','cum. resp', 'cum. resp %'])
    obs_num = len(y)
    resp_num = y.sum()

    centile_point = []
    
    for N in range(10,100,10):
        centile_point.append(np.percentile(pred_score, 100-N))
        
     
    decile_index =[[s >= centile_point[0] for s in pred_score]
                   ,[centile_point[0] > s >= centile_point[1] for s in pred_score]
                   ,[centile_point[1] > s >= centile_point[2] for s in pred_score]
                   ,[centile_point[2] > s >= centile_point[3] for s in pred_score]
                   ,[centile_point[3] > s >= centile_point[4] for s in pred_score]
                   ,[centile_point[4] > s >= centile_point[5] for s in pred_score]
                   ,[centile_point[5] > s >= centile_point[6] for s in pred_score]
                   ,[centile_point[6] > s >= centile_point[7] for s in pred_score]
                   ,[centile_point[7] > s >= centile_point[8] for s in pred_score]
                   ,[s < centile_point[8] for s in pred_score]]
    
    for i in range(10):
        perf['decile'][i] = i+1
        perf['size'][i] = len(y[decile_index[i]])
        perf['resp'][i] = y[decile_index[i]].sum()
        perf['resp %'][i] = round(perf['resp'][i]/resp_num*100,2)
        perf['cum. resp'][i] = perf['resp'][0:i+1].sum()
        perf['cum. resp %'][i] = round(perf['cum. resp'][i]/resp_num*100,2)
        
    perf[['decile','size','resp','cum. resp']] = perf[['decile','size','resp','cum. resp']].astype(int)
    
    return perf
    
    
def vif_cal(input_data):
    import statsmodels.formula.api as sm
    x_vars=input_data
    xvar_names=input_data.columns
    vif_lst=[]
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_lst.append(vif)  
    vif_df = pd.concat([pd.DataFrame(xvar_names,columns=['Variable']),pd.DataFrame(vif_lst,columns=['VIF'])],axis=1)
    return vif_df
        

def auto_countplot(data_df, y_name):
    sns.set_style('whitegrid')
    for var in data_df.columns.drop(y_name):
        sns.countplot(x = y_name , hue = var, data = data_df, palette='RdBu_r')
        plt.show()
        print('\n')
        print(data_df.groupby([var, y_name]).count())
        print('\n')
        

class Missing_Imputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with median of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    

def check_non_num(X_col, X):
    for c in X_col:
        if X[c].dtype == np.dtype('O'):
            print(c,' is still object dtype!')


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    import numpy as np
    from scipy import stats, linalg
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr
