import pandas as pd
import logging
from sklearn.metrics import(
    f1_score,
    roc_auc_score,
    confusion_matrix,
    auc,
    accuracy_score,
    roc_curve
)
from sklearn.utils import resample,shuffle
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,

)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

import xgboost

from sklearn import (
    preprocessing,
    ensemble,
    tree,
)
logging.basicConfig(filename='info.log')
cc = pd.read_csv('creditcard.csv')
X = cc.drop(columns=['Class'])
y = cc['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,
)

df_train = pd.concat([X_train,y_train],axis=1)
df_test = pd.concat([X_test,y_test],axis=1)


models = [GaussianNB, DecisionTreeClassifier,RandomForestClassifier]
sca = preprocessing.StandardScaler()




from data_sampling import df_sampling

def base_estimate(df: pd.DataFrame,sc,model_list):
    ''' baseline check '''
    df =df_sampling(df,n = 10000, up_sample=True)
    X = df.drop(columns = ['Class'])
    y = df['Class']
    
    for model in model_list :
        p = make_pipeline(sc,model())
        kf =KFold(n_splits=10,shuffle= True, random_state= 42)
        cv = cross_val_score(p,X,y,cv=kf,scoring='f1')
        logging.info(f'{model.__name__:22}   {cv.mean()}')

if __name__ == "__main__":
    
    base_estimate(df_train,sca,models)






