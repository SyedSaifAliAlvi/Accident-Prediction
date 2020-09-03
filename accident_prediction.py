from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as pyplot
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV
import warnings
warnings.filterwarnings('ignore')

def load_data():
  train_data = pd.read_csv('Train_data.csv')
  test_data = pd.read_csv('Test_Data.csv')
  return train_data,test_data
#train_data,_=load_data()
#train_data

def drop_redundant(df):
  df=df.drop(['gender','Unnamed: 0'],1)
  #df=df.drop(['gender','code','Unnamed: 0','clientType','registrationMode','planName','country'],1)
  return df

def reshape_data(df):
  df['country'][df['country']=='SINGAPORE']='Singapore'
  df['country'][df['country']!='Singapore']='NonSingapore'
  df['code'][df['code']=='C2B']='C2B'
  df['code'][df['code']!='C2B']='NonC2B'
  return df

def encode_col(df):
  le = LabelEncoder() 
  df['planName']= le.fit_transform(df['planName'])
  df = pd.get_dummies(df)
  return df

def data_sampling(df):
  df = df.sample(frac=1)
  accident_df = df.loc[df['accident'] == 1]
  non_accident_df = df.loc[df['accident'] == 0][:880]#880
  normal_distributed_df = pd.concat([accident_df, non_accident_df])
  new_df = normal_distributed_df.sample(frac=1, random_state=42)
  return new_df

def prepare_X_y(new_df):
  X = new_df.drop('accident', axis=1)
  y = new_df['accident']
  return X,y

def gridcv(X,label_encoded_y):
  model = RandomForestClassifier()
  n_estimators = [70,75,78,88,80,95,100, 200, 300, 400, 500]
  #learning_rate = [0.0001, 0.001, 0.01, 0.1]
  max_features = ['auto', 'sqrt', 'log2']
  min_samples_leaf = [3,4,5,6,7]
  param_grid = dict(n_estimators=n_estimators,max_features=max_features,min_samples_leaf=min_samples_leaf)
  kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
  grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
  grid_result = grid_search.fit(X, label_encoded_y)
  # summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

def feature_selection_model(X,y):
  model = RandomForestClassifier(n_estimators=88,oob_score=False,n_jobs=-1,random_state=11,max_features="sqrt",min_samples_leaf=4)
  model.fit(X, y)
  importance = model.feature_importances_
  for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
  pyplot.bar([x for x in range(len(importance))], importance)
  pyplot.show()

def more_redundant_in_X(X,i):
  col=X.columns
  X=X.drop(col[i],axis=1)
  return X

def model_selection(X,y):
  rfm = RandomForestClassifier(n_estimators=88,oob_score=False,n_jobs=-1,random_state=11,max_features="sqrt",min_samples_leaf=6)
  #rfm = GradientBoostingClassifier(random_state=134,learning_rate=0.05,n_estimators=84)
  #rfm = AdaBoostClassifier(n_estimators=91, random_state=0,learning_rate=0.01)
  #rfm =  SVC(gamma='auto',probability=True)
  #rfm = xgb.XGBClassifier()
  #scores = cross_val_score(rfm, X, y)
  #print("Cross Val score of the model={0}%".format(scores.mean()*100))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  rfm = rfm.fit(X_train,y_train)
  y_pred = rfm.predict(X_test)
  accuracy = metrics.accuracy_score(y_test,y_pred)
  print("Accuracy of the model={0}%".format(accuracy*100))
  predictions = rfm.predict_proba(X_test)
  y_score=[]
  for i in predictions:
    #print(i[1])
    y_score.append(i[1])
  
  calibrated = CalibratedClassifierCV(rfm, method='sigmoid', cv=5)
  calibrated.fit(X_train, y_train)
  # predict probabilities
  probs = calibrated.predict_proba(X_test)[:, 1]
  # reliability diagram
  fop, mpv = calibration_curve(y_test, probs, n_bins=10, normalize=True)
  # plot perfectly calibrated
  pyplot.plot([0, 1], [0, 1], linestyle='--')
  # plot calibrated reliability
  pyplot.plot(mpv, fop, marker='.')
  pyplot.show()
  
  auc=roc_auc_score(np.array(y_test),np.array(y_score))
  brier_score = brier_score_loss(np.array(y_test),np.array(y_score))
  avg_logloss = log_loss(np.array(y_test),np.array(y_score))
  print('P(class0=1): Log Loss=%.3f' % (avg_logloss))
  print("AUC score of the model={0}".format(auc*100))
  print("Brier score of the model={0}".format(brier_score))
  rec = recall_score(y_test,y_pred) 
  print("The recall is {}".format(rec)) 
    
  f1 = f1_score(y_test,y_pred) 
  print("The F1-Score is {}".format(f1)) 
    
  MCC = matthews_corrcoef(y_test,y_pred) 
  print("The Matthews correlation coefficient is{}".format(MCC))
  print("_________________________________________________")

  return rfm,X,y,auc

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = metrics.accuracy_score(test_labels,predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

def write_a_file(result,i):
  k = "Onslaught "+str(i)+' output_value.txt'
  with open(k, 'w') as f:
    for item in result:
        f.write("%s\n" % item)

def out_mode(rfm,dk):
  output_value = rfm.predict_proba(dk)
  result=[]
  for i in output_value:
    result.append(round(i[1],3))
  return result

def pipeLine(i):
  train_data,test_data=load_data()
  df = drop_redundant(train_data)
  df = reshape_data(df)
  df = encode_col(df)
  df = data_sampling(df) 
  #sm = SMOTE(random_state = 7) 
  X,y=prepare_X_y(df)
  #X,y = sm.fit_sample(X, y.ravel())
  #gridcv(X,y)
  #print(X.columns)
  #feature_selection_model(X,y)
  model_selected,X,y,auc_sr=model_selection(X,y)
  #hyperCv(X,y)
  dk = drop_redundant(test_data)
  dk = reshape_data(dk)
  dk = encode_col(dk)
  res = out_mode(model_selected,dk)
  write_a_file(result=res,i=i)
  return auc_sr

for i in range(1):
  scr = pipeLine(i)
  print(".............................................")

