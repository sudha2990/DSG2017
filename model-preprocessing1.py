import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


def get_data():
  #loading processed data
  data = pd.read_pickle("music.pickle")
  test = pd.read_pickle("test.pickle")
  #Target Generation
  target=data['is_listened']
  del data['is_listened']
  test2=test.iloc[:,1:]
  data=pd.concat([data,test2],axis=0)
  data=data.fillna(0)

  #dummies creation
  dummie1 = pd.get_dummies(data['context_type'], prefix='context_type', prefix_sep='_')
  dummie2 = pd.get_dummies(data['platform_name'], prefix='platform_name', prefix_sep='_')
  dummie3 = pd.get_dummies(data['platform_family'], prefix='platform_family', prefix_sep='_')
  dummie4 = pd.get_dummies(data['listen_type'], prefix='listen_type', prefix_sep='_')
  dummie5 = pd.get_dummies(data['user_gender'], prefix='user_gender', prefix_sep='_')
  dummie6 = pd.get_dummies(data['genre_id'], prefix='genre_id', prefix_sep='_')
  dummie7 = pd.get_dummies(data['album_id'], prefix='album_id', prefix_sep='_')
  dummie8 = pd.get_dummies(data['release_date'], prefix='album_id', prefix_sep='_')
  dummie9 = pd.get_dummies(data['usergencluster'], prefix='usergencluster', prefix_sep='_')
  dummie10 = pd.get_dummies(data['artist_id'], prefix='artist_id', prefix_sep='_')
  dummie11 = pd.get_dummies(data['usermedcluster'], prefix='usermedcluster', prefix_sep='_')
  dummie12= pd.get_dummies(data['useralbcluster'], prefix='useralbcluster', prefix_sep='_')
  dummie13= pd.get_dummies(data['userartcluster'], prefix='userartcluster', prefix_sep='_')
  dummie14= pd.get_dummies(data['userdatecluster'], prefix='userdatecluster', prefix_sep='_')

  data=pd.concat([dummie1,dummie2,dummie3,dummie4,dummie5,dummie6,dummie7,dummie8,dummie9,dummie10,dummie11,dummie12,dummie13,dummie14],axis=1)

  return data,target
  
  
def get_model(n_cols):
# Model
  model = Sequential()
  model.add(Dense(98 , activation = 'relu' , input_shape = (n_cols,)))
  model.add(Dense(98 , activation = 'relu'))
  model.add(Dense(56 , activation = 'relu'))
  model.add(Dense(56 , activation = 'relu'))
  model.add(Dense(56 , activation = 'relu'))
  model.add(Dense(42 , activation = 'relu'))
  model.add(Dense(14 , activation = 'relu'))
  model.add(Dense(2 , activation = 'softmax'))
  model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
  
  return model
  
def prediction(test_data):
  # test prediction
  preds = model.predict_proba(test_data, verbose=0)[:, 1]
  submission = pd.DataFrame(preds, columns=['is_listened'])
  submission.to_csv('model1.csv')


data,target = get_data()
# converting to matrix 
predictors = data.as_matrix()# assign feature dataframe to predictors removing the target column
target1 = to_categorical(target)# assign target dataframe to y
n_cols = predictors.shape[1]
predictors.shape

# train model
model = get_model(n_cols)
early_stopping_monitor = EarlyStopping(patience = 2)
model.fit(predictors[0:7558834,:] , target1 , validation_split = 0.20 , epochs = 2 , callbacks = [early_stopping_monitor])
prediction(predictors[7558834:, :])

