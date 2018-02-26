import numpy as np
import pandas as pd
import datetime
from datetime import date
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier


data=pd.read_csv('train.csv')  ##preprocessed data with age of song(diff) and user hour of listen of song features
test=pd.read_csv("test.csv")##preprocessed test data with age of song (diff)and user hour of listen of song features

matrix = data.pivot_table(index=['user_id'], columns=['genre_id'], values='is_listened')
from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=70)
matrix=matrix.fillna(0)
x_cols = matrix.columns[1:]
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["usergencluster"] = data["user_id"].map(mat2)
test["usergencluster"]=test["user_id"].map(mat2)

data['album_id_1']=data['album_id']
test['album_id_1']=test['album_id']
mmo3=pd.DataFrame(data['album_id_1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.album_id_1.values)],axis=1)
mmo31=list(mmo3.iloc[1500:,0])
data['album_id_1'].loc[data['album_id_1'].isin(mmo31)]=-1
matrix = data.pivot_table(index=['user_id'], columns=['album_id_1'], values='is_listened')
cluster = KMeans(n_clusters=100)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["useralbcluster"] = data["user_id"].map(mat2)
test["useralbcluster"]=test["user_id"].map(mat2)

data['media_id_1']=data['media_id']
test['media_id_1']=test['media_id']
mmo3=pd.DataFrame(data['media_id_1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.media_id_1.values)],axis=1)
mmo31=list(mmo3.iloc[3000:,0])
data['media_id_1'].loc[data['media_id_1'].isin(mmo31)]=-1
matrix = data.pivot_table(index=['user_id'], columns=['media_id_1'], values='is_listened')
cluster = KMeans(n_clusters=100)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["usermedcluster"] = data["user_id"].map(mat2)
test["usermedcluster"]=test["user_id"].map(mat2)

data['artist_id_1']=data['artist_id']
test['artist_id_1']=test['artist_id']
mmo3=pd.DataFrame(data['artist_id_1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.artist_id_1.values)],axis=1)
mmo31=list(mmo3.iloc[1700:,0])
data['artist_id_1'].loc[data['artist_id_1'].isin(mmo31)]=-1
matrix = data.pivot_table(index=['user_id'], columns=['artist_id_1'], values='is_listened')
cluster = KMeans(n_clusters=80)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["userartcluster"] = data["user_id"].map(mat2)
test["userartcluster"]=test["user_id"].map(mat2)

matrix=0
data['release_date_1']=data['release_date']
test['release_date_1']=test['release_date']
mmo3=pd.DataFrame(data['release_date_1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.release_date_1.values)],axis=1)
mmo31=list(mmo3.iloc[1000:,0])
data['release_date_1'].loc[data['release_date_1'].isin(mmo31)]=-1
matrix = data.pivot_table(index=['user_id'], columns=['release_date_1'], values='is_listened')
cluster = KMeans(n_clusters=10)
matrix=matrix.fillna(0)
#the below line is server specific and is used to predict cluster of users and then we have mapped users on clusters
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
#matrix1=pd.concat([matrix.iloc[:,0],matrix['cluster']],axis=1)
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["userdatecluster"] = data["user_id"].map(mat2)
test["userdatecluster"]=test["user_id"].map(mat2)

matrix=0
data['time1_1']=data['time1']
test['time1_1']=test['time1']
mmo3=pd.DataFrame(data['time1_1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.time1_1.values)],axis=1)
mmo31=list(mmo3.iloc[:,0])
matrix = data.pivot_table(index=['user_id'], columns=['time1_1'], values='is_listened')
cluster = KMeans(n_clusters=4)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["usertimecluster"] = data["user_id"].map(mat2)
test["usertimecluster"]=test["user_id"].map(mat2)

matrix=0
data['user_age1']=data['user_age']
test['user_age1']=test['user_age']
mmo3=pd.DataFrame(data['user_age1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.user_age1.values)],axis=1)
mmo31=list(mmo3.iloc[1000:,0])
data['user_age1'].loc[data['user_age1'].isin(mmo31)]=-1
matrix = data.pivot_table(index=['user_id'], columns=['user_age1'], values='is_listened')
cluster = KMeans(n_clusters=4)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["useragecluster"] = data["user_id"].map(mat2)
test["useragecluster"]=test["user_id"].map(mat2)

matrix=0
data['context_type1']=data['context_type']
test['context_type1']=test['context_type']
mmo3=pd.DataFrame(data['context_type1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.context_type1.values)],axis=1)
mmo31=list(mmo3.iloc[1000:,0])
data['context_type1'].loc[data['context_type1'].isin(mmo31)]=-1
matrix = data.pivot_table(index=['user_id'], columns=['context_type1'], values='is_listened')
cluster = KMeans(n_clusters=10)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["usercontcluster"] = data["user_id"].map(mat2)
test["usercontcluster"]=test["user_id"].map(mat2)

matrix=0
data['platform_name1']=data['platform_name']
test['platform_name1']=test['platform_name']
mmo3=pd.DataFrame(data['platform_name1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.platform_name1.values)],axis=1)
mmo31=list(mmo3.iloc[1000:,0])
data['platform_name1'].loc[data['platform_name1'].isin(mmo31)]=-1
matrix = data.pivot_table(index=['user_id'], columns=['platform_name1'], values='is_listened')
cluster = KMeans(n_clusters=10)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["userplatcluster"] = data["user_id"].map(mat2)
test["userplatcluster"]=test["user_id"].map(mat2)

matrix=0
matrix = data.pivot_table(index=['user_id'], columns=['user_gender'], values='is_listened')
cluster = KMeans(n_clusters=6)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["usergendcluster"] = data["user_id"].map(mat2)
test["usergendcluster"]=test["user_id"].map(mat2)

matrix=0
matrix = data.pivot_table(index=['genre_id'], columns=['user_age'], values='is_listened')
cluster = KMeans(n_clusters=5)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[0:]])
opp=pd.DataFrame(matrix.index.values)
opp.reset_index(drop=True)
matrix1=pd.concat([opp,(pd.DataFrame(matrix['cluster'])).reset_index(drop=True)],axis=1,ignore_index=True).reset_index(drop=True)
matrix1.columns=['genre_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.genre_id.values).to_dict()
data["genreagecluster"] = data["genre_id"].map(mat2)
test["genreagecluster"]=test["genre_id"].map(mat2)

matrix = data.pivot_table(index=['genre_id'], columns=['context_type1'], values='is_listened')
cluster = KMeans(n_clusters=10)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[0:]])
opp=pd.DataFrame(matrix.index.values)
opp.reset_index(drop=True)
matrix1=pd.concat([opp,(pd.DataFrame(matrix['cluster'])).reset_index(drop=True)],axis=1,ignore_index=True).reset_index(drop=True)
matrix1.columns=['genre_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.genre_id.values).to_dict()
data["genrecontcluster"] = data["genre_id"].map(mat2)
test["genrecontcluster"]=test["genre_id"].map(mat2)

matrix=0
data['listen_type1']=data['listen_type']
test['listen_type1']=test['listen_type']
matrix = data.pivot_table(index=['genre_id'], columns=['listen_type1'], values='is_listened')
cluster = KMeans(n_clusters=5)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[0:]])
opp=pd.DataFrame(matrix.index.values)
opp.reset_index(drop=True)
matrix1=pd.concat([opp,(pd.DataFrame(matrix['cluster'])).reset_index(drop=True)],axis=1,ignore_index=True).reset_index(drop=True)
matrix1.columns=['genre_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.genre_id.values).to_dict()
data["genrelistcluster"] = data["genre_id"].map(mat2)
test["genrelistcluster"]=test["genre_id"].map(mat2)

matrix = data.pivot_table(index=['genre_id'], columns=['user_gender'], values='is_listened')
cluster = KMeans(n_clusters=5)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[0:]])
opp=pd.DataFrame(matrix.index.values)
opp.reset_index(drop=True)
matrix1=pd.concat([opp,(pd.DataFrame(matrix['cluster'])).reset_index(drop=True)],axis=1,ignore_index=True).reset_index(drop=True)
matrix1.columns=['genre_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.genre_id.values).to_dict()
data["genregendcluster"] = data["genre_id"].map(mat2)
test["genregendcluster"]=test["genre_id"].map(mat2)


matrix = data.pivot_table(index=['user_id'], columns=['listen_type1'], values='is_listened')
cluster = KMeans(n_clusters=10)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["userlisttypecluster"] = data["user_id"].map(mat2)
test["userlisttypecluster"]=test["user_id"].map(mat2)

matrix = data.pivot_table(index=['user_id'], columns=['platform_family'], values='is_listened')
cluster = KMeans(n_clusters=10)
matrix=matrix.fillna(0)
x_cols = matrix.columns[1:]
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["userplafamcluster"] = data["user_id"].map(mat2)
test["userplafamcluster"]=test["user_id"].map(mat2)

matrix = data.pivot_table(index=['artist_id'], columns=['user_gender'], values='is_listened')
cluster = KMeans(n_clusters=6)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
opp=pd.DataFrame(matrix.index.values)
opp.reset_index(drop=True)
matrix1=pd.concat([opp,(pd.DataFrame(matrix['cluster'])).reset_index(drop=True)],axis=1,ignore_index=True).reset_index(drop=True)
matrix1.columns=['artist_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.artist_id.values).to_dict()
data["artistgendcluster"] = data["artist_id"].map(mat2)
test["artistgendcluster"]=test["artist_id"].map(mat2)

matrix = data.pivot_table(index=['artist_id'], columns=['listen_type1'], values='is_listened')
cluster = KMeans(n_clusters=10)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
opp=pd.DataFrame(matrix.index.values)
opp.reset_index(drop=True)
matrix1=pd.concat([opp,(pd.DataFrame(matrix['cluster'])).reset_index(drop=True)],axis=1,ignore_index=True).reset_index(drop=True)
matrix1.columns=['artist_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.artist_id.values).to_dict()
data["artistlisttypecluster"] = data["artist_id"].map(mat2)
test["artistlisttypecluster"]=test["artist_id"].map(mat2)



mmo3=pd.DataFrame(data['media_id'].value_counts())
mmo98=pd.DataFrame(test['media_id'].value_counts())
unmatched = set(list(mmo98.index.values))-set(list(mmo3.index.values))
test['media_id'].loc[test['media_id'].isin(list(unmatched)[65:])]=-2
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.media_id.values)],axis=1)
df=pd.Series(mmo3.iloc[:,1].values,index=mmo3.iloc[:,0].values).to_dict()
data["media_id_freq"] = data["media_id"].map(df)
test["media_id_freq"] = test["media_id"].map(df)
mmo31=list(mmo3.iloc[100:,0])
data['media_id'].loc[data['media_id'].isin(mmo31)]=-1
test['media_id'].loc[test['media_id'].isin(mmo31)]=-1


mmo3=pd.DataFrame(data['album_id'].value_counts())
mmo98=pd.DataFrame(test['album_id'].value_counts())
unmatched = set(list(mmo98.index.values))-set(list(mmo3.index.values))
test['album_id'].loc[test['album_id'].isin(list(unmatched)[65:])]=-2
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.album_id.values)],axis=1)
df=pd.Series(mmo3.iloc[:,1].values,index=mmo3.iloc[:,0].values).to_dict()
data["album_id_freq"] = data["album_id"].map(df)
test["album_id_freq"] = test["album_id"].map(df)
mmo31=list(mmo3.iloc[100:,0])
data['album_id'].loc[data['album_id'].isin(mmo31)]=-1
test['album_id'].loc[test['album_id'].isin(mmo31)]=-1

mmo4=pd.DataFrame(data['artist_id'].value_counts())
mmo98=pd.DataFrame(test['artist_id'].value_counts())
unmatched = set(list(mmo98.index.values))-set(list(mmo4.index.values))
test['artist_id'].loc[test['artist_id'].isin(list(unmatched)[40:])]=-2
mmo4=pd.concat([pd.DataFrame(mmo4.index.values),pd.DataFrame(mmo4.artist_id.values)],axis=1)
df=pd.Series(mmo4.iloc[:,1].values,index=mmo4.iloc[:,0].values).to_dict()
data["artist_id_freq"] = data["artist_id"].map(df)
test["artist_id_freq"] = test["artist_id"].map(df)
mmo41=list(mmo4.iloc[100:,0])
data['artist_id'].loc[data['artist_id'].isin(mmo41)]=-1
test['artist_id'].loc[test['artist_id'].isin(mmo41)]=-1

      
mmo5=pd.DataFrame(data['genre_id'].value_counts())
mmo98=pd.DataFrame(test['genre_id'].value_counts())
unmatched = set(list(mmo98.index.values))-set(list(mmo5.index.values))
test['genre_id'].loc[test['genre_id'].isin(unmatched)]=-2
mmo5=pd.concat([pd.DataFrame(mmo5.index.values),pd.DataFrame(mmo5.genre_id.values)],axis=1)
df=pd.Series(mmo5.iloc[:,1].values,index=mmo5.iloc[:,0].values).to_dict()
data["genre_id_freq"] = data["genre_id"].map(df)
test["genre_id_freq"] = test["genre_id"].map(df)
mmo51=list(mmo5.iloc[50:,0])
data['genre_id'].loc[data['genre_id'].isin(mmo51)]=-1
test['genre_id'].loc[test['genre_id'].isin(mmo51)]=-1

mmo6=pd.DataFrame(data['context_type'].value_counts())
mmo6=pd.concat([pd.DataFrame(mmo6.index.values),pd.DataFrame(mmo6.context_type.values)],axis=1)
mmo61=list(mmo6.iloc[25:,0])
data['context_type'].loc[data['context_type'].isin(mmo61)]=-1
test['context_type'].loc[test['context_type'].isin(mmo61)]=-1


mmo7=pd.DataFrame(data['release_date'].value_counts())
mmo98=pd.DataFrame(test['release_date'].value_counts())
#unmatched = set(list(mmo98.index.values))-set(list(mmo7.index.values))
#test['release_date'].loc[test['release_date'].isin(unmatched)]=-1
mmo7=pd.concat([pd.DataFrame(mmo7.index.values),pd.DataFrame(mmo7.release_date.values)],axis=1)
mmo71=list(mmo7.iloc[25:,0])
data['release_date'].loc[data['release_date'].isin(mmo71)]=-1
test['release_date'].loc[test['release_date'].isin(mmo71)]=-1
matrix=0

data.to_pickle("music.pickle")
test.to_pickle("test.pickle")