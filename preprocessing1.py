
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


data=pd.read_csv('train.csv')
test=pd.read_csv("test.csv")

df43 = pd.DataFrame(data, columns = ['genre_id', 'is_listened'])
df45 = pd.DataFrame(data, columns = ['media_id', 'is_listened'])
df46=  pd.DataFrame(data, columns = ['album_id', 'is_listened'])
df47=  pd.DataFrame(data, columns = ['artist_id', 'is_listened'])
df51=  pd.DataFrame(data, columns = ['user_id', 'is_listened'])
df53=  pd.DataFrame(data, columns = ['release_date', 'is_listened'])


#TO create fractional features. First groupby ids and then calculate mean of is listened for every level of id
#and then map it on data using dictionary creation
count1=df43.groupby(['genre_id'],as_index=False).agg({'is_listened':'mean'})
df44=pd.Series(count1.is_listened.values,index=count1.genre_id).to_dict()
data["B"] = data["genre_id"].map(df44)
test["B"] = test["genre_id"].map(df44)


count2=df45.groupby(['media_id'],as_index=False).agg({'is_listened':'mean'})
df48=pd.Series(count2.is_listened.values,index=count2.media_id).to_dict()
data["C"] = data["media_id"].map(df48)
test["C"] = test["media_id"].map(df48)

count3=df46.groupby(['album_id'],as_index=False).agg({'is_listened':'mean'})
df49=pd.Series(count3.is_listened.values,index=count3.album_id).to_dict()
data["D"] = data["album_id"].map(df49)
test["D"] = test["album_id"].map(df49)

count4=df47.groupby(['artist_id'],as_index=False).agg({'is_listened':'mean'})
df50=pd.Series(count4.is_listened.values,index=count4.artist_id).to_dict()
data["E"] = data["artist_id"].map(df50)
test["E"] = test["artist_id"].map(df50)

count5=df51.groupby(['user_id'],as_index=False).agg({'is_listened':'mean'})
df52=pd.Series(count5.is_listened.values,index=count5.user_id).to_dict()
data["F"] = data["user_id"].map(df52)
test["F"] = test["user_id"].map(df52)

count6=df53.groupby(['release_date'],as_index=False).agg({'is_listened':'mean'})
df54=pd.Series(count6.is_listened.values,index=count6.release_date).to_dict()
data["G"] = data["release_date"].map(df54)
test["G"] = test["release_date"].map(df54)

#pivot table for clustering of users based on listening preferences wrt various attributes

matrix = data.pivot_table(index=['user_id'], columns=['genre_id'], values='is_listened')
cluster = KMeans(n_clusters=50)
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
cluster = KMeans(n_clusters=50)
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
cluster = KMeans(n_clusters=50)
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
cluster = KMeans(n_clusters=50)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["userartcluster"] = data["user_id"].map(mat2)
test["userartcluster"]=test["user_id"].map(mat2)

data['release_date_1']=data['release_date']
test['release_date_1']=test['release_date']
mmo3=pd.DataFrame(data['release_date_1'].value_counts())
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.release_date_1.values)],axis=1)
mmo31=list(mmo3.iloc[1000:,0])
data['release_date_1'].loc[data['release_date_1'].isin(mmo31)]=-1
matrix = data.pivot_table(index=['user_id'], columns=['release_date_1'], values='is_listened')
cluster = KMeans(n_clusters=10)
matrix=matrix.fillna(0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix1=pd.concat([pd.DataFrame(matrix.index.values),matrix['cluster']],axis=1)
matrix1.columns=['user_id','cluster']
mat2=pd.Series(matrix1.cluster.values,index=matrix1.user_id.values).to_dict()
data["userdatecluster"] = data["user_id"].map(mat2)
test["userdatecluster"]=test["user_id"].map(mat2)



#checking if all levels of ids and other of test occur in train. Assigning some of them as extra level
mmo3=pd.DataFrame(data['album_id'].value_counts())
mmo98=pd.DataFrame(test['album_id'].value_counts())
unmatched = set(list(mmo98.index.values))-set(list(mmo3.index.values))
test['album_id'].loc[test['album_id'].isin(unmatched)]=-1
mmo3=pd.concat([pd.DataFrame(mmo3.index.values),pd.DataFrame(mmo3.album_id.values)],axis=1)
df=pd.Series(mmo3.iloc[:,1].values,index=mmo3.iloc[:,0].values).to_dict()
data["album_id_freq"] = data["album_id"].map(df)
test["album_id_freq"] = test["album_id"].map(df)
mmo31=list(mmo3.iloc[40:,0])
data['album_id'].loc[data['album_id'].isin(mmo31)]=-1
test['album_id'].loc[test['album_id'].isin(mmo31)]=-1

mmo4=pd.DataFrame(data['artist_id'].value_counts())
mmo98=pd.DataFrame(test['artist_id'].value_counts())
unmatched = set(list(mmo98.index.values))-set(list(mmo4.index.values))
test['artist_id'].loc[test['artist_id'].isin(unmatched)]=-1
mmo4=pd.concat([pd.DataFrame(mmo4.index.values),pd.DataFrame(mmo4.artist_id.values)],axis=1)
df=pd.Series(mmo4.iloc[:,1].values,index=mmo4.iloc[:,0].values).to_dict()
data["artist_id_freq"] = data["artist_id"].map(df)
test["artist_id_freq"] = test["artist_id"].map(df)
mmo41=list(mmo4.iloc[40:,0])
data['artist_id'].loc[data['artist_id'].isin(mmo41)]=-1
test['artist_id'].loc[test['artist_id'].isin(mmo41)]=-1

      
mmo5=pd.DataFrame(data['genre_id'].value_counts())
mmo98=pd.DataFrame(test['genre_id'].value_counts())
unmatched = set(list(mmo98.index.values))-set(list(mmo5.index.values))
test['genre_id'].loc[test['genre_id'].isin(unmatched)]=-1
mmo5=pd.concat([pd.DataFrame(mmo5.index.values),pd.DataFrame(mmo5.genre_id.values)],axis=1)
df=pd.Series(mmo5.iloc[:,1].values,index=mmo5.iloc[:,0].values).to_dict()
data["genre_id_freq"] = data["genre_id"].map(df)
test["genre_id_freq"] = test["genre_id"].map(df)
mmo51=list(mmo5.iloc[30:,0])
data['genre_id'].loc[data['genre_id'].isin(mmo51)]=-1
test['genre_id'].loc[test['genre_id'].isin(mmo51)]=-1

mmo6=pd.DataFrame(data['context_type'].value_counts())
mmo6=pd.concat([pd.DataFrame(mmo6.index.values),pd.DataFrame(mmo6.context_type.values)],axis=1)
mmo61=list(mmo6.iloc[24:,0])
data['context_type'].loc[data['context_type'].isin(mmo61)]=-1
test['context_type'].loc[test['context_type'].isin(mmo61)]=-1

data['release_date1']=data.release_date
test['release_date1']=test.release_date
mmo7=pd.DataFrame(data['release_date'].value_counts())
mmo98=pd.DataFrame(test['release_date'].value_counts())
unmatched = set(list(mmo98.index.values))-set(list(mmo7.index.values))
test['release_date'].loc[test['release_date'].isin(unmatched)]=-1
mmo7=pd.concat([pd.DataFrame(mmo7.index.values),pd.DataFrame(mmo7.release_date.values)],axis=1)
mmo71=list(mmo7.iloc[30:,0])
data['release_date'].loc[data['release_date'].isin(mmo71)]=-1
test['release_date'].loc[test['release_date'].isin(mmo71)]=-1

data.to_pickle("music.pickle")
test.to_pickle("test.pickle")