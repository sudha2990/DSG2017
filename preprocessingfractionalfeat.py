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


data=pd.read_csv('train.csv')  ##preprocessed train data with age of song(diff) and user hour of listen of song features
test=pd.read_csv("test.csv")  ##preprocessed test data with age of song(diff) and user hour of listen of song features

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


target=data['is_listened']
del data['is_listened']
test2=test.iloc[:,1:] 

fulldata=pd.concat([data,test2],axis=0)


fulldata['media_duration'].loc[(fulldata['media_duration']>=500) ]=0
fulldata['media_duration'].loc[(fulldata['media_duration']>=250 )&( fulldata['media_duration']<500) ]=-3
fulldata['media_duration'].loc[(fulldata['media_duration']>=150 )&( fulldata['media_duration']<250) ]=-4
fulldata['media_duration'].loc[(fulldata['media_duration']>=30 )&( fulldata['media_duration']<150) ]=-5
fulldata['media_duration'].loc[(fulldata['media_duration'] <30) &( fulldata['media_duration']>0) ]=-6

#diff
fulldata['diff'].loc[(fulldata['diff']>=6000) ]=0
fulldata['diff'].loc[(fulldata['diff']>=2000 )&( fulldata['diff']<6000) ]=-5
fulldata['diff'].loc[(fulldata['diff']>=1000 )&( fulldata['diff']<2000) ]=-6
fulldata['diff'].loc[(fulldata['diff']>=200 )&( fulldata['diff']<1000) ]=-7
fulldata['diff'].loc[(fulldata['diff']>=100 )&( fulldata['diff']<200) ]=-8
fulldata['diff'].loc[(fulldata['diff']>=50 )&( fulldata['diff']<100) ]=-10
fulldata['diff'].loc[(fulldata['diff']>=25 )&( fulldata['diff']<50) ]=-11
fulldata['diff'].loc[(fulldata['diff']>=10 )&( fulldata['diff']<25) ]=-12
fulldata['diff'].loc[(fulldata['diff']<10 ) &( fulldata['diff']>0)]=-13

#B
fulldata['B'].loc[(fulldata['B']>=0 )&( fulldata['B']<0.5) ]=-1
fulldata['B'].loc[(fulldata['B']>=0.5 )&( fulldata['B']<0.7) ]=-2
fulldata['B'].loc[(fulldata['B']>=0.7 )&( fulldata['B']<=1) ]=-7


#C
fulldata['C'].loc[(fulldata['C']>=0 )&( fulldata['C']<0.5) ]=-1
fulldata['C'].loc[(fulldata['C']>=0.5 )&( fulldata['C']<0.9) ]=-2
fulldata['C'].loc[(fulldata['C']>=0.9 )&( fulldata['C']<=1) ]=-8

#D
fulldata['D'].loc[(fulldata['D']>=0 )&( fulldata['D']<0.5) ]=-1
fulldata['D'].loc[(fulldata['D']>=0.5 )&( fulldata['D']<=0.95) ]=-2
fulldata['D'].loc[(fulldata['D']>0.95 )&( fulldata['D']<=1) ]=-7

#E
fulldata['E'].loc[(fulldata['E']>0 )&( fulldata['E']<=0.5) ]=-1
fulldata['E'].loc[(fulldata['E']>0.5 )&( fulldata['E']<=0.8) ]=-2
fulldata['E'].loc[(fulldata['E']>0.8 )&( fulldata['E']<=1) ]=-7

#F
fulldata['F'].loc[(fulldata['F']>=0 )&( fulldata['F']<=0.3) ]=-1
fulldata['F'].loc[(fulldata['F']>0.3)&( fulldata['F']<=0.5) ]=-2
fulldata['F'].loc[(fulldata['F']>0.5 )&( fulldata['F']<=0.8) ]=-3
fulldata['F'].loc[(fulldata['F']>0.8 )&( fulldata['F']<=1) ]=-9

#G
fulldata['G'].loc[(fulldata['G']>=0 )&( fulldata['G']<=0.5) ]=-1
fulldata['G'].loc[(fulldata['G']>0.5 )&( fulldata['G']<=0.7) ]=-2
fulldata['G'].loc[(fulldata['G']>0.7 )&( fulldata['G']<=0.8) ]=-3
fulldata['G'].loc[(fulldata['G']>0.8 )&( fulldata['G']<=1) ]=-8


data=pd.concat([fulldata.iloc[:,0:7558834],target],axis=1)
test=fulldata.iloc[:,7558834:]

data.to_pickle("music.pickle")
test.to_pickle("test.pickle")