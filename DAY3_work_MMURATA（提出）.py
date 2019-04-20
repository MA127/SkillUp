
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container{width:100% !important;}</style>")) #作業領域を広くする

（題意から）11月からデータを記録。冬にE10からSP98に変更。春にE10に戻している。
# In[2]:


#kaggle課題　car fuel consumption ツールのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression,SGDClassifier
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import classification_report, accuracy_score #予測結果の識別率を算出
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画
# 回帰問題における性能評価に関する関数ラベルを予測
from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix 
import io
import datetime
from datetime import datetime as dt


# In[3]:


df = pd.read_csv("measurements.csv")
df.head()


# In[4]:


df.notnull().sum() 

データがほぼ揃っている変数で回帰することを考える。目的変数がガソリン消費量であるが、ガソリンには2種類（E10、SP98）あり、かつGasのrefillが行われているので、その分（refill litersとrefill gas）も考慮する必要がある。データがほぼ揃っておらず削除しようとする変数の「specials」を確認しておく。（次のとおりであり、「specials」は一旦削除する。）
# In[5]:


df['specials'].value_counts().to_dict()


# In[6]:


#「specials」以外の変数でデータフレーム再作成
df2 = pd.read_csv("measurements.csv")[['distance','consume','speed', 'temp_inside', 'temp_outside','gas_type', 'AC',
                                                      'rain','sun','refill liters','refill gas']]
df2.head()


# In[7]:


df3=df2.dropna(subset=['refill liters'])
df3.head(13)
#以下の結果から「gas_type」と「refill gas」は同種であり、混在がないと分かる。
#またrefill litersとconsumeは水準が1桁違うので、refill litersはconsumeに影響しないと考える。


# In[8]:


df['gas_type'].value_counts().to_dict() #gas_typeの数を確認する


# In[9]:


#consumeとdistanceの関係を確認する
df_S=df.sort_values(by='consume',ascending=False)
plt.figure(figsize=(25,8))
y=df_S["consume"]
x=df_S["distance"]
plt.bar(x,y,color="b")
plt.xlabel("distance")
plt.ylabel("consumel")
plt.show()

全般に消費量と距離はほぼ比例はしているが、距離が短いところでは過剰に消費している。gas_typeを分けてやり直してみる。
# In[10]:


#SP98のデータフレーム
df_SP=df.drop(df.index[df["gas_type"] =='E10'], axis=0) 
df_SP.head(3)


# In[11]:


#SP10のデータフレーム
df_E=df.drop(df.index[df["gas_type"] =='SP98'], axis=0) 
df_E.head(3)


# In[12]:


#SP98のconsumeとdistanceの関係を確認する
df_S_SP=df_SP.sort_values(by='consume',ascending=False)
plt.figure(figsize=(25,6))
y=df_S_SP["consume"]
x=df_S_SP["distance"]
plt.bar(x,y,color="b")
plt.xlabel("distance")
plt.ylabel("consume")
plt.show()


# In[13]:


#E10のconsumeとdistanceの関係を確認する
df_S_E=df_E.sort_values(by='consume',ascending=False)
plt.figure(figsize=(25,6))
y=df_S_E["consume"]
x=df_S_E["distance"]
plt.bar(x,y,color="b")
plt.xlabel("distance")
plt.ylabel("consume")
plt.show()

距離が短いところで、消費量が多いのはGasの種類に依らないことが分かった。次はスピードとの関係を調べる。
# In[14]:


#全体のspeedとconsumeの関係を確認する
df_speed=df2.sort_values(by='speed',ascending=False)
plt.figure(figsize=(25,6))
y=df_speed["consume"]
x=df_speed["speed"]
plt.bar(x,y,color="b")
plt.xlabel("speed")
plt.ylabel("consume")
plt.show()

スピードが遅い方が、進んだ距離に関わらずconsumeが大きい。
# In[15]:


df2.describe()

temp_insideの欠損値を埋める。またdescribe()で扱えなかったdistanceとconsumeの型を扱いやすいよう変換する。
# In[16]:


df2['distance']=df2['distance'].str.replace(",",".").astype(float)
df2['consume']=df2['consume'].str.replace(",",".").astype(float)
df2['temp_inside']=df2['temp_inside'].str.replace(",",".").astype(float)

gas_typeをOne-Hot化する。
# In[17]:


df3=pd.get_dummies(df2['refill gas'])
df2=pd.concat([df2,df3],axis=1)
df2.head()


# In[18]:


#temp_insideの空白値等を平均値で埋める
value=df2['temp_inside'].mean()
df2['temp_inside']=df2['temp_inside'].replace([""],value)
df2['temp_inside']=df2['temp_inside'].replace(["nan"],value)
df2.describe()


# In[19]:


# warning出力時にjupyterがインストールされているフォルダが見えてしまうので抑制する
import warnings
#warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=Warning)

回帰でエラー発生したため、df2をCSVで出力。df2['temp_inside']に「nan」が12個あることを確認。（上記のreplaceでは平均値に置き換わっていないようである）　以下で「nan」を平均値に確実に置き換える。
# In[10]:


df2['temp_inside'][93,95,97,98,99,100,102,201,203,261,267,268]=value


# In[13]:


#「nan」が置き換わったことの確認
df2['temp_inside'][93]


# In[19]:


#平均値で置換して良かったか分布を確認
df2['temp_inside'].describe()


# In[22]:


df2['temp_outside'].describe()

outsideとinsideの標準偏差の差が大きく、この差が燃費に影響している可能性あり。するとtemp_insideは「平均値」ではなく、直前の値に置き換えるべき
# In[20]:


df2['temp_inside'][93]=df2['temp_inside'][92]
df2['temp_inside'][95]=df2['temp_inside'][94]
df2['temp_inside'][97,98,99,100]=df2['temp_inside'][96]
df2['temp_inside'][102]=df2['temp_inside'][101]
df2['temp_inside'][201]=df2['temp_inside'][200]
df2['temp_inside'][203]=df2['temp_inside'][202]
df2['temp_inside'][261]=df2['temp_inside'][260]
df2['temp_inside'][267,268]=df2['temp_inside'][266]


# In[21]:


df2['temp_inside'].describe()


# In[22]:


#室内温度と室外温度の絶対値を新たな説明変数とし、室内温度と室外温度を除外する
df2['temp']=np.abs(df2['temp_inside']-df2['temp_outside'])
df2=df2.drop(['temp_inside','temp_outside'],axis=1)
df2.head()


# In[23]:


#相関係数を算出
df2.corr()


# In[27]:


#相関係数をヒートマップにして可視化
sns.heatmap(df2.corr())
plt.show()


# In[28]:


# 散布図行列を書いてみる
df10=df2[["consume","distance","speed","temp"]]
pd.plotting.scatter_matrix(df10,figsize=(20,10)) #数字または数字化した要素での散布図の作成


# In[23]:


#精度格納用
df_precision = pd.DataFrame(index=['MSE','RMSE','MAE','RMSE/MAE','決定係数','MSE（訓練）','RMSE（訓練）','MAE（訓練）','RMSE/MAE（訓練）','決定係数（訓練）'])
display(df_precision)


# In[24]:


#全てのデータで回帰
y=df2['consume'].values
X=df2.drop(["consume","refill liters","refill gas","gas_type"],axis=1).values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 回帰
regr = LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(y_train, y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(y_train, y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)


df_precision['線形（全）'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)

DAY1では、決定係数はtrain,testデータに分割する前の全体データで行っていたが、testデータで行うこととする。全体データでは0.2を超えていたが、testデータのみの決定係数は0.08を下回っている。consumeと相関の高いデータは、"distance","speed","temp","rain","sun","SP98"であり、これらで再回帰する。
その前にこの6変数間の相関係数を求め、相互に高い相関係数の変数あれば、調整する。
# In[30]:


df11=df2[["distance","speed","temp","rain","sun","SP98"]]
df11.corr()

上記回帰での説明変数から一部をカットする。
「sun,temp」「distance,speed」は相互に相関が高いので、この中からconsumeと相関が高い「temp」、「speed」を選択する。
次の結果のとおり、決定係数は若干下がるが、MSE、RMSE、MAEは若干改善。
# In[25]:


#consumeと相関の高いデータのうち、相互に相関が高い変数を除外して回帰　
y=df2['consume'].values
X=df2.drop(["consume","refill liters","refill gas","gas_type","sun","distance"],axis=1).values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 回帰
regr = LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(y_train, y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(y_train, y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)

df_precision['線形（高相）'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)


# In[26]:


#consumeと相関の高いデータのみで回帰、説明変数間の相関は考慮せず（"AC","E10"を最初の回帰の説明変数からはずす）
y=df2['consume'].values
X=df2.drop(["consume","refill liters","refill gas","gas_type","AC","E10"],axis=1).values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 回帰
regr = LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(y_train, y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(y_train, y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)

df_precision['線形（高相2）'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)


決定係数を初めとする全ての評価指数が悪化。2番目の回帰が一番評価が良いと判断。ヒートマップから選んだ説明変数による回帰はさらに悪化。
以下では多項式回帰を行う。上記のうち上から2つについて順に行う。
# In[27]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline


# In[28]:


#全てのデータで多項式回帰
y=df2['consume'].values
X=df2.drop(["consume","refill liters","refill gas","gas_type"],axis=1).values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 回帰
regr = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())])

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(y_train, y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(y_train, y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)

df_precision['多項式（全）'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)

（多項式回帰につき、degreeを変化させた結果を以下に記録する）
●degree=2
    決定係数=-0.988、MSE=1.367、RMSE=1.169、MAE=0.682
    
●degree=3で決定係数は-347とマイナス幅が拡大する。

（参考：線形回帰）
    決定係数=0.077、MSE=0.635、RMSE=0.797、MAE=0.65
    
多項式回帰はdegree=2から決定係数がマイナスとなり、このケースでは使えない。
# In[29]:


#consumeと相関の高いデータのうち、相互に相関が高い変数を除外して回帰　
y=df2['consume'].values
X=df2.drop(["consume","refill liters","refill gas","gas_type","sun","distance"],axis=1).values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 回帰
regr = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())])

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(y_train, y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(y_train, y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)

df_precision['多項式（高相）'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)

（多項式回帰につき、degreeを変化させた結果を以下に記録する）
●degree=2
    決定係数=0.243、MSE=0.521、RMSE=0.722、MAE=0.561
    
●degree=3
    決定係数=0.087、MSE=0.628、RMSE=0.793、MAE=0.606
    
●degree=4　から決定係数マイナス
    
（参考：線形回帰）
    決定係数=0.084、MSE=0.63、RMSE=0.794、MAE=0.646
    
degree=2は単線形回帰よりも性能が大きく上昇。（上記DAY1で見直したこと）
●temp_insideとtemp_outsideを、その差の絶対値のtempに置き換えて、やり直し。
●決定係数はtrain,testスプリットを行う前の全体データで算出していたものから、スプリット後のtestデータのみで算出し直し。

⇒前回DAY1提出時は多項式回帰の成績が良かったが、全般に成績が悪化。
　多項式回帰で、説明変数間の相関の高いデータを一部除外した場合に単回帰よりも成績が良くなった（決定係数　0.243）

以下、DAY2
変数の加工（対数化、標準化）を検討する。
# In[30]:


from IPython.display import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[31]:


#全体のtempとconsumeの関係を確認する
df_temp=df2.sort_values(by='temp',ascending=False)
plt.figure(figsize=(25,6))
y=df_temp["consume"]
x=df_temp["temp"]
plt.bar(x,y,color="b")
plt.xlabel("temp")
plt.ylabel("consume")
plt.show()


# In[32]:


#晴れている日の数を確認
df2['sun'].value_counts().to_dict()


# In[33]:


#sunの燃費の平均を算出し、「晴れていない日」「晴れている日」の燃費を比較
df_consume=df2.groupby(["sun"]).sum()["consume"]
df_consume_n=df2["sun"].value_counts()
dm_average=df_consume/df_consume_n
dm_average.sort_values(ascending=False)
#sunの日は燃費が若干良い


# In[34]:


#gas_typeの数を再確認
df2['gas_type'].value_counts().to_dict()


# In[35]:


#gas_typeの燃費の平均を算出し、各typeの燃費を比較
df_consume=df2.groupby(["gas_type"]).sum()["consume"]
df_consume_n=df2["gas_type"].value_counts()
dm_average=df_consume/df_consume_n
dm_average.sort_values(ascending=False)
#gas_typeの燃費の差はほとんどない


# In[36]:


#consumeの分布を確認する
df_consume=df2.sort_values(by='consume',ascending=False)
plt.figure(figsize=(25,6))
y=df_consume["consume"]
x = np.linspace(0, 387, 388)
plt.plot(x,y,color="b")
plt.xlabel("number")
plt.ylabel("consume")
plt.show()


# In[37]:


#distanceの分布を確認する(4月19日、斉藤先生のアドバイスで追加するもの)
df_distance=df2.sort_values(by='distance',ascending=False)
plt.figure(figsize=(25,6))
y=df_distance["distance"]
x = np.linspace(0, 387, 388)
plt.plot(x,y,color="b")
plt.xlabel("number")
plt.ylabel("distance")
plt.show()

グラフの形から目的変数のconsumeだけは、対数化しても良いと思われるので対数化する。
またdistanceは、べき正規変換の1つである対数化を行う。
# In[38]:


df3=df2
a=np.array(df3['consume'])
df3['consume']=np.log(a)
df3.head()


# In[39]:


#logを取った後のconsumeの分布を確認する
df_consume=df3.sort_values(by='consume',ascending=False)
plt.figure(figsize=(25,6))
y=df_consume["consume"]
x = np.linspace(0, 387, 388)
plt.plot(x,y,color="b")
plt.xlabel("number")
plt.ylabel("consume")
plt.show()


# In[40]:


b=np.array(df3['distance'])
df3['distance']=np.log(b)
df3.head()


# In[41]:


#logを取った後のdistanceの分布を確認する
df_distance=df3.sort_values(by='distance',ascending=False)
plt.figure(figsize=(25,6))
y=df_distance["distance"]
x = np.linspace(0, 387, 388)
plt.plot(x,y,color="b")
plt.xlabel("number")
plt.ylabel("distance")
plt.show()


# In[390]:


#（参考：べき正規変換を検討）使用せず
df100=df3['distance']/8


# In[391]:


#（参考：べき正規変換を検討）使用せず
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
h=np.array(df100)
h=h.reshape(1, -1)
L=pt.fit_transform(h)
df100=pd.DataFrame(L.reshape(-1, 1))
df100.head()


# In[392]:


#（参考：べき正規変換を検討）使用せず
df_distance=df100.sort_values(by=[0],ascending=False)
plt.figure(figsize=(25,6))
y=df_distance[0]
x = np.linspace(0, 387, 388)
plt.plot(x,y,color="b")
plt.xlabel("number")
plt.ylabel("distance")
plt.show()

consume、distanceを対数化したデータで前出の回帰を行う
# In[42]:


#consumeと相関の高いデータのうち、相互に相関が高い変数を除外して回帰　
y=df3['consume'].values
X=df3.drop(["consume","refill liters","refill gas","gas_type","sun","distance"],axis=1).values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 回帰
regr = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())])

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(np.e**y_test, np.e**y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(np.e**y_test, np.e**y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(np.e**y_train, np.e**y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)

df_precision['多項式（高相2）'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)

Cosume,distanceの対数化前と比較し、決定係数は悪化、他の指標は大幅に改善
（対数化前）　　　　　　　　　　　　　　　（対数化後）
決定係数=0.243　　　　　　　→　　　0.214
MSE=0.521　　　　　　　　　　　→　　　0.603
RMSE=0.722　　　　　　　　　　　→　　　0.777
MAE=0.561　　　　　　　　　　　→　　　0.594
# In[43]:


from sklearn.model_selection import KFold # 交差検証法に関する関数
from sklearn.metrics import mean_absolute_error # 回帰問題における性能評価に関する関数

#consumeと相関の高いデータのうち、相互に相関が高い変数を除外して回帰　
y=df3['consume'].values
X=df3.drop(["consume","refill liters","refill gas","gas_type","sun","distance"],axis=1).values

n_split = 5 # グループ数を設定（今回は5分割）

cross_valid_mae = 0
cross_valid_mae2 = 0

cro_val_決定 = 0
cro_val_決定2 = 0

split_num = 1

# テスト役を交代させながら学習と評価を繰り返す
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] #学習用データ
    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

# 学習用データを使って回帰
    regr = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())])

    regr.fit(X_train, y_train)
    
    # テストデータに対する予測を実行
    y_pred = regr.predict(X_test) 
    
    # 訓練データに対する予測を実行
    y_pred2 = regr.predict(X_train)
    
    # テストデータに対するMAEを計算
    mae = mean_absolute_error(np.e**y_test, np.e**y_pred)
    cro_val_決定 = regr.score(X_test, y_test)
    print("Fold %s"%split_num)
    print("MAE = %s"%round(mae, 3))
    print("決定係数=%s"%regr.score(X_test, y_test))
    print()

    # 訓練データに対するMAEを計算
    mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)
    cro_val_決定2 = regr.score(X_train, y_train)
    
    cross_valid_mae += mae #後で平均を取るためにMAEを加算
    cross_valid_mae2 += mae2
    cro_val_決定 += cro_val_決定
    cro_val_決定2 += cro_val_決定2
    split_num += 1

# MAEの平均値を最終的な汎化誤差値とする
final_mae = cross_valid_mae / n_split
final_mae2 = cross_valid_mae2 / n_split
final_決定 = cro_val_決定 / n_split
final_決定2 = cro_val_決定2 / n_split
print("Cross Validation MAE = %s"%round(final_mae, 3))
print("Cross Vali 決定係数 = %s"%round(final_決定, 5))
print()
print("（訓練）Cross Validation MAE = %s"%round(final_mae2, 3))
print("（訓練）Cross Validation 決定 = %s"%round(final_決定2, 5))


# In[44]:


# 標準化。distance、speed、tempを標準化する。
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
df3["distance"]=stdsc.fit_transform(df3[["distance"]].values)
df3["speed"]=stdsc.fit_transform(df3[["speed"]].values)
df3["temp"]=stdsc.fit_transform(df3[["temp"]].values)
df3.head()


# In[45]:


#consumeと相関の高いデータのうち、相互に相関が高い変数を除外して回帰　
y=df3['consume'].values
X=df3.drop(["consume","refill liters","refill gas","gas_type","sun","distance"],axis=1).values

n_split = 5 # グループ数を設定（今回は5分割）

cross_valid_mae = 0
cross_valid_mae2 = 0

cro_val_決定 = 0
cro_val_決定2 = 0

split_num = 1

# テスト役を交代させながら学習と評価を繰り返す
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] #学習用データ
    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

# 学習用データを使って回帰
    regr = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())])

    regr.fit(X_train, y_train)
    
    # テストデータに対する予測を実行
    y_pred = regr.predict(X_test)   
    
    # 訓練データに対する予測を実行
    y_pred2 = regr.predict(X_train)
    
    # テストデータに対するMAEを計算
    mae = mean_absolute_error(np.e**y_test, np.e**y_pred)
    cro_val_決定 = regr.score(X_test, y_test)
    
    print("Fold %s"%split_num)
    print("MAE = %s"%round(mae, 3))
    print("決定係数=%s"%regr.score(X_test, y_test))
    print()

    # 訓練データに対するMAEを計算
    mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)
    cro_val_決定2 = regr.score(X_train, y_train)    
    
    cross_valid_mae += mae #後で平均を取るためにMAEを加算
    cross_valid_mae2 += mae2
    cro_val_決定 += cro_val_決定
    cro_val_決定2 += cro_val_決定2
    split_num += 1

# MAEの平均値を最終的な汎化誤差値とする
final_mae = cross_valid_mae / n_split
final_mae2 = cross_valid_mae2 / n_split
final_決定 = cro_val_決定 / n_split
final_決定2 = cro_val_決定2 / n_split
print("Cross Validation MAE = %s"%round(final_mae, 3))
print("Cross Vali 決定係数 = %s"%round(final_決定, 5))
print()
print("（訓練データ）Cross Validation MAE = %s"%round(final_mae2, 3))
print("（訓練）Cross Validation 決定 = %s"%round(final_決定2, 5))

3つの説明変数（distance、speed、temp）の標準化は、交差検証法における（平均）MAEの改善には影響がなかった。次に相互に相関が高い各2変数の「sun,temp」「distance,speed」を順次白色化する。
# In[46]:


df3.head()#データフレームの変数を確認するため表示


# In[47]:


#「sun,temp」の白色化
data1=df3.drop(["consume","distance","speed","gas_type","AC","rain","refill liters","refill gas","E10","SP98"],axis=1).values
cov = np.cov(data1, rowvar=0) # 分散・共分散を求める
_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
data1_decorr = np.dot(S.T, data1.T).T #データを無相関化


# In[48]:


#交差検証法にあたり、「sun,temp」の白色化したもの、およびdistanceとspeedのうちspeedを採用
y=df3['consume'].values
df20=df3.drop(["distance","consume","gas_type","refill liters","refill gas","sun","temp"],axis=1)
X=pd.concat([df20,pd.DataFrame(data1_decorr)],axis=1).values

n_split = 5 # グループ数を設定（今回は5分割）

cross_valid_mae = 0
cross_valid_mae2 = 0
cro_val_決定 = 0
cro_val_決定2 = 0
split_num = 1

# テスト役を交代させながら学習と評価を繰り返す
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] #学習用データ
    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

# 学習用データを使って回帰
    regr = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())])

    regr.fit(X_train, y_train)
    
    # テストデータに対する予測を実行
    y_pred = regr.predict(X_test)
    
    # 訓練データに対する予測を実行
    y_pred2 = regr.predict(X_train)
    
    # テストデータに対するMAEを計算
    mae = mean_absolute_error(np.e**y_test, np.e**y_pred)
    cro_val_決定 = regr.score(X_test, y_test)    
    print("Fold %s"%split_num)
    print("MAE = %s"%round(mae, 3))
    print()
    
    # 訓練データに対するMAEを計算
    mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)
    cro_val_決定2 = regr.score(X_train, y_train)    
    
    cross_valid_mae += mae #後で平均を取るためにMAEを加算
    cross_valid_mae2 += mae2
    cro_val_決定 += cro_val_決定
    cro_val_決定2 += cro_val_決定2    
    split_num += 1

# MAEの平均値を最終的な汎化誤差値とする
final_mae = cross_valid_mae / n_split
final_mae2 = cross_valid_mae2 / n_split
print("Cross Validation MAE = %s"%round(final_mae, 3))
print("Cross Vali 決定係数 = %s"%round(final_決定, 5))
print()
print("（訓練データ）Cross Validation MAE = %s"%round(final_mae2, 3))
print("（訓練）Cross Validation 決定 = %s"%round(final_決定2, 5))


# In[49]:


#「distance,speed」の白色化
data2=df3.drop(["consume","sun","temp","gas_type","AC","rain","refill liters","refill gas","E10","SP98"],axis=1).values
cov = np.cov(data2, rowvar=0) # 分散・共分散を求める
_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
data2_decorr = np.dot(S.T, data2.T).T #データを無相関化


# In[50]:


df20.head(1)


# In[51]:


df3.head(1)


# In[52]:


#交差検証法にあたり、白色化した「sun,temp」、「distance,speed」を全て採用
y=df3['consume'].values
df30=df3.drop(["distance","speed","consume","gas_type","refill liters","refill gas","sun","temp","AC","rain","E10","SP98"],axis=1)
X=pd.concat([df20,df30,pd.DataFrame(data1_decorr)],axis=1).values

n_split = 5 # グループ数を設定（今回は5分割）

cross_valid_mae = 0
cross_valid_mae2 = 0
cro_val_決定 = 0
cro_val_決定2 = 0
split_num = 1

# テスト役を交代させながら学習と評価を繰り返す
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] #学習用データ
    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

# 学習用データを使って回帰
    regr = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())])

    regr.fit(X_train, y_train)
    
    # テストデータに対する予測を実行
    y_pred = regr.predict(X_test)
    
    # 訓練データに対する予測を実行
    y_pred2 = regr.predict(X_train)
    
    # テストデータに対するMAEを計算
    mae = mean_absolute_error(np.e**y_test, np.e**y_pred)
    cro_val_決定 = regr.score(X_test, y_test)      
    print("Fold %s"%split_num)
    print("MAE = %s"%round(mae, 3))
    print()
    
    # 訓練データに対するMAEを計算
    mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)
    cro_val_決定2 = regr.score(X_train, y_train)    
    
    cross_valid_mae += mae #後で平均を取るためにMAEを加算
    cross_valid_mae2 += mae2
    cro_val_決定 += cro_val_決定
    cro_val_決定2 += cro_val_決定2       
    split_num += 1

# MAEの平均値を最終的な汎化誤差値とする
final_mae = cross_valid_mae / n_split
final_mae2 = cross_valid_mae2 / n_split
print("Cross Validation MAE = %s"%round(final_mae, 3))
print("Cross Vali 決定係数 = %s"%round(final_決定, 5))
print()
print("（訓練データ）Cross Validation MAE = %s"%round(final_mae2, 3))
print("（訓練）Cross Validation 決定 = %s"%round(final_決定2, 5))

白色化した説明変数により交差検証法を行ってもMAEは、むしろ悪化した。
サポートベクター回帰を行ってみる。
# In[53]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# In[54]:


#consumeと相関の高いデータのうち、相互に相関が高い変数を除外して回帰(白色化する前の性能が良かった変数を選択)
y=df3['consume'].values
X=df3.drop(["consume","refill liters","refill gas","gas_type","sun","distance"],axis=1).values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#交差検証データのジェネレータ
def gen_cv():
    m_train = np.floor(len(y)*0.8).astype(int)
    train_idx = np.arange(m_train)
    test_idx = np.arange(m_train, len(y))
    yield (train_idx, test_idx)

train_idx = next(gen_cv())[0]

# ハイパーパラメータのチューニング
params_cnt = 10
params = {"C":np.logspace(0,2,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}
gridsearch = GridSearchCV(SVR(), params, cv=gen_cv(), scoring="r2", return_train_score=True)
gridsearch.fit(X, y)


# In[55]:


print("C, εのチューニング")
print("最適なパラメーター =", gridsearch.best_params_)
print("精度 =", gridsearch.best_score_)
print()    


# In[56]:


# チューニングしたC,εでフィット
regr = SVR(C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
train_idx = next(gen_cv())[0]
valid_idx = next(gen_cv())[1]
regr.fit(X[train_idx, :], y[train_idx])

print("決定係数=%s"%regr.score(X[test_idx, :], y[test_idx]))
print()
print("※参考")
print("訓練データの精度 =", regr.score(X[train_idx, :], y[train_idx]))
print("交差検証データの精度 =", regr.score(X[valid_idx, :], y[valid_idx]))

以下、DAY3
# In[57]:


#　ステップワイズ法
# estimatorにモデルをセット
# 回帰問題であるためLinearRegressionを使用
estimator = LinearRegression(normalize=True)

# RFECVは交差検証によってステップワイズ法による特徴選択を行う
# cvにはFold（=グループ）の数，scoringには評価指標を指定する
# 回帰なのでneg_mean_absolute_errorを評価指標に指定
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator, cv=5, scoring='neg_mean_absolute_error')


# In[58]:


y=df3['consume'].values
df30=df3.drop(["distance","speed","consume","gas_type","refill liters","refill gas","sun","temp","AC","rain","E10","SP98"],axis=1)
X=pd.concat([df20,df30,pd.DataFrame(data1_decorr)],axis=1).values

# fitで特徴選択を実行
rfecv.fit(X, y)


# In[59]:


# 特徴のランキングを表示（1が最も重要な特徴）
print('Feature ranking: \n{}'.format(rfecv.ranking_))


# In[60]:


# 特徴数とスコアの変化をプロット
# 負のMAEが評価基準になっており，値がゼロに近いほど汎化誤差は小さい
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[61]:


# rfecv.support_でランキング1位以外はFalseとするindexを取得できる
# Trueになっている特徴を使用すれば汎化誤差は最小となる
rfecv.support_


# In[62]:


# 埋め込み法
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


# In[63]:


# estimatorにモデルをセット
# LassoCVを使って、正則化の強さは自動決定
estimator = LassoCV(normalize=True, cv=10)

# モデルの情報を使って特徴選択を行う場合は、SelectFromModelを使う
# 今回は係数が1e-5以下である特徴を削除する
# 係数のしきい値はthresholdで指定する
sfm = SelectFromModel(estimator, threshold=1e-5)


# In[64]:


# fitで特徴選択を実行
sfm.fit(X, y)


# In[65]:


# get_support関数で使用する特徴のインデックスを使用
# Trueになっている特徴が使用する特徴
sfm.get_support()


# In[66]:


# 削除すべき特徴の名前を取得 
removed_idx  = ~sfm.get_support()
pd.DataFrame(X).columns[removed_idx]


# In[67]:


pd.DataFrame(X).head(1)


# In[68]:


# LASSOで得た各特徴の係数の値を確認してみる
# 係数の絶対値を取得
abs_coef = np.abs(sfm.estimator_.coef_)
abs_coef


# In[69]:


# 係数を棒グラフで表示
plt.barh(np.arange(0, len(abs_coef)), abs_coef, tick_label=pd.DataFrame(X).columns.values)
plt.show()
#SP98も係数がほぼゼロと判明

Lassoの特徴選択によっても、削除するべき変数はなかった。
# In[70]:


import graphviz
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO


# In[72]:


from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import tree

#import decision_tree as dt

regr = DecisionTreeRegressor(criterion='mae', max_depth=2)
regr = regr.fit(X_train, y_train)

# テストデータに対する予測を実行
y_pred = regr.predict(X_test)

mse = mean_squared_error(np.e**y_test, np.e**y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(np.e**y_test, np.e**y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(np.e**y_train, np.e**y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)

df_precision['決定木'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)
    

決定木回帰ではmax_depth=2でもtestデータの決定係数はマイナスであり、最初から過学習が起こっており、使えない。
ちなみにmax_depth=Noneの場合、訓練データの決定係数は0.96まで上昇するが、testデータの決定係数は-1.59に悪化する。
# In[73]:


from sklearn.ensemble import RandomForestRegressor # ランダムフォレスト回帰用
regr = RandomForestRegressor(criterion='mae', max_depth=3,n_estimators=500)
regr = regr.fit(X_train, y_train)

# テストデータに対する予測を実行
y_pred = regr.predict(X_test)
 
mse = mean_squared_error(np.e**y_test, np.e**y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(np.e**y_test, np.e**y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(np.e**y_train, np.e**y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)

df_precision['ランダム'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)    

RandomForest回帰ではmax_depth=3でtestデータの決定係数は0.115。MSEは0.236と一番低い。
# In[74]:


from sklearn.ensemble import AdaBoostRegressor
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3,criterion="mae"),
                                           n_estimators=10, random_state=1234)

regr = regr.fit(X_train, y_train)

# テストデータに対する予測を実行
y_pred = regr.predict(X_test)

mse = mean_squared_error(np.e**y_test, np.e**y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(np.e**y_test, np.e**y_pred)

y_pred2 = regr.predict(X_train)
mse2 = mean_squared_error(np.e**y_train, np.e**y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)

score1 = regr.score(X_test, y_test)
score2 = regr.score(X_train, y_train)

df_precision['アダブスト'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(mse2,3),round(rmse2,3),round(mae2,3),round(rmse2/mae2,3),round(score2,3)]
display(df_precision)    

ランダムフォレスト回帰の決定係数0.115が後半の3つの中で最良。決定木、アダブーストはマイナスやゼロ近辺。
# In[75]:


df_x=pd.DataFrame(X)
df_x.head(1)


# In[76]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD,RMSprop, Adagrad, Adadelta, Adam

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes

NNの最適化手法（交差検証法）を後記のとおり、試した。
（交差検証法におけるMAEの平均比較）・・・RMSpropが最良。次点がSGD。グラフのとおり、成績の悪いものほど、epochsの進捗に対するMAEの下げ方が鈍い。
SGD：0.646　（活性化関数はrelu）

Adam：0.664　（活性化関数はrelu）

RMSprop：0.627　（活性化関数はrelu）

Adagrad：0.937　（活性化関数はrelu）

Adadelta：3.63　（活性化関数はrelu）
# In[77]:


#交差検証法にあたり、白色化した「sun,temp」、「distance,speed」を全て採用
y=df3['consume'].values
df30=df3.drop(["distance","speed","consume","gas_type","refill liters","refill gas","sun","temp","AC","rain","E10","SP98"],axis=1)
X=pd.concat([df20,df30,pd.DataFrame(data1_decorr)],axis=1).values

n_split = 5 # グループ数を設定（今回は5分割）

cross_valid_mae = 0
cross_valid_mae2 = 0
cro_val_決定 = 0
cro_val_決定2 = 0
split_num = 1

# テスト役を交代させながら学習と評価を繰り返す
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] #学習用データ
    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

# 学習用データを使って回帰
    regr = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())])

    regr.fit(X_train, y_train)
    
    # テストデータに対する予測を実行
    y_pred = regr.predict(X_test)
    
    # 訓練データに対する予測を実行
    y_pred2 = regr.predict(X_train)
    
    # テストデータに対するMAEを計算
    mae = mean_absolute_error(np.e**y_test, np.e**y_pred)
    cro_val_決定 = regr.score(X_test, y_test)      
    print("Fold %s"%split_num)
    print("MAE = %s"%round(mae, 3))
    print()
    
    # 訓練データに対するMAEを計算
    mae2 = mean_absolute_error(np.e**y_train, np.e**y_pred2)
    cro_val_決定2 = regr.score(X_train, y_train)    
    
    cross_valid_mae += mae #後で平均を取るためにMAEを加算
    cross_valid_mae2 += mae2
    cro_val_決定 += cro_val_決定
    cro_val_決定2 += cro_val_決定2       
    split_num += 1

# MAEの平均値を最終的な汎化誤差値とする
final_mae = cross_valid_mae / n_split
final_mae2 = cross_valid_mae2 / n_split
print("Cross Validation MAE = %s"%round(final_mae, 3))
print("Cross Vali 決定係数 = %s"%round(final_決定, 5))
print()
print("（訓練データ）Cross Validation MAE = %s"%round(final_mae2, 3))
print("（訓練）Cross Validation 決定 = %s"%round(final_決定2, 5))


# In[78]:


def run(X, y):    

    # ------ 最適化手法 ------
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
    # rms = RMSprop(lr=0.01)
    # adag = Adagrad(lr=0.01)
    # adad = Adadelta(lr=0.01)
    # adam = Adam(lr=0.01)
    # -----------------------------

    n_split = 5
    cross_valid_mae = 0
    split_num = 1

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        model = Sequential()
        model.add(Dense(7, activation='relu', input_dim=7))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1)) #, activation='softmax')
        
        # 回帰にはcategorical_crossentropyではなくmae。
        model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['mae'])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        fit = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=20,
                        validation_data=(X_test, y_test),
                       verbose=0)

        
        print(f"Cross Validation Try {split_num} / {n_split}")
        # テストデータに対するMAEを計算
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(np.e**y_test, np.e**y_pred_test)
        # cro_val_決定 = model.best_score_

        print(f"MAE : {mae}")
        # print(f"決定係数 : {cro_val_決定}")
        
        
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        # グラフ化
        df = pd.DataFrame(fit.history)
        axL.plot(df[["loss", "val_loss"]])
        axL.set_ylabel("loss")
        axL.set_xlabel("epoch")

        axR.plot(df[["val_mean_absolute_error"]])
        axR.set_ylabel("mae")
        axR.set_xlabel("epoch")
        fig.show()

        cross_valid_mae += mae #後で平均を取るためにMAEを加算
        # cro_val_決定 += cro_val_決定
        split_num += 1

    # MAEの平均値を最終的な汎化誤差値とする
    final_mae = cross_valid_mae / n_split
    
    print()
    print("Cross Validation MAE = %s"% (round(final_mae, 3)))

run(X, y)

# show the model summary（defで作成した関数について）
#run().summary()


# In[79]:


def run(X, y):    

    # ------ 最適化手法 ------
    #sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
    # rms = RMSprop(lr=0.01)
    # adag = Adagrad(lr=0.01)
    # adad = Adadelta(lr=0.01)
    # adam = Adam(lr=0.01)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    n_split = 5
    cross_valid_mae = 0
    split_num = 1

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        model = Sequential()
        model.add(Dense(7, activation='relu', input_dim=7))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1)) #, activation='softmax')
        
        # 回帰にはcategorical_crossentropyではなくmae。
        model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mae'])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        fit = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=20,
                        validation_data=(X_test, y_test),
                       verbose=0)

        
        print(f"Cross Validation Try {split_num} / {n_split}")
        # テストデータに対するMAEを計算
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(np.e**y_test, np.e**y_pred_test)
        print(f"MAE : {mae}")

        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        # グラフ化
        df = pd.DataFrame(fit.history)
        axL.plot(df[["loss", "val_loss"]])
        axL.set_ylabel("loss")
        axL.set_xlabel("epoch")

        axR.plot(df[["val_mean_absolute_error"]])
        axR.set_ylabel("mae")
        axR.set_xlabel("epoch")
        fig.show()

        cross_valid_mae += mae #後で平均を取るためにMAEを加算
        split_num += 1

    # MAEの平均値を最終的な汎化誤差値とする
    final_mae = cross_valid_mae / n_split
    print()
    print("Cross Validation MAE = %s"% (round(final_mae, 3)))

run(X, y)


# In[80]:


def run(X, y):    

    # ------ 最適化手法 ------
    #sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
    rms = RMSprop(lr=0.01)
    # adag = Adagrad(lr=0.01)
    # adad = Adadelta(lr=0.01)
    # adam = Adam(lr=0.01)
    # -----------------------------

    n_split = 5
    cross_valid_mae = 0
    split_num = 1

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        model = Sequential()
        model.add(Dense(7, activation='relu', input_dim=7))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1)) #, activation='softmax')
        
        # 回帰にはcategorical_crossentropyではなくmae。
        model.compile(loss='mean_squared_error',
                  optimizer=rms,
                  metrics=['mae'])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        fit = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=20,
                        validation_data=(X_test, y_test),
                       verbose=0)

        
        print(f"Cross Validation Try {split_num} / {n_split}")
        # テストデータに対するMAEを計算
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(np.e**y_test, np.e**y_pred_test)
        print(f"MAE : {mae}")
        
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        # グラフ化
        df = pd.DataFrame(fit.history)
        axL.plot(df[["loss", "val_loss"]])
        axL.set_ylabel("loss")
        axL.set_xlabel("epoch")

        axR.plot(df[["val_mean_absolute_error"]])
        axR.set_ylabel("mae")
        axR.set_xlabel("epoch")
        fig.show()

        cross_valid_mae += mae #後で平均を取るためにMAEを加算
        split_num += 1

    # MAEの平均値を最終的な汎化誤差値とする
    final_mae = cross_valid_mae / n_split
    print()
    print("Cross Validation MAE = %s"% (round(final_mae, 3)))

run(X, y)


# In[81]:


def run(X, y):    

    # ------ 最適化手法 ------
    #sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
    #rms = RMSprop(lr=0.01)
    adag = Adagrad(lr=0.01)
    # adad = Adadelta(lr=0.01)
    # adam = Adam(lr=0.01)
    # -----------------------------

    n_split = 5
    cross_valid_mae = 0
    split_num = 1

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        model = Sequential()
        model.add(Dense(7, activation='relu', input_dim=7))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1)) #, activation='softmax')
        
        # 回帰にはcategorical_crossentropyではなくmae。
        model.compile(loss='mean_squared_error',
                  optimizer=adag,
                  metrics=['mae'])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        fit = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=20,
                        validation_data=(X_test, y_test),
                       verbose=0)

        
        print(f"Cross Validation Try {split_num} / {n_split}")
        # テストデータに対するMAEを計算
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(np.e**y_test, np.e**y_pred_test)
        print(f"MAE : {mae}")
        
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        # グラフ化
        df = pd.DataFrame(fit.history)
        axL.plot(df[["loss", "val_loss"]])
        axL.set_ylabel("loss")
        axL.set_xlabel("epoch")

        axR.plot(df[["val_mean_absolute_error"]])
        axR.set_ylabel("mae")
        axR.set_xlabel("epoch")
        fig.show()

        cross_valid_mae += mae #後で平均を取るためにMAEを加算
        split_num += 1

    # MAEの平均値を最終的な汎化誤差値とする
    final_mae = cross_valid_mae / n_split
    print()
    print("Cross Validation MAE = %s"% (round(final_mae, 3)))

run(X, y)


# In[82]:


def run(X, y):    

    # ------ 最適化手法 ------
    #sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
    #rms = RMSprop(lr=0.01)
    # adag = Adagrad(lr=0.01)
    adad = Adadelta(lr=0.01)
    # adam = Adam(lr=0.01)
    # -----------------------------

    n_split = 5
    cross_valid_mae = 0
    split_num = 1

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        model = Sequential()
        model.add(Dense(7, activation='relu', input_dim=7))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1)) #, activation='softmax')
        
        # 回帰にはcategorical_crossentropyではなくmae。
        model.compile(loss='mean_squared_error',
                  optimizer=adad,
                  metrics=['mae'])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        fit = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=20,
                        validation_data=(X_test, y_test),
                       verbose=0)

        
        print(f"Cross Validation Try {split_num} / {n_split}")
        # テストデータに対するMAEを計算
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(np.e**y_test, np.e**y_pred_test)
        print(f"MAE : {mae}")
        
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        # グラフ化
        df = pd.DataFrame(fit.history)
        axL.plot(df[["loss", "val_loss"]])
        axL.set_ylabel("loss")
        axL.set_xlabel("epoch")

        axR.plot(df[["val_mean_absolute_error"]])
        axR.set_ylabel("mae")
        axR.set_xlabel("epoch")
        fig.show()

        cross_valid_mae += mae #後で平均を取るためにMAEを加算
        split_num += 1

    # MAEの平均値を最終的な汎化誤差値とする
    final_mae = cross_valid_mae / n_split
    print()
    print("Cross Validation MAE = %s"% (round(final_mae, 3)))

run(X, y)

DAY4の内容も試してみる（LSTMなど）
# In[83]:


from keras.layers import LSTM
import math


# In[84]:


pd.DataFrame(X).head()


# In[85]:


y=df3['consume'].values
df30=df3.drop(["distance","speed","consume","gas_type","refill liters","refill gas","sun","temp","AC","rain","E10","SP98"],axis=1)
X=pd.concat([df20,df30,pd.DataFrame(data1_decorr)],axis=1).values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(5, input_shape=(1, 7)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2)


# In[86]:


# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train, trainPredict[:,0]))
print('Train Score: %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test, testPredict[:,0]))
print('Test Score: %.3f RMSE' % (testScore))

