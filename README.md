# Trabalho Final

**Autor**: Matheus Jericó Palhares <br>
**LinkedIn**: https://linkedin.com/in/matheusjerico

### Dataset - Gender Recognition by Voice
**Link**:<br> https://www.kaggle.com/mlg-ulb/creditcardfraud

**Contexto**: <br>
Esse dataset foi criado com objetivo de identificar uma voz como masculina ou feminina, com base nas propriedades acústicas da voz e da fala. O Dataset consistem em 3168 registros de vozes, coletadas de falantes masculinos e femininos. As amostras de voz são pré-processadas por análise acústica em R usando os pacotes seewave e tuneR, com uma faixa de frequência analisada de 0hz-280hz (faixa vocal humana).

**Conteúdo**:<br>
Features do dataset:
- meanfreq: mean frequency (in kHz)
- sd: standard deviation of frequency
- median: median frequency (in kHz)
- Q25: first quantile (in kHz)
- Q75: third quantile (in kHz)
- IQR: interquantile range (in kHz)
- skew: skewness (see note in specprop description)
- kurt: kurtosis (see note in specprop description)
- sp.ent: spectral entropy
- sfm: spectral flatness
- mode: mode frequency
- centroid: frequency centroid (see specprop)
- peakf: peak frequency (frequency with highest energy)
- meanfun: average of fundamental frequency measured across acoustic signal
- minfun: minimum fundamental frequency measured across acoustic signal
- maxfun: maximum fundamental frequency measured across acoustic signal
- meandom: average of dominant frequency measured across acoustic signal
- mindom: minimum of dominant frequency measured across acoustic signal
- maxdom: maximum of dominant frequency measured across acoustic signal
- dfrange: range of dominant frequency measured across acoustic signal
- modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
- label: male or female


**Resolução**: <br>
Para resolução do problema, utilizaremos algoritmos de aprendizagem supervisionada.

## 1. Carregando Bibliotecas


```python
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, classification_report, f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
from pandas_profiling import ProfileReport
```

## 2. Carregando dados

- Foi feito o donwload dos dados e inseridos no diretório ```/dataset```;
- Utilizamos a biblioteca Pandas para carregar o dataset.


```python
dataset = pd.read_csv("dataset/voice.csv")
```


```python
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3168 entries, 0 to 3167
    Data columns (total 21 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   meanfreq  3168 non-null   float64
     1   sd        3168 non-null   float64
     2   median    3168 non-null   float64
     3   Q25       3168 non-null   float64
     4   Q75       3168 non-null   float64
     5   IQR       3168 non-null   float64
     6   skew      3168 non-null   float64
     7   kurt      3168 non-null   float64
     8   sp.ent    3168 non-null   float64
     9   sfm       3168 non-null   float64
     10  mode      3168 non-null   float64
     11  centroid  3168 non-null   float64
     12  meanfun   3168 non-null   float64
     13  minfun    3168 non-null   float64
     14  maxfun    3168 non-null   float64
     15  meandom   3168 non-null   float64
     16  mindom    3168 non-null   float64
     17  maxdom    3168 non-null   float64
     18  dfrange   3168 non-null   float64
     19  modindx   3168 non-null   float64
     20  label     3168 non-null   object 
    dtypes: float64(20), object(1)
    memory usage: 519.9+ KB


**ANÁLISE**:
- Dataset não possui valores NaN;
- 20 variáveis;
- Label é a coluna target;
- Todos as variáveis independentes são numéricos.


```python
dataset.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meanfreq</th>
      <th>sd</th>
      <th>median</th>
      <th>Q25</th>
      <th>Q75</th>
      <th>IQR</th>
      <th>skew</th>
      <th>kurt</th>
      <th>sp.ent</th>
      <th>sfm</th>
      <th>mode</th>
      <th>centroid</th>
      <th>meanfun</th>
      <th>minfun</th>
      <th>maxfun</th>
      <th>meandom</th>
      <th>mindom</th>
      <th>maxdom</th>
      <th>dfrange</th>
      <th>modindx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
      <td>3168.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.180907</td>
      <td>0.057126</td>
      <td>0.185621</td>
      <td>0.140456</td>
      <td>0.224765</td>
      <td>0.084309</td>
      <td>3.140168</td>
      <td>36.568461</td>
      <td>0.895127</td>
      <td>0.408216</td>
      <td>0.165282</td>
      <td>0.180907</td>
      <td>0.142807</td>
      <td>0.036802</td>
      <td>0.258842</td>
      <td>0.829211</td>
      <td>0.052647</td>
      <td>5.047277</td>
      <td>4.994630</td>
      <td>0.173752</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.029918</td>
      <td>0.016652</td>
      <td>0.036360</td>
      <td>0.048680</td>
      <td>0.023639</td>
      <td>0.042783</td>
      <td>4.240529</td>
      <td>134.928661</td>
      <td>0.044980</td>
      <td>0.177521</td>
      <td>0.077203</td>
      <td>0.029918</td>
      <td>0.032304</td>
      <td>0.019220</td>
      <td>0.030077</td>
      <td>0.525205</td>
      <td>0.063299</td>
      <td>3.521157</td>
      <td>3.520039</td>
      <td>0.119454</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.039363</td>
      <td>0.018363</td>
      <td>0.010975</td>
      <td>0.000229</td>
      <td>0.042946</td>
      <td>0.014558</td>
      <td>0.141735</td>
      <td>2.068455</td>
      <td>0.738651</td>
      <td>0.036876</td>
      <td>0.000000</td>
      <td>0.039363</td>
      <td>0.055565</td>
      <td>0.009775</td>
      <td>0.103093</td>
      <td>0.007812</td>
      <td>0.004883</td>
      <td>0.007812</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.163662</td>
      <td>0.041954</td>
      <td>0.169593</td>
      <td>0.111087</td>
      <td>0.208747</td>
      <td>0.042560</td>
      <td>1.649569</td>
      <td>5.669547</td>
      <td>0.861811</td>
      <td>0.258041</td>
      <td>0.118016</td>
      <td>0.163662</td>
      <td>0.116998</td>
      <td>0.018223</td>
      <td>0.253968</td>
      <td>0.419828</td>
      <td>0.007812</td>
      <td>2.070312</td>
      <td>2.044922</td>
      <td>0.099766</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.184838</td>
      <td>0.059155</td>
      <td>0.190032</td>
      <td>0.140286</td>
      <td>0.225684</td>
      <td>0.094280</td>
      <td>2.197101</td>
      <td>8.318463</td>
      <td>0.901767</td>
      <td>0.396335</td>
      <td>0.186599</td>
      <td>0.184838</td>
      <td>0.140519</td>
      <td>0.046110</td>
      <td>0.271186</td>
      <td>0.765795</td>
      <td>0.023438</td>
      <td>4.992188</td>
      <td>4.945312</td>
      <td>0.139357</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.199146</td>
      <td>0.067020</td>
      <td>0.210618</td>
      <td>0.175939</td>
      <td>0.243660</td>
      <td>0.114175</td>
      <td>2.931694</td>
      <td>13.648905</td>
      <td>0.928713</td>
      <td>0.533676</td>
      <td>0.221104</td>
      <td>0.199146</td>
      <td>0.169581</td>
      <td>0.047904</td>
      <td>0.277457</td>
      <td>1.177166</td>
      <td>0.070312</td>
      <td>7.007812</td>
      <td>6.992188</td>
      <td>0.209183</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.251124</td>
      <td>0.115273</td>
      <td>0.261224</td>
      <td>0.247347</td>
      <td>0.273469</td>
      <td>0.252225</td>
      <td>34.725453</td>
      <td>1309.612887</td>
      <td>0.981997</td>
      <td>0.842936</td>
      <td>0.280000</td>
      <td>0.251124</td>
      <td>0.237636</td>
      <td>0.204082</td>
      <td>0.279114</td>
      <td>2.957682</td>
      <td>0.458984</td>
      <td>21.867188</td>
      <td>21.843750</td>
      <td>0.932374</td>
    </tr>
  </tbody>
</table>
</div>



**ANÁLISE**:
- Os dados já foram pre processados.

## 3. Exploração dos dados


```python
print(f"Valores únicos da label: {dataset.label.unique()}")
```

    Valores únicos da label: ['male' 'female']



```python
print("Divisão dos dados:")
print(f"Male: {round(dataset['label'].value_counts()[0]/len(dataset) * 100,2)}%.")
print(f"Female: {round(dataset['label'].value_counts()[1]/len(dataset) * 100,2)}%.")
```

    Divisão dos dados:
    Male: 50.0%.
    Female: 50.0%.


**ANÁLISE**:
- O dataset é balanceado e pre processado.

### 3.1. Existe diferência entre as vozes masculinas e femininas?


```python
male = dataset[dataset['label'] == 'male']
female = dataset[dataset['label'] == 'female']
```


```python
male.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meanfreq</th>
      <th>sd</th>
      <th>median</th>
      <th>Q25</th>
      <th>Q75</th>
      <th>IQR</th>
      <th>skew</th>
      <th>kurt</th>
      <th>sp.ent</th>
      <th>sfm</th>
      <th>mode</th>
      <th>centroid</th>
      <th>meanfun</th>
      <th>minfun</th>
      <th>maxfun</th>
      <th>meandom</th>
      <th>mindom</th>
      <th>maxdom</th>
      <th>dfrange</th>
      <th>modindx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.170813</td>
      <td>0.065110</td>
      <td>0.175299</td>
      <td>0.115562</td>
      <td>0.226346</td>
      <td>0.110784</td>
      <td>3.295460</td>
      <td>48.331698</td>
      <td>0.917188</td>
      <td>0.471670</td>
      <td>0.152022</td>
      <td>0.170813</td>
      <td>0.115872</td>
      <td>0.034175</td>
      <td>0.253836</td>
      <td>0.728877</td>
      <td>0.040307</td>
      <td>4.358447</td>
      <td>4.318139</td>
      <td>0.177430</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.026254</td>
      <td>0.009455</td>
      <td>0.037392</td>
      <td>0.031999</td>
      <td>0.024050</td>
      <td>0.020415</td>
      <td>5.135190</td>
      <td>163.115940</td>
      <td>0.028938</td>
      <td>0.150473</td>
      <td>0.084024</td>
      <td>0.026254</td>
      <td>0.017179</td>
      <td>0.015749</td>
      <td>0.036003</td>
      <td>0.445997</td>
      <td>0.049199</td>
      <td>3.000285</td>
      <td>3.000605</td>
      <td>0.130132</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.039363</td>
      <td>0.041747</td>
      <td>0.010975</td>
      <td>0.000240</td>
      <td>0.042946</td>
      <td>0.021841</td>
      <td>0.326033</td>
      <td>2.068455</td>
      <td>0.786650</td>
      <td>0.080963</td>
      <td>0.000000</td>
      <td>0.039363</td>
      <td>0.055565</td>
      <td>0.010953</td>
      <td>0.103093</td>
      <td>0.007812</td>
      <td>0.004883</td>
      <td>0.007812</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.155625</td>
      <td>0.058957</td>
      <td>0.149952</td>
      <td>0.101205</td>
      <td>0.211918</td>
      <td>0.100960</td>
      <td>1.461931</td>
      <td>5.003020</td>
      <td>0.899557</td>
      <td>0.363316</td>
      <td>0.098914</td>
      <td>0.155625</td>
      <td>0.104171</td>
      <td>0.017719</td>
      <td>0.246154</td>
      <td>0.399170</td>
      <td>0.007812</td>
      <td>1.759766</td>
      <td>1.751953</td>
      <td>0.099184</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.176343</td>
      <td>0.061781</td>
      <td>0.180612</td>
      <td>0.122315</td>
      <td>0.228117</td>
      <td>0.109940</td>
      <td>1.880420</td>
      <td>6.970088</td>
      <td>0.917309</td>
      <td>0.461636</td>
      <td>0.157557</td>
      <td>0.176343</td>
      <td>0.117254</td>
      <td>0.036166</td>
      <td>0.271186</td>
      <td>0.686687</td>
      <td>0.023438</td>
      <td>4.457031</td>
      <td>4.429688</td>
      <td>0.139904</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.190593</td>
      <td>0.070915</td>
      <td>0.202362</td>
      <td>0.136044</td>
      <td>0.244819</td>
      <td>0.119331</td>
      <td>2.645467</td>
      <td>12.282596</td>
      <td>0.936048</td>
      <td>0.576902</td>
      <td>0.228117</td>
      <td>0.190593</td>
      <td>0.128236</td>
      <td>0.047572</td>
      <td>0.277457</td>
      <td>1.032536</td>
      <td>0.031250</td>
      <td>6.035156</td>
      <td>6.000000</td>
      <td>0.212205</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.225582</td>
      <td>0.096030</td>
      <td>0.248840</td>
      <td>0.226740</td>
      <td>0.268924</td>
      <td>0.196168</td>
      <td>34.537488</td>
      <td>1271.353628</td>
      <td>0.981997</td>
      <td>0.831347</td>
      <td>0.280000</td>
      <td>0.225582</td>
      <td>0.179051</td>
      <td>0.121212</td>
      <td>0.279070</td>
      <td>2.805246</td>
      <td>0.458984</td>
      <td>21.867188</td>
      <td>21.843750</td>
      <td>0.932374</td>
    </tr>
  </tbody>
</table>
</div>




```python
female.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meanfreq</th>
      <th>sd</th>
      <th>median</th>
      <th>Q25</th>
      <th>Q75</th>
      <th>IQR</th>
      <th>skew</th>
      <th>kurt</th>
      <th>sp.ent</th>
      <th>sfm</th>
      <th>mode</th>
      <th>centroid</th>
      <th>meanfun</th>
      <th>minfun</th>
      <th>maxfun</th>
      <th>meandom</th>
      <th>mindom</th>
      <th>maxdom</th>
      <th>dfrange</th>
      <th>modindx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
      <td>1584.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.191000</td>
      <td>0.049142</td>
      <td>0.195942</td>
      <td>0.165349</td>
      <td>0.223184</td>
      <td>0.057834</td>
      <td>2.984875</td>
      <td>24.805224</td>
      <td>0.873066</td>
      <td>0.344763</td>
      <td>0.178541</td>
      <td>0.191000</td>
      <td>0.169742</td>
      <td>0.039429</td>
      <td>0.263848</td>
      <td>0.929544</td>
      <td>0.064987</td>
      <td>5.736107</td>
      <td>5.671120</td>
      <td>0.170073</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.029960</td>
      <td>0.018380</td>
      <td>0.032149</td>
      <td>0.049767</td>
      <td>0.023121</td>
      <td>0.042924</td>
      <td>3.091454</td>
      <td>97.669114</td>
      <td>0.047288</td>
      <td>0.179854</td>
      <td>0.067175</td>
      <td>0.029960</td>
      <td>0.018460</td>
      <td>0.021845</td>
      <td>0.021529</td>
      <td>0.576884</td>
      <td>0.072739</td>
      <td>3.854042</td>
      <td>3.856124</td>
      <td>0.107639</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.078847</td>
      <td>0.018363</td>
      <td>0.035114</td>
      <td>0.000229</td>
      <td>0.127637</td>
      <td>0.014558</td>
      <td>0.141735</td>
      <td>2.209673</td>
      <td>0.738651</td>
      <td>0.036876</td>
      <td>0.000000</td>
      <td>0.078847</td>
      <td>0.091912</td>
      <td>0.009775</td>
      <td>0.163934</td>
      <td>0.007812</td>
      <td>0.004883</td>
      <td>0.007812</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.177031</td>
      <td>0.034977</td>
      <td>0.181021</td>
      <td>0.157892</td>
      <td>0.206280</td>
      <td>0.031106</td>
      <td>1.962717</td>
      <td>6.764500</td>
      <td>0.839784</td>
      <td>0.208125</td>
      <td>0.168883</td>
      <td>0.177031</td>
      <td>0.157395</td>
      <td>0.019116</td>
      <td>0.258065</td>
      <td>0.450566</td>
      <td>0.023438</td>
      <td>2.560547</td>
      <td>2.435547</td>
      <td>0.101201</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.192732</td>
      <td>0.041965</td>
      <td>0.198226</td>
      <td>0.175373</td>
      <td>0.223744</td>
      <td>0.042689</td>
      <td>2.435808</td>
      <td>9.607635</td>
      <td>0.865861</td>
      <td>0.277228</td>
      <td>0.193670</td>
      <td>0.192732</td>
      <td>0.169408</td>
      <td>0.047013</td>
      <td>0.274286</td>
      <td>0.867405</td>
      <td>0.023438</td>
      <td>6.042969</td>
      <td>5.964844</td>
      <td>0.138995</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.211981</td>
      <td>0.060452</td>
      <td>0.216214</td>
      <td>0.195243</td>
      <td>0.241486</td>
      <td>0.061268</td>
      <td>3.086396</td>
      <td>14.448639</td>
      <td>0.908557</td>
      <td>0.478122</td>
      <td>0.218152</td>
      <td>0.211981</td>
      <td>0.181832</td>
      <td>0.048534</td>
      <td>0.277457</td>
      <td>1.338521</td>
      <td>0.140625</td>
      <td>8.607422</td>
      <td>8.531250</td>
      <td>0.201557</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.251124</td>
      <td>0.115273</td>
      <td>0.261224</td>
      <td>0.247347</td>
      <td>0.273469</td>
      <td>0.252225</td>
      <td>34.725453</td>
      <td>1309.612887</td>
      <td>0.978482</td>
      <td>0.842936</td>
      <td>0.280000</td>
      <td>0.251124</td>
      <td>0.237636</td>
      <td>0.204082</td>
      <td>0.279114</td>
      <td>2.957682</td>
      <td>0.449219</td>
      <td>21.796875</td>
      <td>21.773438</td>
      <td>0.857764</td>
    </tr>
  </tbody>
</table>
</div>



- 1º quartil em KHz


```python
labels = ['Female', 'Male']
sns.boxplot(x = dataset['label'], y = dataset['Q25'])
plt.xticks(range(2), labels)
plt.xlabel("Label")
plt.ylabel("Q25");
```


![png](imagens/output_22_0.png)


- Planicidade Espectral


```python
labels = ['Female', 'Male']
sns.boxplot(x = dataset['label'], y = dataset['sfm'])
plt.xticks(range(2), labels)
plt.xlabel("Label")
plt.ylabel("sfm");
```


![png](imagens/output_24_0.png)


- Centróide de Frequência 


```python
labels = ['Female', 'Male']
sns.boxplot(x = dataset['label'], y = dataset['centroid'])
plt.xticks(range(2), labels)
plt.xlabel("Label")
plt.ylabel("centroid");
```


![png](imagens/output_26_0.png)


- Média da Frequência dominante medida no sinal acústico


```python
labels = ['Female', 'Male']
sns.boxplot(x = dataset['label'], y = dataset['meanfun'])
plt.xticks(range(2), labels)
plt.xlabel("Label")
plt.ylabel("meanfun");
```


![png](imagens/output_28_0.png)


**ANÁLISE**:
- Podemos ver diferenças significantes entre as vozes masculinas e femininas utilizando BoxPlot

### 3.2. Transformando a variável target em binário


```python
dataset['label_cat'] = dataset['label'].apply(lambda x: 1 if x == 'male' else 0)
dataset.sample(5).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meanfreq</th>
      <th>sd</th>
      <th>median</th>
      <th>Q25</th>
      <th>Q75</th>
      <th>IQR</th>
      <th>skew</th>
      <th>kurt</th>
      <th>sp.ent</th>
      <th>sfm</th>
      <th>...</th>
      <th>meanfun</th>
      <th>minfun</th>
      <th>maxfun</th>
      <th>meandom</th>
      <th>mindom</th>
      <th>maxdom</th>
      <th>dfrange</th>
      <th>modindx</th>
      <th>label</th>
      <th>label_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1156</th>
      <td>0.180804</td>
      <td>0.070863</td>
      <td>0.175456</td>
      <td>0.128046</td>
      <td>0.243531</td>
      <td>0.115485</td>
      <td>2.056474</td>
      <td>7.450579</td>
      <td>0.913646</td>
      <td>0.489821</td>
      <td>...</td>
      <td>0.134689</td>
      <td>0.049130</td>
      <td>0.279070</td>
      <td>0.533078</td>
      <td>0.023438</td>
      <td>3.726562</td>
      <td>3.703125</td>
      <td>0.099156</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3166</th>
      <td>0.143659</td>
      <td>0.090628</td>
      <td>0.184976</td>
      <td>0.043508</td>
      <td>0.219943</td>
      <td>0.176435</td>
      <td>1.591065</td>
      <td>5.388298</td>
      <td>0.950436</td>
      <td>0.675470</td>
      <td>...</td>
      <td>0.172375</td>
      <td>0.034483</td>
      <td>0.250000</td>
      <td>0.791360</td>
      <td>0.007812</td>
      <td>3.593750</td>
      <td>3.585938</td>
      <td>0.311002</td>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1018</th>
      <td>0.197688</td>
      <td>0.056555</td>
      <td>0.212774</td>
      <td>0.146527</td>
      <td>0.246061</td>
      <td>0.099534</td>
      <td>1.748615</td>
      <td>6.888620</td>
      <td>0.910723</td>
      <td>0.286311</td>
      <td>...</td>
      <td>0.126432</td>
      <td>0.047198</td>
      <td>0.279070</td>
      <td>0.589453</td>
      <td>0.023438</td>
      <td>6.187500</td>
      <td>6.164062</td>
      <td>0.065883</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2149</th>
      <td>0.197306</td>
      <td>0.021782</td>
      <td>0.200416</td>
      <td>0.186343</td>
      <td>0.212062</td>
      <td>0.025719</td>
      <td>2.620309</td>
      <td>9.423510</td>
      <td>0.771301</td>
      <td>0.093359</td>
      <td>...</td>
      <td>0.185434</td>
      <td>0.038835</td>
      <td>0.228571</td>
      <td>0.200721</td>
      <td>0.164062</td>
      <td>0.242188</td>
      <td>0.078125</td>
      <td>0.183333</td>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1527</th>
      <td>0.121727</td>
      <td>0.077395</td>
      <td>0.108819</td>
      <td>0.053303</td>
      <td>0.200677</td>
      <td>0.147374</td>
      <td>3.539520</td>
      <td>33.242832</td>
      <td>0.953774</td>
      <td>0.672605</td>
      <td>...</td>
      <td>0.109024</td>
      <td>0.017957</td>
      <td>0.262295</td>
      <td>0.080631</td>
      <td>0.007812</td>
      <td>0.234375</td>
      <td>0.226562</td>
      <td>0.187821</td>
      <td>male</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



### 3.3. Correlação entre as variáveis

- Primeiramente, correlação não implica causalidade.


```python
plt.figure(figsize=(25, 20))
heat_map = sns.heatmap(dataset.corr(),annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show(heat_map)
```


![png](imagens/output_34_0.png)


### 3.4. Remover multicolinearidade?


```python
dataset.drop(columns=['label'], inplace = True)
```


```python
corr_matrix = dataset[:-1].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
dataset.drop(dataset[to_drop], axis=1, inplace=True)
```


```python
dataset.shape
```




    (3168, 16)



### 3.5. Features mais importantes

- Para analisar as features mais relevantes para detecção de Fraude, utilizamos dois algoritmos do método Ensemble para extrair as features.

#### 3.5.1. Separando feature e target.


```python
dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meanfreq</th>
      <th>sd</th>
      <th>Q75</th>
      <th>IQR</th>
      <th>skew</th>
      <th>sp.ent</th>
      <th>sfm</th>
      <th>mode</th>
      <th>meanfun</th>
      <th>minfun</th>
      <th>maxfun</th>
      <th>meandom</th>
      <th>mindom</th>
      <th>maxdom</th>
      <th>modindx</th>
      <th>label_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.059781</td>
      <td>0.064241</td>
      <td>0.090193</td>
      <td>0.075122</td>
      <td>12.863462</td>
      <td>0.893369</td>
      <td>0.491918</td>
      <td>0.000000</td>
      <td>0.084279</td>
      <td>0.015702</td>
      <td>0.275862</td>
      <td>0.007812</td>
      <td>0.007812</td>
      <td>0.007812</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.066009</td>
      <td>0.067310</td>
      <td>0.092666</td>
      <td>0.073252</td>
      <td>22.423285</td>
      <td>0.892193</td>
      <td>0.513724</td>
      <td>0.000000</td>
      <td>0.107937</td>
      <td>0.015826</td>
      <td>0.250000</td>
      <td>0.009014</td>
      <td>0.007812</td>
      <td>0.054688</td>
      <td>0.052632</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.077316</td>
      <td>0.083829</td>
      <td>0.131908</td>
      <td>0.123207</td>
      <td>30.757155</td>
      <td>0.846389</td>
      <td>0.478905</td>
      <td>0.000000</td>
      <td>0.098706</td>
      <td>0.015656</td>
      <td>0.271186</td>
      <td>0.007990</td>
      <td>0.007812</td>
      <td>0.015625</td>
      <td>0.046512</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.151228</td>
      <td>0.072111</td>
      <td>0.207955</td>
      <td>0.111374</td>
      <td>1.232831</td>
      <td>0.963322</td>
      <td>0.727232</td>
      <td>0.083878</td>
      <td>0.088965</td>
      <td>0.017798</td>
      <td>0.250000</td>
      <td>0.201497</td>
      <td>0.007812</td>
      <td>0.562500</td>
      <td>0.247119</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.135120</td>
      <td>0.079146</td>
      <td>0.206045</td>
      <td>0.127325</td>
      <td>1.101174</td>
      <td>0.971955</td>
      <td>0.783568</td>
      <td>0.104261</td>
      <td>0.106398</td>
      <td>0.016931</td>
      <td>0.266667</td>
      <td>0.712812</td>
      <td>0.007812</td>
      <td>5.484375</td>
      <td>0.208274</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = dataset.drop(columns=['label_cat'])
y = dataset['label_cat']
```

#### 3.5.2. Padronizando as features para analisar a importância real.


```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

#### 3.5.3. Criando os Algoritmos para Feature Importance


```python
random_forest_features = RandomForestClassifier(n_estimators=100, max_depth=15)
tree_features = DecisionTreeClassifier()
```

#### 3.5.4. Treinando os Algoritmos


```python
random_forest_features.fit(X, y)
tree_features.fit(X, y)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')



#### 3.5.5. Ordenando as features mais importantes


```python
random_forest_features_imp = pd.Series(random_forest_features.feature_importances_,
                                       index = dataset.columns[:-1])
random_forest_features_imp_order = random_forest_features_imp.sort_values()

tree_features_imp = pd.Series(tree_features.feature_importances_,
                              index = dataset.columns[:-1])
tree_features_imp_order = tree_features_imp.sort_values()
```

#### 3.5.6 Visualizando resultados das features mais importantes


```python
plt.figure(figsize=[16,6])
plt.subplot(1,2,1)
sns.barplot(x=random_forest_features_imp_order, y = random_forest_features_imp_order.index)
plt.xlabel("Features Importance")
plt.ylabel("Index")
plt.title("Feature Importance - Random Forest")
plt.legend()
plt.subplot(1,2,2)
sns.barplot(x=tree_features_imp_order, y = tree_features_imp_order.index)
plt.xlabel("Features Importance")
plt.ylabel("Features")
plt.title("Feature Importance- Decision Tree")
plt.legend()
plt.show()
```

    No handles with labels found to put in legend.
    No handles with labels found to put in legend.



![png](imagens/output_53_1.png)


**ANÁLISE**:
Entre os dois algoritmos, tivemos pouca divergência. Entretando, considerei o resultado das features mais importantes do algoritmo **Random Forest**, pois obteve considera uma maior quantidade de variáveis para tomada de decisão.

## 4. Processamento dos dados

### 4.1. Removendo as features com menor relevância


```python
list_features = []
for k, v in random_forest_features_imp_order[:5].items():
    list_features.append(k)
    
print(f'Features com menor importância: {list_features}')
```

    Features com menor importância: ['maxfun', 'modindx', 'mindom', 'minfun', 'meandom']



```python
df = dataset.drop(columns=list_features)
```


```python
df.shape
```




    (3168, 11)



### 5.1. Separando dados de treino e teste


```python
# df = dataset.copy()
```


```python
X = df.drop(columns=['label_cat'])
y = df['label_cat']
```


```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 6. ITS TIME!! Machine Learning

### 6.1 Métodos Ensemble
<img width="512" height="312" src="https://www.globalsoftwaresupport.com/wp-content/uploads/2018/02/ggff5544hh.png" />
Fonte: Global Software

#### 6.1.1. Bagging (Random Forest)

No Bagging os classificadores são treinados de forma independente por diferentes conjuntos de treinamento através do método de inicialização. Para construí-los é necessário montar k conjuntos de treinamento idênticos e replicar esses dados de treinamento de forma aleatória para construir k redes independentes por re-amostragem com reposição. Em seguida, deve-se agregar as k redes através de um método de combinação apropriada, tal como a maioria de votos (Maisa Aniceto, 2017).

#### 6.1.2. Boosting

No Boosting, de forma semelhante ao Bagging, cada classificador é treinado usando um conjunto de treinamento diferente. A principal diferença em relação ao Bagging é que os conjuntos de dados re-amostrados são construídos especificamente para gerar aprendizados complementares e a importância do voto é ponderado com base no desempenho de cada modelo, em vez da atribuição de mesmo peso para todos os votos. Essencialmente, esse procedimento permite aumentar o desempenho de um limiar arbitrário simplesmente adicionando learners mais fracos (Maisa Aniceto, 2017). Dada a utilidade desse achado, Boosting é considerado uma das descobertas mais significativas em aprendizado de máquina (LANTZ, 2013).

### 6.2. Comparando acurácia de 7 modelos de classificação utilizando validação cruzada
- Será selecionado 4 modelos para tunning de hiperparâmetros;
- Posteriormente, será selecionado o modelo que obter as melhores métricas.




```python
# Definindo os valores para o número de folds
num_folds = 3
seed = 7

modelos = []
modelos.append(("Logistic Regression", LogisticRegression()))
modelos.append(('Naive Bayes', GaussianNB()))
modelos.append(("Decision Tree", DecisionTreeClassifier()))
modelos.append(("Random Forest", RandomForestClassifier()))
modelos.append(("XGB Classifier", XGBClassifier()))
modelos.append(("Gradient Boosting Classifier", GradientBoostingClassifier()))
modelos.append(('SVM', SVC()))

# Avaliando cada modelo em um loop
resultados = []
nomes = []
print(f"{'Nome do Modelo':{30}}| {'Acurácia Média':{15}} | {'Desvio Padrão':{5}}")
for nome, modelo in modelos:
    kfold = KFold(n_splits = num_folds, shuffle = True)
    cv_results = cross_val_score(modelo, X_train, y_train, cv = kfold, scoring = 'accuracy')
    resultados.append(cv_results)
    nomes.append(nome)
    print(f"{nome:{30}}: {(np.around(cv_results.mean(), decimals=4))*100:{10}}%\
    {(np.around(cv_results.std(), decimals=4))*100:{10}}%")
```

    Nome do Modelo                | Acurácia Média  | Desvio Padrão
    Logistic Regression           :       96.8%          0.13%
    Naive Bayes                   :      93.69%          0.65%
    Decision Tree                 :      96.93%    0.16999999999999998%
    Random Forest                 : 97.92999999999999%    0.38999999999999996%
    XGB Classifier                :      97.97%          0.69%
    Gradient Boosting Classifier  : 97.78999999999999%    0.7799999999999999%
    SVM                           : 97.92999999999999%    0.27999999999999997%


**ANÁLISE**:
    - Selecionei os seguintes modelos:
        - SVM;
        - Random Forest;
        - XGB Classifier;
        - Logistic Regression.



### 6.3. Aplicando GridSearch para tunning e Validando com os dados de teste
- A métrica escolhida para otimizar os hiperparâmetros foi a Acurácia, tendo em vista que o dataset é balanceado.

#### 6.3.1. Logistic Regression


```python
# parâmetros da LogisticRegression
grid_rl = {"solver": ["liblinear", "lbfgs"],
           "C":[10, 25],
           "penalty" : ["l2"]}
# Criando modelo
logistic = LogisticRegression()
# Aplicando GridSearchCV
clf_lr = GridSearchCV(logistic, param_grid = grid_rl, cv=5, scoring = 'accuracy', verbose=0)
# Treinando modelo
clf_lr.fit(X_train, y_train)
# Fazendo predições
y_pred_logistic = clf_lr.predict(X_test)
# Avaliando modelo
print(classification_report(y_test, y_pred_logistic))
print("------------------------------------------------------")
print(confusion_matrix(y_test, y_pred_logistic))
print("------------------------------------------------------")
print("LogisticRegression accuracy: {}".format(accuracy_score(y_test, y_pred_logistic)))
```

                  precision    recall  f1-score   support
    
               0       0.97      0.97      0.97       452
               1       0.97      0.97      0.97       499
    
        accuracy                           0.97       951
       macro avg       0.97      0.97      0.97       951
    weighted avg       0.97      0.97      0.97       951
    
    ------------------------------------------------------
    [[437  15]
     [ 13 486]]
    ------------------------------------------------------
    LogisticRegression accuracy: 0.9705573080967402



```python
print(f'Melhores parâmetros para o algorítmo Logistic Regression: {clf_lr.best_params_}')
```

    Melhores parâmetros para o algorítmo Logistic Regression: {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}


#### 6.3.2 SVM


```python
# parâmetros da GradientBoostingClassifier
grid_svc = {'C': [0.1,1, 10, 100], 
            'gamma': [1,0.1,0.01,0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
           }

# Criando modelo
svm = SVC()
# Aplicando GridSearchCV
clf_svc = GridSearchCV(svm, param_grid = grid_svc, cv=5, scoring = 'accuracy', verbose=0, n_jobs=-1)
# Treinando modelo
clf_svc.fit(X_train, y_train)
# Fazendo predições
y_pred_svc = clf_svc.predict(X_test)
# Avaliando modelo
print(classification_report(y_test, y_pred_svc))
print("------------------------------------------------------")
print(confusion_matrix(y_test, y_pred_svc))
print("------------------------------------------------------")
print("SVM accuracy: {}".format(accuracy_score(y_test, y_pred_svc)))
```

                  precision    recall  f1-score   support
    
               0       0.97      0.98      0.98       452
               1       0.98      0.98      0.98       499
    
        accuracy                           0.98       951
       macro avg       0.98      0.98      0.98       951
    weighted avg       0.98      0.98      0.98       951
    
    ------------------------------------------------------
    [[444   8]
     [ 12 487]]
    ------------------------------------------------------
    SVM accuracy: 0.9789695057833859



```python
print(f'Melhores parâmetros para o algorítmo SVM: {clf_svc.best_params_}')
```

    Melhores parâmetros para o algorítmo SVM: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}


#### 6.3.3. XGB Classifier


```python
# parâmetros do XGBClassifier
grid_xgbc = { 'n_estimators': [50, 100, 200],
              'learning_rate': [0.01, 0.15],
              'max_depth': [15, 25, 50]}
# Criando o modelo
xgb = XGBClassifier()
# Aplicando GridSearchCV
clf_xgb = GridSearchCV(xgb, param_grid= grid_xgbc, cv=3,  scoring = 'accuracy', n_jobs = -1)
# Treinando modelo
clf_xgb.fit(X_train, y_train)
# Fazendo predições
y_pred_xgb = clf_xgb.predict(X_test)
# Avaliando modelo
print(classification_report(y_test, y_pred_xgb))
print("------------------------------------------------------")
print(confusion_matrix(y_test, y_pred_xgb))
print("------------------------------------------------------")
print("XGBClassifier accuracy: {}".format(accuracy_score(y_test, y_pred_xgb)))
```

                  precision    recall  f1-score   support
    
               0       0.97      0.99      0.98       452
               1       0.99      0.97      0.98       499
    
        accuracy                           0.98       951
       macro avg       0.98      0.98      0.98       951
    weighted avg       0.98      0.98      0.98       951
    
    ------------------------------------------------------
    [[446   6]
     [ 16 483]]
    ------------------------------------------------------
    XGBClassifier accuracy: 0.9768664563617245



```python
print(f'Melhores parâmetros para o algorítmo XGB: {clf_xgb.best_params_}')
```

    Melhores parâmetros para o algorítmo XGB: {'learning_rate': 0.15, 'max_depth': 15, 'n_estimators': 50}


#### 6.3.4 Random Forest Classifier


```python
# parâmetros da Random Forest
grid_rf = {
    "n_estimators" : [50, 100, 200],
    "max_depth": [15, 25, 50],
    "max_features": ['auto']
    }

# Criando modelo
rf = RandomForestClassifier()
# Aplicando GridSearchCV
clf_rf = GridSearchCV(rf, param_grid = grid_rf, cv=3, scoring = 'accuracy', n_jobs = -1)
# Treinando modelo
clf_rf.fit(X_train, y_train)
# Fazendo predições
y_pred_rf = clf_rf.predict(X_test)
# Avaliando modelo
print(classification_report(y_test, y_pred_rf))
print("------------------------------------------------------")
print(confusion_matrix(y_test, y_pred_rf))
print("------------------------------------------------------")
print("RandomForestClassifier accuracy: {}".format(accuracy_score(y_test, y_pred_rf)))
```

                  precision    recall  f1-score   support
    
               0       0.97      0.98      0.98       452
               1       0.99      0.97      0.98       499
    
        accuracy                           0.98       951
       macro avg       0.98      0.98      0.98       951
    weighted avg       0.98      0.98      0.98       951
    
    ------------------------------------------------------
    [[445   7]
     [ 14 485]]
    ------------------------------------------------------
    RandomForestClassifier accuracy: 0.9779179810725552



```python
print(confusion_matrix(y_test, y_pred_rf))

```

    [[445   7]
     [ 14 485]]



```python
print(f'Melhores parâmetros para o algorítmo Random Forest: {clf_rf.best_params_}')
```

    Melhores parâmetros para o algorítmo Random Forest: {'max_depth': 25, 'max_features': 'auto', 'n_estimators': 100}


## 7. Métricas

### 7.1. ROC AUC


```python
print(f"Métrica ROC AUC:\n\
{'Logistic Regression:':{30}} {(np.around(roc_auc_score(y_test, y_pred_logistic), decimals=4))*100}%\n\
{'SVM:':{30}} {(np.around(roc_auc_score(y_test, y_pred_svc), decimals=4))*100}%\n\
{'XGB Classifier:':{30}} {(np.around(roc_auc_score(y_test, y_pred_xgb), decimals=3))*100}%\n\
{'Random Forest Classifier:':{30}} {(np.around(roc_auc_score(y_test, y_pred_rf), decimals=3))*100}%")
```

    Métrica ROC AUC:
    Logistic Regression:           97.04%
    SVM:                           97.91%
    XGB Classifier:                97.7%
    Random Forest Classifier:      97.8%


**ANÁLISE**:
- Os quatro algoritmos tiverem resultados muito próximo.
- Dessa forma, não podemos selecionar nenhum algoritmo utilizando métrica ROC AUC.

### 7.2. Precision


```python
print(f"Métrica Precisão (Precision):\n\
{'Logistic Regression:':{30}} {(np.around(precision_score(y_test, y_pred_logistic), decimals=4))*100}%\n\
{'SVM:':{30}} {(np.around(precision_score(y_test, y_pred_svc), decimals=4))*100}%\n\
{'XGB Classifier:':{30}} {(np.around(precision_score(y_test, y_pred_xgb), decimals=3))*100}%\n\
{'Random Forest Classifier:':{30}} {(np.around(precision_score(y_test, y_pred_rf), decimals=4))*100}%")
```

    Métrica Precisão (Precision):
    Logistic Regression:           97.00999999999999%
    SVM:                           98.38%
    XGB Classifier:                98.8%
    Random Forest Classifier:      98.58%


**ANÁLISE**:
- Os quatro algoritmos tiveram resultados muito próximos.

### 7.3. Recall


```python
print(f"Métrica Revocação (Recall):\n\
{'Logistic Regression:':{30}} {(np.around(recall_score(y_test, y_pred_logistic), decimals=4))*100}%\n\
{'SVM:':{30}} {(np.around(recall_score(y_test, y_pred_svc), decimals=4))*100}%\n\
{'XGB Classifier:':{30}} {(np.around(recall_score(y_test, y_pred_xgb), decimals=3))*100}%\n\
{'Random Forest Classifier:':{30}} {(np.around(recall_score(y_test, y_pred_rf), decimals=3))*100}%")
```

    Métrica Revocação (Recall):
    Logistic Regression:           97.39%
    SVM:                           97.6%
    XGB Classifier:                96.8%
    Random Forest Classifier:      97.2%


**ANÁLISE**:
- Os quatro algorítmos tiveram resultados muito próximos

### 7.4. F1-Score


```python
print(f"Métrica F1-Score:\n\
{'Logistic Regression:':{30}} {(np.around(f1_score(y_test, y_pred_logistic), decimals=4))*100}%\n\
{'SVM:':{30}} {(np.around(f1_score(y_test, y_pred_svc), decimals=3))*100}%\n\
{'XGB Classifier:':{30}} {(np.around(f1_score(y_test, y_pred_xgb), decimals=3))*100}%\n\
{'Random Forest Classifier:':{30}} {(np.around(f1_score(y_test, y_pred_rf), decimals=4))*100}%")
```

    Métrica F1-Score:
    Logistic Regression:           97.2%
    SVM:                           98.0%
    XGB Classifier:                97.8%
    Random Forest Classifier:      97.88%


**ANÁLISE**:
- Os quatro algoritmos tiveram resultados muito próximos

### 7.5. Accuracy


```python
print(f"Métrica Accuracy:\n\
{'Logistic Regression:':{30}} {(np.around(accuracy_score(y_test, y_pred_logistic), decimals=4))*100}%\n\
{'SVM:':{30}} {(np.around(accuracy_score(y_test, y_pred_svc), decimals=4))*100}%\n\
{'XGB Classifier:':{30}} {(np.around(accuracy_score(y_test, y_pred_xgb), decimals=3))*100}%\n\
{'Random Forest Classifier:':{30}} {(np.around(accuracy_score(y_test, y_pred_rf), decimals=3))*100}%")
```

    Métrica Accuracy:
    Logistic Regression:           97.06%
    SVM:                           97.89999999999999%
    XGB Classifier:                97.7%
    Random Forest Classifier:      97.8%


## 8. Matriz de Confusão
- Vamos analisar a matriz de confusão do algoritmo que obteve melhor desempenho.


```python
labels = ['Female', 'Male']
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
ax.set_title("Matriz de Confusão - Random Forest", fontsize=20)
ax.set_ylabel('Classe Verdadeira', fontsize=15)
ax.set_xlabel('Classe Predita', fontsize=15)
```




    Text(0.5, 38.5, 'Classe Predita')




![png](imagens/output_104_1.png)


# CONCLUSÃO

**ANÁLISE**:

- A comparação dos resultados dos algorítmos são muito próximos. Podemos concluir que esse é um problema relativamente fácil de resolver. As vozes masculinas e femininas possuem características muito distintas.
- O SVM obteve a melhor performace utilizando a métrica de Acurácia (devido ao fato do dataset ser balanceado e pré processado).
- Entretanto, os algorítmos do método ensamble ficaram com valores muito próximos ao do SVM