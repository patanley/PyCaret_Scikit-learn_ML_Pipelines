# Creating Machine Learning Pipeline with PyCaret and Scikit-learn

The scope of this project encompasses the entire machine learning pipeline, including tasks such as data ingestion, pre-processing, model training, hyper-parameter tuning, making predictions, and storing the resulting model for future use.


<div align="center"><a href="https://github.com/patanley/PyCaret_Scikit-learn_ML_Pipelines/blob/main/Images/ML1.png"><img src="https://github.com/patanley/PyCaret_Scikit-learn_ML_Pipelines/blob/main/Images/ML1.png" alt="IoT-to-Cloud (Nebula) network" style="width:70%;height:70%"/></a></div>


## 🎯 Introduction
The objective of this project is to develop a predictive model that can accurately forecast the selling price of cars on the Australian Automobile Market.

`Steps: create_model(), tune_model(), compare_models(), plot_model(), evaluate_model(), predict_model()`



## 🎯 Data
Data was source from the kaggle. Below is the kaggle source and the original source of the data

Access Data through [Original source](https://www.autotrader.com.au/for-sale), 
[Kaggle source](https://www.kaggle.com/datasets/nguyenthicamlai/cars-sold-in-australia)


## 🎯 Libraries & platform

<p align="left"> 
    <a href="https://www.microsoft.com/en-us/sql-server" target="_blank" rel="noreferrer"> <img src="https://www.svgrepo.com/show/303229/microsoft-sql-server-logo.svg" alt="mssql" width="60" height="60"/> </a> 
    <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="60" height="60"/> </a> 
    <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="60" height="60"/> </a>
    <a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://numpy.org/images/logo.svg" alt="Numpy" width="60" height="60"/> </a> 
    <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="60" height="60"/> </a> 
    <a href="https://pycaret.org/" target="_blank" rel="noreferrer"> <img src="https://www.gitbook.com/cdn-cgi/image/width=60,dpr=2,height=60,fit=contain,format=auto/https%3A%2F%2F1927305171-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FjAq5m5T7Qtz03TnB0Wve%252Ficon%252FKaPEfJEWupL6s9rsdFyF%252FGit%2520500-500_v5WhiteBG.png%3Falt%3Dmedia%26token%3D83cdee15-29e2-4fd3-8392-d1688963a063" alt="Pycaret" width="60" height="60"/> </a> 
 </p>

<br>



## 🎯 Notebooks

> Car Price Prediction with PyCaret3 [GitHub](https://github.com/patanley/PyCaret_Scikit-learn_ML_Pipelines/blob/main/Car%20Price%20Prediction%20with%20PyCaret3.ipynb)

> Prediction with Scikit-Learn [GitHub](https://github.com/patanley/PyCaret_Scikit-learn_ML_Pipelines/blob/main/Prediction%20with%20Scikit-Learn.ipynb)

> Access Data through [Google colab](https://colab.research.google.com/drive/1bjh0T_9ux-hA6zQBctaR5ycPlf5_yy1y#scrollTo=ffb2029b)


```# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns

from pycaret.regression import *
# from ydata_profiling import ProfileReport

pd.set_option('display.max_columns', None)

# Load data
# Cars-sold-in-Australia
data = pd.read_csv('./cars.csv')
print(data.shape)
data.head()

data.describe()

import plotly.express as px 
fig = px.box(data, y="Price")
fig.show()

data_outliers = data[data['Price']>94000]
data_outliers.shape

data= data.drop(data_outliers.index)
print(f'Data for model: {data.shape}, \nData for unseen predicitions: {data_outliers.shape}')
data_outliers.to_csv('./cars_outliers.csv', index=False)

data_unseen = data.sample(frac=0.1)
data= data.drop(data_unseen.index)
print(f'Data for model: {data.shape}, \nData for unseen predicitions: {data_unseen.shape}')
data_unseen.to_csv('./cars_unseen.csv', index=False)

%%time
from pycaret.regression import *
s = setup(data, target = 'Price', ignore_features=['Model','Variant', 'Series'])

transform = s.dataset_transformed
transform.to_csv('./cars_transformed.csv', index=False)

%%time
models()

%%time
et = create_model('et')
t_et = tune_model(et)
f_et = finalize_model(et)
f_et

%%time
gbr = create_model('gbr')
t_gbr = tune_model(gbr)
f_gbr = finalize_model(t_gbr)
f_gbr

%%time
plot_model(et, plot='feature')

%%time
plot_model(,plot='residuals')

%%time
plot_model(et, plot='error')

pred = predict_model(f_et,data=data_unseen)

%%time
pred = pred.loc[:,['Price','prediction_label']]
pred['Percent Diff'] = (pred['Price']-pred['prediction_label'])/pred['Price']
pred

pred.describe().T

save_model(et, 'car_FinalModel')
```



## License

Licensed under GPL-3.0 license

© patanley 2023 ca

**Follow me** at<br />
[![Follow me on twitter](https://img.shields.io/twitter/follow/patanley.svg?style=social)](https://twitter.com/home) 
[![Follow me on LinkedIn](https://img.shields.io/badge/LinkedIn-Patanley-blue?style=flat&logo=linkedin&logoColor=b0c0c0&labelColor=363D44)](https://www.linkedin.com/feed/)

**Share** the project link with your network on social media.
