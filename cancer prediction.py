#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install dtreeviz')


# In[28]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install shape')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb

import matplotlib.gridspec as gridspec

from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, r2_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

        
import warnings
warnings.filterwarnings("ignore")


# In[4]:


df = pd.read_csv("cancer patient data sets.csv")
df


# In[5]:


df.drop(columns=['index', 'Patient Id'], axis=1, inplace=True)
df


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


print('Cancer Level: ', df['Level'].unique())
df["Level"].replace({'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)
print('Cancer Levels: ', df['Level'].unique())


# In[9]:


df.info()


# # Разведовательный анализ(EDA)

# In[10]:


plt.figure(figsize=(10,4))
plt.boxplot(df['Age'], vert=False, patch_artist=True, 
            boxprops=dict(facecolor='skyblue', linewidth=2),
            whiskerprops=dict(color='green',linewidth=3), 
            medianprops=dict(color='red',linewidth=2)
           )
plt.title('Распределение по возрасту')
plt.xticks(np.arange(10, max(df['Age'])+1, 3))
plt.show()


# In[11]:


import statsmodels.api as sm

data = data = df['Age'].dropna()
 
#QQ график
sm.qqplot(data, line='s')
plt.title("Распределение по столбцу'Age'")
plt.show()


# In[12]:


df_corr = df.corr()
df_corr


# In[13]:


plt.title("Correlation Matrix")
sns.heatmap(df_corr, cmap='viridis')


# In[14]:


# Heatmap

print('\n')
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True, cmap=plt.cm.PuBu)
plt.show()
print('\n')


# In[15]:


sea = sns.FacetGrid(df, col = "Level", height = 5)
sea.map(sns.distplot, "Age", color="blue")


# In[16]:


sea = sns.FacetGrid(df, col = "Level", height = 5)
sea.map(sns.distplot, "Gender", color="purple")


# In[17]:


plt.figure(figsize = (15, 55))

for i in range(24):
    plt.subplot(16, 2, i+1)
    sns.distplot(df.iloc[:, i], color = 'green')
    plt.grid()


# In[18]:


plt.figure(figsize = (15,7))
colors = ['red', 'yellow', 'green']
plt.title("Шанс на рак легких ")
plt.pie(df['Level'].value_counts(), explode = (0.1, 0.02, 0.02), labels = ['High', 'Medium', 'Low'], autopct = "%1.2f%%", shadow = True, colors = colors)
plt.legend(title = "Шанс на рак легких разной стадии", loc = "lower left")


# In[19]:


sns.displot(df['Level'], kde=True, color = 'blue')


# In[20]:


dfviz = df.copy()


# In[21]:


y = df.pop('Level')


# In[22]:


x = df


# # Обучаем и тренируем

# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[25]:


results = pd.DataFrame(columns = ['model', 'f1_train', 'f1_test', 'r2_train', 'r2_test'])


# In[26]:


print('X train shape: ', x_train.shape)
print('Y train shape: ', y_train.shape)
print('\nTest Shape\n')
print('X test shape: ', x_test.shape)
print('Y test shape: ', y_test.shape)


# In[27]:


def performTest(y_pred):
    print("Test Data Metrics:")
    print("Precision : ", precision_score(y_test, y_pred, average = 'micro'))
    print("Recall : ", recall_score(y_test, y_pred, average = 'micro'))
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred, average = 'micro'))
    print("R2 Score : ", r2_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    print("\n")
    print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
    print(classification_report(y_test, y_pred))
    print("**"*27+"\n")
    
    cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Low', 'Medium', 'High'])
    
    cm.plot( cmap='plasma', ax=None, xticks_rotation='horizontal')
    print("\n")


# In[28]:


def performTrain(y_pred_train):
    print("Train Data Metrics:")
    print("Precision : ", precision_score(y_train, y_pred_train, average='micro'))
    print("Recall : ", recall_score(y_train, y_pred_train, average='micro'))
    print("Accuracy : ", accuracy_score(y_train, y_pred_train))
    print("F1 Score : ", f1_score(y_train, y_pred_train, average='micro'))
    print("R2 Score : ", r2_score(y_train, y_pred_train))

    print("\n")


# # Садим случайный лес))

# In[29]:


from sklearn.model_selection import GridSearchCV
param_RF1 = {
    'n_estimators': 50, 
    'max_depth': 3,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_samples': 0.8,
    'criterion': 'gini'    
}
param_RF = {
    'n_estimators': [50, 100, 150], 
    'max_depth': [2],
    'criterion':['gini'],
    'min_samples_split': [2, 3],  
    'min_samples_leaf': [2, 3],
    'random_state' : [42],
    'max_samples': [0.4]
    
}


# In[36]:


model_tuning = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_RF, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_rf = RandomForestClassifier(**best_params)
model_rf.fit(x_train, y_train)


# In[30]:


#Тренируем
train_predictions = model_rf.predict(x_train)
r2_train = r2_score(y_train, train_predictions)
f1_train = f1_score(y_train, train_predictions, average = 'micro')

#Тестируем
test_predictions = model_rf.predict(x_test)
r2_test = r2_score(y_test, test_predictions)
f1_test = f1_score(y_test, test_predictions, average = 'micro')

#Итог
performTrain(train_predictions)
performTest(test_predictions)

#Сохраняем
results.loc[0,'model'] = 'RandomForest Classifier'
results.loc[0,'f1_train'] = f1_train
results.loc[0,'f1_test'] = f1_test
results.loc[0,'r2_train'] = r2_train
results.loc[0,'r2_test'] = r2_test
results.loc[0,'short'] = 'RF'
score_model_rf = model_rf.score(x_test, y_test)


# # AdaBoost

# In[43]:


param_ADA = {
    'n_estimators': [50, 100, 200], 
    'learning_rate': [0.01, 0.1, 1],
    'random_state' : [42],
    'algorithm': ['SAMME.R'],   
}


# In[45]:


model_tuning = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_ADA, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_ada = AdaBoostClassifier(**best_params)
model_ada.fit(x_train, y_train)


# In[46]:


#Трениурем
train_predictions = model_ada.predict(x_train)
r2_train = r2_score(y_train, train_predictions)
f1_train = f1_score(y_train, train_predictions, average = 'micro')

#Тест
test_predictions = model_ada.predict(x_test)
r2_test = r2_score(y_test, test_predictions)
f1_test = f1_score(y_test, test_predictions, average = 'micro')

#Итог
performTrain(train_predictions)
performTest(test_predictions)

#Сохраняем
results.loc[1,'model'] = 'ADABoost Classifier'
results.loc[1,'f1_train'] = f1_train
results.loc[1,'f1_test'] = f1_test
results.loc[1,'r2_train'] = r2_train
results.loc[1,'r2_test'] = r2_test
results.loc[1,'short'] = 'ADA'
score_model_ada = model_ada.score(x_test, y_test)


# In[64]:


params_etc ={'n_estimators':[200, 300, 500],
            'max_depth': [3, 4, 5],
            'min_samples_split':  [2, 3],
            'min_samples_leaf':  [2, 3],
            'max_features': ['sqrt'],
            'ccp_alpha': [0.1],
            'random_state' : [42]}


# In[65]:


model_tuning = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid=params_etc, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_etc = ExtraTreesClassifier(**best_params)
model_etc.fit(x_train, y_train)


# In[66]:


# Train
train_predictions = model_etc.predict(x_train)
r2_train = r2_score(y_train, train_predictions)
f1_train = f1_score(y_train, train_predictions, average = 'micro')

# Test
test_predictions = model_etc.predict(x_test)
r2_test = r2_score(y_test, test_predictions)
f1_test = f1_score(y_test, test_predictions, average = 'micro')

#Result
performTrain(train_predictions)
performTest(test_predictions)

# Save
results.loc[2,'model'] = 'ExtraTrees Classifier'
results.loc[2,'f1_train'] = f1_train
results.loc[2,'f1_test'] = f1_test
results.loc[2,'r2_train'] = r2_train
results.loc[2,'r2_test'] = r2_test
results.loc[2,'short'] = 'ExT'
score_model_etc = model_etc.score(x_test, y_test)


# # Дерево решений

# In[37]:


params_DT2 ={'max_depth': [3],
            'min_samples_split':  [2],
            'min_samples_leaf':  [1],
            'max_features': ['sqrt'],
            }

params_DT1 ={'max_depth': [1, 2, 3],
            'min_samples_split': [1, 2, 3],
            'min_samples_leaf': [0, 1, 2],
            'max_features': ['sqrt', 'log2', 'auto'],
            'max_leaf_nodes': [ 5, 10, 15, 17],
            'random_state' : [42],
            'min_impurity_decrease': [0, 0.001, 0.01],
            'ccp_alpha': [0.0, 0.001, 0.01, 0.1]
           }

params_DT ={'max_depth': [3, 4],
            'min_samples_split': [1, 2, 3],
            'min_samples_leaf': [0, 1, 2],
            'max_features': ['sqrt', 'log2', 'auto'],
            'max_leaf_nodes': [ 5, 10, 15, 17],
            'random_state' : [42]}


# In[68]:


model_tuning = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params_DT, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_dt = DecisionTreeClassifier(**best_params)
model_dt.fit(x_train, y_train)


# In[38]:


train_predictions = model_etc.predict(x_train)
r2_train = r2_score(y_train, train_predictions)
f1_train = f1_score(y_train, train_predictions, average = 'micro')

# Test
test_predictions = model_etc.predict(x_test)
r2_test = r2_score(y_test, test_predictions)
f1_test = f1_score(y_test, test_predictions, average = 'micro')

#Result
performTrain(train_predictions)
performTest(test_predictions)

# Save
results.loc[2,'model'] = 'ExtraTrees Classifier'
results.loc[2,'f1_train'] = f1_train
results.loc[2,'f1_test'] = f1_test
results.loc[2,'r2_train'] = r2_train
results.loc[2,'r2_test'] = r2_test
results.loc[2,'short'] = 'ExT'
score_model_etc = model_etc.score(x_test, y_test)


# # Визуализация дерева решений

# In[44]:


import sys
get_ipython().system('{sys.executable} -m pip install dfviz')


# In[39]:


import dfviz
feature_names = dfviz.columns[0:23]
viz = dfviz.copy()
viz["Level"]=viz["Level"].values.astype(str)
print(viz.dtypes)
target_names = viz['Level'].unique().tolist()


# Диаграмма дерева решений

# In[47]:


from sklearn.tree import plot_tree

plt.figure(figsize=(15, 10))
plot_tree(model_dt, feature_names = feature_names, class_names = target_names, filled = True, rounded = False)

plt.savefig('tree_visualization.png')


# In[48]:


import dtreeviz

viz_model = dtreeviz.model(model_dt,
                           X_train=x_train, y_train=y_train,
                           feature_names=feature_names,
                           target_name='Lung Cancer',
                           class_names=['Low', 'Medium', 'High'])

v = viz_model.view()
v.save("Lung Cancer.svg")


# In[6]:


viz_model.view()


# # XGBoost

# In[49]:


params_XGB1 ={'n_estimators': 366,
                  'num_leaves': 10,
                  'max_depth': 9,
                 'lambda': 0.1444861779926268,
                  'subsample': 0.01,
                  'alpha': 2.603602561261043e-06,
                   'colsample_bytree': 1.0  }

params_XGB ={'n_estimators': [100, 200, 300],
             'num_leaves': [2, 5, 7, 10],
             'max_depth': [2, 3, 5],         
             'subsample': [0.01],
             'learning_rate': [0.01],
             'objective': ['multi:softmax'],
             'random_state' : [42],
             'num_class': [3]}


# In[50]:


model_tuning = GridSearchCV(estimator=XGBClassifier(), param_grid=params_XGB, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_xgb = XGBClassifier(**best_params)
model_xgb.fit(x_train, y_train)


# In[9]:


# Обучаем
train_predictions = model_xgb.predict(x_train)
r2_train = r2_score(y_train, train_predictions)
f1_train = f1_score(y_train, train_predictions, average = 'micro')

# Тестируем
test_predictions = model_xgb.predict(x_test)
r2_test = r2_score(y_test, test_predictions)
f1_test = f1_score(y_test, test_predictions, average = 'micro')

# Итог
performTrain(train_predictions)
performTest(test_predictions)

# Сохраняем
results.loc[4,'model'] = 'XGB Classifier'
results.loc[4,'f1_train'] = f1_train
results.loc[4,'f1_test'] = f1_test
results.loc[4,'r2_train'] = r2_train
results.loc[4,'r2_test'] = r2_test
results.loc[4,'short'] = 'XGB'
score_model_xgb = model_xgb.score(x_test, y_test)


# # MLP

# In[10]:


params_mlp1 ={'hidden_layer_sizes': 100,
            'random_state': 2,
            'alpha': 0.001,
            'activation': 'relu',
            'learning_rate_init': 0.001,
            'max_iter': 100,
             'batch_size': 32,
             'early_stopping':True,
             'validation_fraction': 0.1,
             'tol': 1e-4
            }

params_mlp2 ={'hidden_layer_sizes': [50, 75, 100],
             'random_state': [2],
             'alpha': [0.01, 0.001],
             'activation': ['relu', 'tanh'],
             'learning_rate_init': [0.001, 0.01],
             'max_iter': [100, 200, 300],
             'solver': ['adam', 'lbfgs'],
             'early_stopping': [True],
             'validation_fraction': [0.1, 0.2],
             
            }

params_mlp ={'hidden_layer_sizes': [50, 75, 100],
             'learning_rate_init': [0.001],
             'max_iter': [100, 200, 300],
             'solver': ['adam'],
             'early_stopping': [True],
             'random_state' : [42]}


# In[11]:


model_tuning = GridSearchCV(estimator=MLPClassifier(), param_grid=params_mlp, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_mlp = MLPClassifier(**best_params)
model_mlp.fit(x_train, y_train)


# In[12]:


# Обучаем
train_predictions = model_mlp.predict(x_train)
r2_train = r2_score(y_train, train_predictions)
f1_train = f1_score(y_train, train_predictions, average = 'micro')

# Тест
test_predictions = model_mlp.predict(x_test)
r2_test = r2_score(y_test, test_predictions)
f1_test = f1_score(y_test, test_predictions, average = 'micro')

# Результат
performTrain(train_predictions)
performTest(test_predictions)

# Сохраняем
results.loc[5,'model'] = 'MLP Classifier'
results.loc[5,'f1_train'] = f1_train
results.loc[5,'f1_test'] = f1_test
results.loc[5,'r2_train'] = r2_train
results.loc[5,'r2_test'] = r2_test
results.loc[5,'short'] = 'MLP'
score_model_mlp = model_mlp.score(x_test, y_test)


# # Анализ результатов

# In[13]:


colors = ['blue', 'green', 'red', 'purple', 'orange', 'c']
colors1 = ['c', 'lime', 'pink', 'magenta', 'yellow', 'cyan']
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(results['short']))
bar1 = plt.bar(index, results['f1_train'], bar_width,
                   label='f1_train ', color='red')
bar2 = plt.bar([i + bar_width for i in index], results['f1_test'], bar_width,
                label='f1_test', color='cyan')
test (light color)', s=500, color=colors1)
plt.xlabel('Модели')
plt.ylabel('f1_score')
plt.title('Результаты оценки f1_train и f1_test по моделями')
plt.xticks([i + bar_width/2 for i in index], results['short'])
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
plt.show()


#  Важность функции для случайной классификации лесов

# In[14]:


feature_importances_model_rf = pd.DataFrame(x_train.columns)
feature_importances_model_rf.columns = ['feature']
feature_importances_model_rf["score_model_rf"] = pd.Series(model_rf.feature_importances_)
feature_importances_model_rf.sort_values(by='score_model_rf', ascending=False)


# In[15]:


importances = model_rf.feature_importances_
feature_names = df.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='lime')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


# In[16]:


explainer = shap.Explainer(model_rf, x)
shap_values = explainer.shap_values(x)
shap.summary_plot(shap_values, x)


# In[ ]:




