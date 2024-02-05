#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import platform
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# In[39]:


# 그래프 한글 표시
from matplotlib import font_manager,rc
font_path="C:/Windows/Fonts/gulim.ttc"
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font',family=font_name)


# In[40]:


train= pd.read_csv("C:/Users/82107/Desktop/open/train.csv")
test= pd.read_csv("C:/Users/82107/Desktop/open/test.csv")
sample_submission= pd.read_csv("C:/Users/82107/Desktop/open/sample_submission.csv")


# In[41]:


train=train.drop('ID',axis=1)
train=train.drop('근로기간',axis=1)
train=train.drop('대출목적',axis=1)
train=train.drop('주택소유상태',axis=1)

train['대출기간']=train['대출기간'].str.replace('months','')
train['대출기간']=pd.to_numeric(train['대출기간'])

train=train.drop('총연체금액',axis=1)
train=train.drop('부채_대비_소득_비율',axis=1)
train=train.drop('최근_2년간_연체_횟수',axis=1)
train=train.drop('연체계좌수',axis=1)


# In[25]:


train.head(20)


# In[49]:


test=test.drop('ID',axis=1)
test=test.drop('근로기간',axis=1)
test=test.drop('대출목적',axis=1)
test=test.drop('주택소유상태',axis=1)

test['대출기간']=test['대출기간'].str.replace('months','')
test['대출기간']=pd.to_numeric(test['대출기간'])

test=test.drop('총연체금액',axis=1)
test=test.drop('부채_대비_소득_비율',axis=1)
test=test.drop('최근_2년간_연체_횟수',axis=1)
test=test.drop('연체계좌수',axis=1)


# In[42]:


label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
train['대출등급'] = train['대출등급'].map(label_mapping)


# In[43]:


X=train.drop('대출등급',axis=1)
y=train['대출등급']


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[45]:


dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# XGBoost 모델 훈련
params = {'objective': 'multi:softprob',
          'num_class': len(train['대출등급'].unique()),
          'max_depth': 10, 'learning_rate': 0.09997721832451799, 
          'subsample': 0.9187741607768906, 
          'colsample_bytree': 0.8630254262399467, 
          'lambda': 0.1, 
          'alpha': 0.1,
         'early_stopping_rounds': 50}

num_rounds = 100
model_log = xgb.train(params, dtrain, num_rounds)

pred_probs_log = model_log.predict(dtest)
pred_classes_log = np.argmax(pred_probs_log, axis=1)
accuracy = np.sum(pred_classes_log == y_test) / len(y_test)
print(f'Accuracy: {accuracy:.7f}')


# In[46]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# 모델 초기화
rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(y_train.unique()),
    max_depth=10,
    learning_rate=0.09997721832451799,
    subsample=0.9187741607768906,
    colsample_bytree=0.8630254262399467,
    reg_lambda=0.1,
    reg_alpha=0.1,
    n_estimators=100
)

# 보팅 앙상블 모델 초기화
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')

# 각 모델 훈련
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
voting_model.fit(X_train, y_train)

# 각 모델의 예측
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
voting_preds = voting_model.predict(X_test)

# 각 모델의 정확도 출력
print(f'Random Forest 정확도: {accuracy_score(y_test, rf_preds)}')
print(f'XGBoost 정확도: {accuracy_score(y_test, xgb_preds)}')
print(f'Voting 앙상블 모델 정확도: {accuracy_score(y_test, voting_preds)}')


# In[47]:


X_test.dtypes


# In[50]:


test.dtypes


# In[51]:


test_preds = voting_model.predict(test)
display(test_preds)


# In[59]:


column_names = ['대출등급']
df = pd.DataFrame(test_preds, columns=column_names)


# In[60]:


print(df)


# In[61]:


# 대출등급 숫자를 문자로 변환
df['대출등급'] = df['대출등급'].apply(lambda class_index: chr(65 + class_index))
print(df)


# In[64]:


predicted_class_column = df['대출등급']
sample_submission['대출등급'] = predicted_class_column
sample_submission.to_csv('newVersion0202.csv', index=False)


# In[70]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[71]:


rf_model = RandomForestClassifier()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Model Accuracy: {accuracy}')


# In[ ]:




