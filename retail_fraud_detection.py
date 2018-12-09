#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import os
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


os.getcwd()


# In[3]:


#Change directory to where dataset is
os.chdir("X:\\Datasets")


# In[4]:


transaction_data=pd.read_excel('Fraud_Analytics_Dataset.xlsx',0)
customer_data=pd.read_excel('Fraud_Analytics_Dataset.xlsx',1)
sales_data=pd.read_excel('Fraud_Analytics_Dataset.xlsx',2)


# In[5]:


display(transaction_data.head())
display(customer_data.head())
display(sales_data.head())


# In[6]:


'''
Joining all the 3 sheets
'''
#caller.join(other.set_index('key'), on='key')
f1=sales_data.join(transaction_data.set_index('Transaction_Id'),on='Transaction_Id')
original_data = f1.join(customer_data.set_index('CUSTOMER_ID'),on='Customer_Id')


# In[298]:


#Preprocessing - > Wrangling + Cleaning

# Correcting column names for ease of usage
original_data.columns = original_data.columns.str.replace(' ','')
original_data.columns = original_data.columns.str.replace('.','')

#Making object type data into categorical variables for now 
# for col_name in master_data.columns:
#     if(master_data[col_name].dtype == 'object'):
#         master_data[col_name]= master_data[col_name].astype('category')
#         #master_data[col_name] = transaction_data[col_name].cat.codes
        
original_data.head(10)


# In[274]:


#Please don't touch this piece of code, otherwise the destruction caused would be equivalent to World War 3
master_data = original_data


# In[275]:


master_data.info()


# ---
# Through a basic Excel Sheet analysis, we have are seeing a lot of redundant variables as well as duplicate ones. 
# 
# 1. Columns representing unique values such as Transaction ID's or Customer ID's can be removed directly.
# 
# 2. Multiple variables such as Loyalty ID, Membership related Info, etc. are representing the same information, leading to **Multicollinearity in the data**. Such variables can be romoved or clubbed together to create a single informative column.
# 
# 3. Some variables, which are not relevant within the aim of this project, such as Country, can be removed, as they contains only a single value. If and when this data is extrapolated for other countries, it may prove to be useful.
# 
# 4. Fields with large number of missing values, as we can't assume the values in such cases for better chances of accuracy.
# 
# We will not be using such columns, and a single list of columns being used for model building will be informed.
# 
# ---

# In[277]:


# feature_list = ['No_Of_Items','Shipment_Flg','Return_Flag','FirstTimeCustomers',
#                 'UnusualLocation','BiggerthanAverageOrders','MultipleShippingAddresses','ShippingandBillingAddressarenotthesame',
#                 'SeveralCardsusedfromtheSameIpAddress','PaymentInformationtypedwithCapitalletters', 'FastShipping',
#                 'ManyTransactionsinashorttimeperiod','MEMBERSHIP_TYPE__ID','CUSTOMER_STATUS_ID','customer_type_id','AGE',
#                 'INCOME_RANGE', 'PROFESSION', 'Preferred_Comm_Sub_Channel','credit_risk_rating', 'Residence_Province', 
#                 'Mode_Of_Payment','Sale_Dt']

# feature_list = ['Shipment_Flg','Return_Flag','FirstTimeCustomers', 'UnusualLocation','BiggerthanAverageOrders',
#                 'MultipleShippingAddresses','ShippingandBillingAddressarenotthesame',
#                 'SeveralCardsusedfromtheSameIpAddress', 'FastShipping',
#                 'ManyTransactionsinashorttimeperiod','MEMBERSHIP_TYPE__ID',
#                 'INCOME_RANGE', 'PROFESSION','credit_risk_rating', 'Lastmodified_Dt']

# feature_list = ['UnusualLocation','BiggerthanAverageOrders','MultipleShippingAddresses','ShippingandBillingAddressarenotthesame',
#                 'SeveralCardsusedfromtheSameIpAddress', 'FastShipping',
#                 'ManyTransactionsinashorttimeperiod']
feature_list = ['FastShipping','credit_risk_score','age_range','Interest_2','INFLUENCE',
 'Interests_1', 'INCOME_RANGE',
 'LOYALITY_FLG', 'NO_TIMES_DELINQUENT_in_365_days',
 'City_Sc', 'credit_risk_rating',
 'CUSTOMER_STATUS_SC',
 'fbck_user_FLG', 'NO_OF_DEPENDANTS',
 'No_Of_Items', 'Shipment_Flg',
 'Order_Item_Paidprice_Gross_Amt_',
 'REGION', 'fbck_following_CNT',
 'Order_Item_Listprice_Gross_Amt_','twt_follower_CNT','BiggerthanAverageOrders',
 'No_Of_Items_After_Return',
 'Return_Flag','MultipleShippingAddresses',
 'SeveralCardsusedfromtheSameIpAddress',
 'UnusualLocation','ShippingandBillingAddressarenotthesame',
 'Order_Returnpaid_Gross_Amt',
 'Return_Amt','PaymentInformationtypedwithCapitalletters',
 'Customer_Type','Lastmodified_Dt',
 'FirstTimeCustomers','Discount_Percentage',
 'customer_type_sc','Quantity_Returned',
 'Discount_Amt','Cancel_Date',
 'ManyTransactionsinashorttimeperiod',
 'Mode_Of_Payment','Return_Dt']
target = 'Fraud'


# In[278]:


target = 'Fraud'
label_data = master_data[target].astype('category').cat.codes
master_data.drop('Fraud', axis = 1)
#feature_list = master_data.columns
print(label_data.head(5))

master_data = master_data[feature_list]
master_data.info()


# In[279]:


obj_type_variables = [column for column in master_data.columns if master_data[column].dtype in ['object', 'datetime64[ns]']]
print(obj_type_variables)


# In[280]:


#Making object type data into categorical variables for now
for column in obj_type_variables:
    master_data[column] = master_data[column].astype('category')


# In[281]:


master_data.info()


# In[282]:


#Converting categorical into numerical values
master_data[obj_type_variables] = master_data[obj_type_variables].apply(lambda column: column.cat.codes)


# In[283]:


master_data.info()


# In[284]:


#list of columns with null values 
list_of_clm_null_value= master_data.columns[master_data.isna().any()].tolist()
print(list_of_clm_null_value)


# In[297]:


null_columns=master_data.columns[master_data.isnull().any()]
count = master_data[null_columns].isnull().sum()
print(count)


# In[285]:


#replacing the nan value with -1 after EDA
for i in list_of_clm_null_value:
    master_data[i].fillna(-1, inplace = True) 


# In[286]:


#Converting numerical based categories into category dtype
for column in obj_type_variables:
    master_data[column] = master_data[column].astype('category')
master_data.info()


# ---
# Model Building
# ---
# ---

# In[287]:


#Stratified Shuffle Split by 70:30 ratio + SelectKBest
retail_data = master_data
skb = SelectKBest(k=8)
retail_data = skb.fit_transform(retail_data, label_data)
features_train, features_test, labels_train, labels_test = train_test_split(retail_data, label_data, test_size=0.3, random_state=42)


# In[288]:


unsorted_list = zip(feature_list, skb.scores_)

sorted_features = sorted(unsorted_list, key=lambda x: x[1], reverse=True)
print(len(sorted_features))
print("Feature Scores:\n")
pprint(sorted_features)


# In[289]:


print(len(labels_test[labels_test==1]))


# In[290]:


gnb = GaussianNB()
y_pred_gnb = gnb.fit(features_train, labels_train).predict(features_test)

DT = DecisionTreeClassifier()
y_pred_DT = DT.fit(features_train, labels_train).predict(features_test)

ada = AdaBoostClassifier()
y_pred_ada = ada.fit(features_train, labels_train).predict(features_test)

logr = LogisticRegression()
y_pred_logr = logr.fit(features_train, labels_train).predict(features_test)


# In[291]:


print("GNB:", accuracy_score(labels_test, y_pred_gnb))
print("DT:", accuracy_score(labels_test, y_pred_DT))
print("ADA:", accuracy_score(labels_test, y_pred_ada))
print("Logr:", accuracy_score(labels_test, y_pred_logr))



# #print precision_recall_fscore_support(predicted_variable, y_pred, average='macro')
# print(precision_recall_fscore_support(predicted_variable, y_pred, average='micro'))
# print(precision_recall_fscore_
#support(predicted_variable, y_pred, average=None))
# #print precision_recall_fscore_support(predicted_variable, y_pred, average='weighted')

# print(math.sqrt(mean_squared_error(predicted_variable, y_pred)))


# In[292]:


print("GNB:", classification_report(labels_test, y_pred_gnb))
print("DT:", classification_report(labels_test, y_pred_DT))
print("ADA:", classification_report(labels_test, y_pred_ada))
print("Logr:", classification_report(labels_test, y_pred_logr))


# ---
# **Tuning the classifiers**
# 
# ---

# In[293]:


def tune_NB() :
    print("------------------ Using Naive Bayes --------------------")
    nb_clf = GaussianNB()
    param_grid = {}             # No parameters for tuning

    return nb_clf, param_grid

# Decision Tree Classifier
def tune_DT():
    print("------------------ Using Decision Trees --------------------")
    clf = DecisionTreeClassifier()
    param_grid = {
        'clf__criterion': ['entropy', 'gini'],
        'clf__splitter': ['best', 'random'],
        'clf__random_state': [42],
        'clf__min_samples_split': [2, 4, 5, 6, 7, 8]
    }

    return clf, param_grid

# AdaBoost Classifier
def tune_ADB():
    print("------------------ Using AdaBoost Ensemble --------------------")
    clf = AdaBoostClassifier()
    param_grid = {
        'clf__algorithm' : ['SAMME', 'SAMME.R'],
        'clf__learning_rate': [1, 2],
        'clf__random_state': [42],
        'clf__n_estimators' : [20, 50, 65, 80, 100]
    }

    return clf, param_grid

# Logistic Regression Classifier
def tune_LogR():
    print("------------------ Using Logistic Regression --------------------")
    clf = LogisticRegression()
    param_grid = {
        'clf__max_iter' : [100, 500, 1000],
        'clf__penalty': ['l1', 'l2'],
        'clf__tol': [0.0001, 0.0005, 0.001, 0.005],
        'clf__C' : [1.0, 1.5, 0.5, 2.0]
    }

    return clf, param_grid

# Random Forest Classifier
def tune_RandomF():
    print("------------------ Using Random Forest --------------------")
    clf = RandomForestClassifier()
    param_grid = {
        'clf__criterion': ['entropy', 'gini'],
        'clf__min_samples_split': [2, 4, 5, 6, 7, 8, 9, 10],
        'clf__random_state': [42],
        'clf__n_estimators' : [20, 40, 60, 80, 100, 200],
        'clf__max_features' : ['auto', 'log2', None] 
    }

    return clf, param_grid


# In[294]:


# Create pipeline
clf, params = tune_DT()
scale = MinMaxScaler()
estimators = [('scale', scale), ('clf', clf)]
pipe = Pipeline(estimators)

# Create GridSearchCV Instance
grid = GridSearchCV(pipe, params)
grid.fit(features_train, labels_train)

# Final classifier
clf = grid.best_estimator_

print('\n=> Chosen parameters :')
print(grid.best_params_)


# In[295]:


# Metrics
predictions = clf.predict(features_test)
print("Accuracy Score:", accuracy_score(labels_test, predictions))
print("Classification Report:\n", classification_report(labels_test, predictions))
tn, fp, fn, tp = confusion_matrix(labels_test, predictions).ravel()
print("Confusion Matrix:\n", tn, fp, fn, tp)


# In[ ]:




