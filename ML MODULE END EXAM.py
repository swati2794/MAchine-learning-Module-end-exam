#!/usr/bin/env python
# coding: utf-8

# #                             Module End Exam -Machine Learning    seat no:220340325053
#            
# 
# 
# 
# # Q2
# Importing required libraries

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Import data set

# In[200]:


data=pd.read_csv("cosmetic.csv")


# In[201]:


data.head()                 #to show first 5 observations


# In[202]:


data.info()                     #to show info about data


# In[203]:


data=data.drop('ID',axis=1)                      #id column is not required so i will drop it
data.head()


# In[204]:


data.isnull().sum()                        # checking null values in the data set or not


# # from above we can say that there is no null values in the data

# In[205]:


data.shape                # to check numvber of column and rows in the data


# In[206]:


data.describe()                #check statistics about the data


# # count how many values in each group in variable

# In[207]:


data['Income'].value_counts()                      # for income varible there are three group high,medium,low


# In[208]:


data['Age'].value_counts()                     # for age there are theree group <21,21-35,>35


# In[209]:


data['Gender'].value_counts()                    # for gender there are two group male and female


# In[210]:


data['Marital status'].value_counts()                # for marital status there are two groups single and married


# In[211]:


data['Buys'].value_counts()                              # buys is target binary variable


# # Countplot for each categorical variable

# In[212]:


plt.figure(figsize=(5,5))
sns.countplot(data['Income'])


# In[213]:


plt.figure(figsize=(5,5))
sns.countplot(data['Age'])


# In[20]:


plt.figure(figsize=(5,5))
sns.countplot(data['Gender'])


# In[22]:


plt.figure(figsize=(5,5))
sns.countplot(data['Marital status'])


# # checking of Balance data 

# In[23]:


plt.figure(figsize=(5,5))
sns.countplot(data['Buys'])


# # 
# Here we can say that count of buys cosmetics is greater than not buying.
# 
# 
# here buys is our target variable and praprotion of both is not balance it may be possible that data is imbalance in some extend
# 64 % data in yes category and remaining in no category.
# 

# # Preproccesing

# In[218]:


# convert categorical data into numerical data


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
data=data.apply(label.fit_transform)


# In[219]:


print(data)


# # Independent and dependent variables

# In[232]:


x=data.drop('Buys',axis=1)
y=data['Buys']


# # split data into train and test data

# In[233]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=12)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Build Model

# In[234]:


#build decision tree model


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=12)
classifier.fit(x,y)


# # Prediction from test data

# In[225]:


y_pred=classifier.predict(x_test)


# In[226]:


y_pred


# # Prediction for given values of test data

# In[81]:


x_test=np.array([1,1,0,0])


# In[84]:



print(x_test,y_pred[0])


# # for given data [Age < 21, Income = Low, Gender = Female, Marital Status = Married] prection is buying cosmetic 

# # Model Evaluation

# In[230]:


from sklearn.metrics import confusion_matrix,accuracy_score
con_m=confusion_matrix(y_test,y_pred)
con_m


# In[235]:


accuracy_score(y_test,y_pred)
   


# In[242]:


from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,12))
a = plot_tree(classifier, feature_names=x.columns, fontsize=12, filled=True, 
              class_names=['No', 'yes'])


# # Overall conclusion:
#     from the conclusion matrix and accuracy score we can say that there all obseravtions are correctly predicted. But there might be possibility of overfitting due to some imbalace data

# # Q1

# # Import libraries

# In[174]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Import dataset

# In[175]:


data=pd.read_csv("data.csv")
data.head()


# In[176]:


data.shape               #to check number of rows and columns in data set


# In[177]:


data.info()                   #to check info about data


# In[178]:


data.isnull().sum()                #to check missing values in data set 


# In[179]:


data.describe()                 # statsistics about data


# # spliting data into independent and dependent values

# In[269]:




x=data.iloc[:,0:2].values
y=data.iloc[:,-1].values


# # Split data training and testing

# In[259]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)


# In[260]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Building Model

# # Here we build Multiple Linear Regression Model

# In[187]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# # Predcition from test data

# In[188]:


y_pred=reg.predict(x_test)


# In[189]:


y_pred


# # Model evalution

# In[192]:


from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
mae=metrics.mean_absolute_error(y_test,y_pred)        #sum(y-ypred)
mse=metrics.mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("Coefficient of determination",r2)
print("Mean Abosolute error",mae)
print("Mean squared error",mse)
print("root mean squared error ",rmse)


# # If we see r2 is good but error is larger so we try to fit another model polynomial regression for better accuracy and minimize the error

# # Polynomial regression 

# In[193]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)


# In[194]:


y_pred=lin_reg_2.predict(poly_reg.fit_transform(x_test))


# In[195]:


y_pred


# # Model evalution

# In[196]:


from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
mae=metrics.mean_absolute_error(y_test,y_pred)        #sum(y-ypred)
mse=metrics.mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("Coefficient of determination",r2)
print("Mean Abosolute error",mae)
print("Mean squared error",mse)
print("root mean squared error ",rmse)


# # Overall conclusion :
#     
#     
#     if we compare both the data set then accuracy for polynomial regression is better than multiple linear regression aslo mean absolute error also reduce much in case of polynomial resgression 
#     
#     so from the model we can suggest that anil has to choose polynomial regression for his prediction

# In[ ]:




