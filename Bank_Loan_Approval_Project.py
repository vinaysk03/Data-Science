#!/usr/bin/env python
# coding: utf-8

# In[345]:


#Project - Bank Loan Case Study - check eligibility for loan approval depending applications


# In[346]:


import pandas as pd
A = pd.read_csv("C:/Users/sai/Desktop/Data Science Course Folders/Project/training_set.csv")
B=pd.read_csv("C:/Users/sai/Desktop/Data Science Course Folders/Project/testing_set.csv")


# In[347]:


A.info()


# In[348]:


#Fill Blank data


# In[349]:


cat=[]
con=[]
for i in A.columns:
    if (A[i].dtypes==object):
        cat.append(i)
    else:
        con.append(i)


# In[350]:


cat.remove("Loan_Status")


# In[351]:


cat


# In[352]:


for i in cat:
    A[i]=A[i].fillna(A[i].value_counts().index[0])
for j in con:
    A[j]=A[j].fillna(A[j].mean())


# In[353]:


for i in cat:
    B[i]=B[i].fillna(B[i].value_counts().index[0])
for j in con:
    B[j]=B[j].fillna(B[j].mean())


# In[354]:


# Y here is Loan_Status...check relationship with other variables
#Using Bivariate analysis


# In[355]:


import matplotlib.pyplot as plt
import seaborn as sb
m=1
plt.figure(figsize=(15,30))

for i in A.columns:
    if (A[i].dtypes==object):
        plt.subplot(7,3,m)
        sb.countplot(A.Loan_Status, hue=A[i])
        m=m+1
    else:
        plt.subplot(7,3,m)
        sb.boxplot(A.Loan_Status,A[i])
        m=m+1 


# In[356]:


#Dependent variables - Most of the variables have linear relationship with Loan_Status except Loan_ID
# so we need to run logistic regression to predict Loan_Status in data set B


# In[357]:


#Lable Encoding


# In[358]:


con.append("Loan_Status")


# In[359]:


A.columns


# In[360]:


A=A[con].join(pd.get_dummies(A[['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Property_Area']]))


# In[361]:


#Run Logisitc Regression


# In[362]:


Y=A["Loan_Status"]
X=A.drop(labels=["Loan_Status"],axis=1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=20)


# In[363]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model=lr.fit(xtrain,ytrain)
pred=model.predict(xtest)


# In[364]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(ytest,pred)


# In[365]:


accuracy_score(ytest,pred)


# In[366]:


#Apply model tested on past data (set A) for open applications received for bank loan which is in dataaset B


# In[371]:


con.remove("Loan_Status")


# In[ ]:





# In[372]:


B=B[con].join(pd.get_dummies(B[['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Property_Area']]))


# In[373]:


pred_Loan_Status=model.predict(B)


# In[377]:


B["Predicted_Loan_Status"]=pred_Loan_Status


# In[380]:


B.groupby(by="Predicted_Loan_Status").count()


# In[381]:


#Above algorithms give Loan Approval Yes / No for new applications


# In[ ]:




