#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


# In[3]:


data = pd.read_csv('emotions_2.csv')
data


# In[4]:


sample = data.loc[0, 'fft_0_b':'fft_749_b']

plt.figure(figsize=(20, 10))
plt.plot(range(len(sample)), sample)
plt.title("Features fft_0_b through fft_749_b")
plt.show()


# In[5]:


df=data.replace(['NEGATIVE','POSITIVE','NEUTRAL'],[-1,1,0])
df


# In[6]:


X=df[['mean_1_a','stddev_1_a','covmat_1_a','eigen_1_a','logm_1_a','entropy1_a','correlate_1_a']]
X.head()


# In[7]:


y=df['label']
y.head()


# In[8]:


#X = np.array(X).reshape(-1, 2)
#y = np.array(y).reshape(-1, 1)


# In[9]:


y


# In[10]:


y = np.array(y).ravel()


# In[11]:


y


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[13]:


len(X_train)


# In[14]:


len(X_test)


# ## Support vector machine (SVM)

# In[15]:


from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1, random_state=0)


# In[16]:


svm.fit(X_train, y_train)


# In[17]:


score_svm= svm.score(X_test, y_test)


# In[18]:


score_svm = svm.score(X_test, y_test)
print('Test accuracy: {:.2f}%'.format(score_svm * 100))


# In[19]:


y_pred_svm = svm.predict(X_test)


# In[20]:


conf_mat_svm = confusion_matrix(y_test, y_pred_svm)
print('Confusion matrix:')
conf_mat_svm


# In[21]:


label_mapping = {'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1}


# In[22]:


clr_svm = classification_report(y_test, y_pred_svm, target_names=label_mapping.keys())


# In[23]:


plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat_svm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')

plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr_svm)


# In[24]:


test_data = pd.read_excel('test_values_3.xlsx',sheet_name='Sheet2')
test_data.head()


# In[25]:


feature1=test_data['mean_1_a']
feature2=test_data['stddev_1_a']
feature3=test_data['covmat_1_a']
feature4=test_data['eigen_1_a']
feature5=test_data['logm_1_a']
feature6=test_data['entropy1_a']
feature7=test_data['correlate_1_a']


# In[26]:


for t1,t2,t3,t4,t5,t6,t7 in zip(feature1,feature2,feature3,feature4,feature5,feature6,feature7):
    
    input_test_data = (t1,t2,t3,t4,t5,t6,t7)
    input_data_array = np.asarray(input_test_data)
    input_data_array_reshape=input_data_array.reshape(1,-1)
    
    prediction=svm.predict(input_data_array_reshape)
    
    if prediction==0:
        emotion_result="neutral"
    elif prediction==1:
        emotion_result="happy"
    elif prediction==-1:
        emotion_result="sad"
    print("Test data ",input_test_data,"\nThe emotion is ",emotion_result)
    print()


# ## k-nearest neighbour(knn) 

# In[27]:


from sklearn.neighbors import KNeighborsClassifierhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) 
knn = KNeighborsClassifier(n_neighbors=7)


# In[29]:


knn.fit(X_train, y_train)


# In[30]:


score_knn = knn.score(X_test, y_test)


# In[31]:


score_knn = knn.score(X_test, y_test)
print('Test accuracy: {:.2f}%'.format(score_knn * 100))


# In[32]:


y_pred_knn=knn.predict(X_test)


# In[33]:


conf_mat_knn = confusion_matrix(y_test, y_pred_knn)
print('Confusion matrix:')
conf_mat_knn


# In[34]:


clr_knn = classification_report(y_test, y_pred_knn, target_names=label_mapping.keys())


# In[35]:


plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat_knn, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')

plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr_knn)


# In[36]:


for t1,t2,t3,t4,t5,t6,t7 in zip(feature1,feature2,feature3,feature4,feature5,feature6,feature7):
    
    input_test_data = (t1,t2,t3,t4,t5,t6,t7)
    input_data_array = np.asarray(input_test_data)
    input_data_array_reshape=input_data_array.reshape(1,-1)
    
    prediction=svm.predict(input_data_array_reshape)
    
    if prediction==0:
        emotion_result="neutral"
    elif prediction==1:
        emotion_result="happy"
    elif prediction==-1:
        emotion_result="sad"
    print("Test data ",input_test_data,"\nThe emotion is ",emotion_result)
    print()






# # #Decision Tree Classifier

# In[37]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[39]:


clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)


# In[40]:


y_pred_dt = clf_dt.predict(X_test)


accuracy = accuracy_score(y_test, y_pred_dt)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[41]:


conf_mat_dt = confusion_matrix(y_test, y_pred_dt)
print('Confusion matrix:')
conf_mat_dt


# In[42]:


clr_dt = classification_report(y_test, y_pred_dt, target_names=label_mapping.keys())


# In[43]:


plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat_dt, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')

plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr_dt)


# In[44]:


for t1,t2,t3,t4,t5,t6,t7 in zip(feature1,feature2,feature3,feature4,feature5,feature6,feature7):
    
    input_test_data = (t1,t2,t3,t4,t5,t6,t7)
    input_data_array = np.asarray(input_test_data)
    input_data_array_reshape=input_data_array.reshape(1,-1)
    
    prediction=clf_dt.predict(input_data_array_reshape)
    
    if prediction==0:
        emotion_result="neutral"
    elif prediction==1:
        emotion_result="happy"
    elif prediction==-1:
        emotion_result="sad"
    print("Test data ",input_test_data,"\nThe emotion is ",emotion_result)
    print()


# ## Random forest classification

# In[45]:


from sklearn.ensemble import RandomForestClassifier


# In[46]:


rfc = RandomForestClassifier(n_estimators=100)


# In[47]:


rfc.fit(X_train, y_train)


# In[48]:


y_pred_rfc = rfc.predict(X_test)


# In[49]:


accuracy = accuracy_score(y_test, y_pred_rfc)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[50]:


conf_mat_rfc = confusion_matrix(y_test, y_pred_rfc)
print('Confusion matrix:')
conf_mat_rfc


# In[51]:


clr_rfc = classification_report(y_test, y_pred_rfc, target_names=label_mapping.keys())


# In[52]:


plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat_rfc, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')

plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr_rfc)


# In[53]:


for t1,t2,t3,t4,t5,t6,t7 in zip(feature1,feature2,feature3,feature4,feature5,feature6,feature7):
    
    input_test_data = (t1,t2,t3,t4,t5,t6,t7)
    input_data_array = np.asarray(input_test_data)
    input_data_array_reshape=input_data_array.reshape(1,-1)
    
    prediction=rfc.predict(input_data_array_reshape)
    
    if prediction==0:
        emotion_result="neutral"
    elif prediction==1:
        emotion_result="happy"
    elif prediction==-1:
        emotion_result="sad"
    print("Test data ",input_test_data,"\nThe emotion is ",emotion_result)
    print()

