#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


# In[ ]:


data = pd.read_csv('emotions_2.csv')
data


# In[ ]:


sample = data.loc[0, 'fft_0_b':'fft_749_b']

plt.figure(figsize=(20, 10))
plt.plot(range(len(sample)), sample)
plt.title("Features fft_0_b through fft_749_b")
plt.show()


# In[ ]:


df=data.replace(['NEGATIVE','POSITIVE','NEUTRAL'],[-1,1,0])
df


# In[ ]:


X=df[['mean_1_a','stddev_1_a','covmat_1_a','eigen_1_a','logm_1_a','entropy1_a','correlate_1_a']]
#X.head()


# In[ ]:


y=df['label']
#y.head()


# In[ ]:


#X = np.array(X).reshape(-1, 2)
#y = np.array(y).reshape(-1, 1)


# In[ ]:


#y


# In[ ]:


y = np.array(y).ravel()


# In[ ]:


#y


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


#len(X_train)


# In[ ]:


#len(X_test)


# ## Random forest classification

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=100)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


y_pred_rfc = rfc.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred_rfc)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:


conf_mat_rfc = confusion_matrix(y_test, y_pred_rfc)
print('Confusion matrix:')
conf_mat_rfc


# In[ ]:


label_mapping = {'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1}


# In[ ]:


clr_rfc = classification_report(y_test, y_pred_rfc, target_names=label_mapping.keys())


# In[ ]:


'''plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat_rfc, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')

plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr_rfc)'''


# In[ ]:


test_data = pd.read_excel('test_values_3.xlsx',sheet_name='Sheet2')
#test_data.head()


# In[ ]:


feature1=test_data['mean_1_a']
feature2=test_data['stddev_1_a']
feature3=test_data['covmat_1_a']
feature4=test_data['eigen_1_a']
feature5=test_data['logm_1_a']
feature6=test_data['entropy1_a']
feature7=test_data['correlate_1_a']


# In[ ]:


import RPi.GPIO as gpio
import time
gpio.setmode(gpio.BCM)
led=[17,18,27]
red_led=17
blue_led=18
green_led=27
gpio.setup(red_led,g.OUT)
gpio.setup(blue_led,g.OUT)
gpio.setup(green_led,g.OUT)


# In[ ]:


for t1,t2,t3,t4,t5,t6,t7 in zip(feature1,feature2,feature3,feature4,feature5,feature6,feature7):
    
    input_test_data = (t1,t2,t3,t4,t5,t6,t7)
    input_data_array = np.asarray(input_test_data)
    input_data_array_reshape=input_data_array.reshape(1,-1)
    
    prediction=rfc.predict(input_data_array_reshape)
    
    if prediction==0:
        emotion_result="neutral"
        gpio.output(green_led,gpio.HIGH)
        time.sleep(1)
        gpio.output(green_led,gpio.LOW)
    elif prediction==1:
        emotion_result="happy"
        gpio.output(blue_led,gpio.HIGH)
        time.sleep(1)
        gpio.output(blue_led,gpio.LOW)
    elif prediction==-1:
        emotion_result="sad"
        gpio.output(red_led,gpio.HIGH)
        time.sleep(1)
        gpio.output(red_led,gpio.LOW)
    print("Test data ",input_test_data,"\nThe emotion is ",emotion_result)
    print()

