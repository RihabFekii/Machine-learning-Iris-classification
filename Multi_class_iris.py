""" Classification multiclass """
from sklearn import datasets

iris = datasets.load_iris() #iris: the dataset is a structure 
data= iris.data 
label= iris.target

from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(data,label,test_size=0.33,random_state=0)


""" modèle de classification en utilisant SVM multi-classe, un contre un """
from sklearn import svm 

"""Linear """

clf_lin=svm.SVC(kernel='linear', C=1 , decision_function_shape='ovo') 

import time
start_time_lin = time.time()


clf_lin.fit(x_train1,train_label)  #fit will do the train on the train dataset

train_time_lin=time.time() - start_time_lin

y_pred_test_lin=clf_lin.predict(x_test1)   #we test with the trained model on a data without labes to get a label

#we work with multiclass no need for roc or auc 

from sklearn.metrics import accuracy_score 

acc_lin=accuracy_score(test_label, y_pred_test_lin)


""" RBF """

clf_RBF=svm.SVC(kernel='rbf', C = 10.0, gamma=0.1, decision_function_shape='ovo') 

import time
start_time_RBF = time.time()


clf_RBF.fit(x_train1,train_label)  #fit will do the train on the train dataset

train_time_RBF =time.time() - start_time_RBF

y_pred_test_RBF=clf_lin.predict(x_test1)   #we test with the trained model on a data without labes to get a label

#we work with multiclass no need for roc or auc 

from sklearn.metrics import accuracy_score 

acc_RBF=accuracy_score(test_label, y_pred_test_RBF)

""" Sigmoid """

clf_sig=svm.SVC(kernel='sigmoid', gamma=0.5, random_state=0, decision_function_shape='ovo') 

import time
start_time_sig = time.time()


clf_sig.fit(x_train1,train_label)  #fit will do the train on the train dataset

train_time_sig=time.time() - start_time_sig

y_pred_test_sig=clf_sig.predict(x_test1)   #we test with the trained model on a data without labes to get a label

#we work with multiclass no need for roc or auc 

from sklearn.metrics import accuracy_score 

acc_sig=accuracy_score(test_label, y_pred_test_sig)


#precision
from sklearn.metrics import classification_report
print(classification_report(test_label, y_pred_test_lin))
print(classification_report(test_label, y_pred_test_RBF))
print(classification_report(test_label, y_pred_test_sig))

#complexity is train time since pc is i7 so it's zero 
