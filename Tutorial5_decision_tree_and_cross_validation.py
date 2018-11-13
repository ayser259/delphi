
# coding: utf-8

# # 1. import dependencies

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score


# # 2. load data

# In[2]:


# load public datasets for the problem
iris = load_iris()


# In[3]:


# understand loaded data object
print(iris)


# In[4]:


# data scheme
iris.feature_names


# # 3. Explore data
# summary stats / plots

# In[5]:


pd.DataFrame(iris.data).describe()


# In[6]:


Counter(iris.target)


# In[7]:


X = iris.data
y = iris.target
plt.figure(figsize=(8, 6))
plt.clf()
plt.scatter(X[:, 1], X[:, 2], c=y) # here we only visualize the second and third features
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('some plot')
plt.show()


# # 4. choose 1st model --> decision tree training

# In[8]:


# if go ahead without training and test splitting
dt = DecisionTreeClassifier()
dt.fit(iris.data, iris.target)


# # 5. tree visualization
# export_decision_tree_graph;
# better formatting, adjust node size according to feature names length/font size;
# pruning doesn't work in export_graphviz. specify in training.

# In[9]:


# 1st way: output a dot file and then change to a png
tree.export_graphviz(dt, 
                     out_file='tree1.dot')
get_ipython().system('dot -Tpng tree1.dot -o tree1.png')


# In[10]:


# 2nd way: view inline
dot_data = tree.export_graphviz(dt, 
                                out_file=None,  
                                max_depth=None,
                                leaves_parallel = True,
                                label = None,
                                node_ids = True,
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,  
                                filled=True, 
                                rounded=True,
                                impurity=False,
                                rotate=False,
                                precision=1,
                                special_characters=True)  
graph = graphviz.Source(dot_data)
graph


# In[11]:


# view in png with a better format
tree.export_graphviz(dt,  out_file='tree2.dot',  
                                max_depth=None,
                                leaves_parallel = True,
                                label = None,
                                node_ids = True,
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,  
                                filled=True, 
                                rounded=True,
                                impurity=False,
                                rotate=False,
                                precision=1,
                                special_characters=True) 
get_ipython().system('dot -Tpng tree2.dot -o tree2.png')


# # 6. overfitting

# In[12]:


# 2nd pros of decision tree: feature selection
dt.feature_importances_


# In[13]:


print("Train Accuracy of the model: \n",dt.score(iris.data, iris.target))


# In[14]:


metrics.confusion_matrix(iris.target, dt.predict(iris.data))


# In[15]:


# without split, no ways of determine if overfitting
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.3)


# In[16]:


# wrong way
print("Train Accuracy of the model: ",dt.score(X_train, y_train))
print("Test Accuracy of the model: ",dt.score(X_test,y_test))


# In[17]:


# right way
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Train Accuracy of the model: ",dt.score(X_train, y_train))
print("Test Accuracy of the model: ",dt.score(X_test,y_test))


# In[18]:


# anther way is by cross_validation
scores = cross_val_score(dt,iris.data, iris.target, cv=5)
print("MSE of every fold in 5 fold cross validation: \n", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))


# # 7. how to combat overfitting

# In[19]:


dt_tune = DecisionTreeClassifier(max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,criterion="entropy")
dt_tune.fit(X_train, y_train)


# In[20]:


print("Train Accuracy of the model: ",dt_tune.score(X_train, y_train))
print("Test Accuracy of the model: ",dt_tune.score(X_test,y_test))


# In[21]:


from sklearn.model_selection import KFold, cross_val_score
scores = cross_val_score(dt_tune,iris.data, iris.target, cv=5)
print("MSE of every fold in 5 fold cross validation: \n", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))


# In[22]:


print("What if an iris has equal sepal length and width, petal length and width as of 5cm:\n", 
      iris.target_names[dt_tune.predict(np.array([5,5,5,5]).reshape(1, -1))],
      '\nthe predicted probability of each class:\n',
      dt_tune.predict_proba(np.array([5,5,5,5]).reshape(1, -1)))


# # 8. try other models

# In[23]:


# Create model object and train
NB = GaussianNB()
NB.fit(X_train, y_train)


# In[24]:


#Model results
print("Probability of the classes: ", NB.class_prior_)
print("Mean of each feature per class:\n", NB.theta_)
print("Variance of each feature per class:\n", NB.sigma_)


# In[25]:


print("Train Accuracy of the model: ",NB.score(X_train, y_train))
print("Test Accuracy of the model: ",NB.score(X_test,y_test))


# In[26]:


print("The confusion matrix:\n", metrics.confusion_matrix(iris.target, NB.predict(iris.data)))


# In[27]:


# Calculating 5 fold cross validation results
scores = cross_val_score(NB,iris.data, iris.target, cv=5)
print("MSE of every fold in 5 fold cross validation: \n", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))


# In[28]:


print("What if an iris has equal sepal length and width, petal length and width as of 5cm:\n", 
      iris.target_names[NB.predict(np.array([5,5,5,5]).reshape(1, -1))],
      '\nthe predicted probability of each class:\n',
      NB.predict_proba(np.array([5,5,5,5]).reshape(1, -1)))

