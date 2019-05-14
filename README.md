# Link-Prediction-in-Social-Network
Given a snapshot of a social network,  we infer which new interactions among its members are likely to occur in the near future.

As a part of  the Hike Social Network Prediction Hackathon: Hikeathon, we were given a set 3 datasets:

## Introduction

Given a snapshot of a social network, can we infer which new interactions among its members are likely to occur in the near future?
Determining the ‘proximity’ of nodes in a network has many applications as given below:
- To suggest interactions or collaborations that haven’t yet been utilized within an organization
- To monitor terrorist networks - to deduce possible interaction between terrorists (without direct evidence) 
- Used in Facebook and Linked In to suggest friends

In the given problem statement, we are asked to find out whether if 2 individuals have  chatted with each other, given their features.


## 1.   Training data:
The data consisted of pairs of nodes, representing people on the Hike social network. 
Each record contained the target value indicating whether the 2 actors have chatted with each other or not.
The is_chat column is the target variable, indicating whether the two actors have communicated or not. 1 meaning ‘yes’ and 0 meaning ‘no’


##  2.   User Features
This dataset contains a numerical representation of all the actors in the network.
There are exactly 13 features representing each actor, with some of the features being of categorical nature and the rest as continuous values.
Each record is a vector of features of a given node as shown below.

## 3.    Test Data
The test data is similar to the training data, except for the absence of the is_chat target variable.


The tools used along with their features are:
**Pandas**	Used for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series and for storing the data in the form of dataframes.

**Scikit-learn**	Framework for machine learning algorithms and data manipulation

**Lightgbm**	LightGradientBoosting is a gradient boosting framework that uses tree based learning algorithm, where leaves in the tree as grown vertically instead of horizontally. This framework is used for prediction of the target variable.

**Networkx**	NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. The social network graph is created from the nodes using NetworkX.

**Matplotlib**	For plotting visualizations in the form of graphs.

