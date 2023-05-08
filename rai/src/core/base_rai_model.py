
from enum import Enum
import json
import string
import numpy as np
from sklearn.cluster import k_means
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.neighbors import *


import torch
import torch.nn as nn

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the MLP model
model = MLP(10, 20, 1)
if hasattr(model, 'state_dict'):
    print('model is a PyTorch model')
else:
    print('model is not a PyTorch model')

class ModelFramework(Enum):
    """Type of Framework used to train the model"""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    OTHER = "other"

class ModelType(Enum):
    LINEAR_REGRESSOR = str(type(LinearRegression()))
    LOGISTIC_REGRESSOR = str(type(LogisticRegression()))
    DECISION_TREE_REGRESSOR = str(type(DecisionTreeRegressor()))
    DECISION_TREE_CLASSIFIER = str(type(DecisionTreeClassifier()))
    RANDOM_FOREST_REGRESSOR = str(type(RandomForestRegressor()))
    RANDOM_FOREST_CLASSIFIER = str(type(RandomForestClassifier()))
    XGBOOST_REGRESSOR = str(type(GradientBoostingRegressor()))
    XGBOOST_CLASSIFIER = str(type(GradientBoostingClassifier()))
    SUPPORT_VECTOR_REGRESSOR = str(type(SVR()))
    SUPPORT_VECTOR_CLASSIFIER = str(type(SVC()))
    K_MEANS = str(type(k_means(np.array([[2,3,4,5]]), 1)))
    KN_CLASSIFIER = str(type(KNeighborsClassifier()))
    KN_REGRESSOR = str(type(KNeighborsRegressor()))
    PYTORCH_MODEL = 'PyTorch Model'
    TENSORFLOW_MODEL = 'TensorFlow Model'

class BaseRAIModel:
    """Type of Framework used to train the model"""
    
    # __model_name = None
    # __framework = None
    
    # __emissions = 0.0
    # __interpretability = 0.0
    
    # __emissions_index = 0
    # __interpretability_index = 0
    
    # __model_index = 0.0
    # __model_accuracy = 0.0
    
    # index_weightage = "EQUAL"
    'sklearn.linear_model._base.LinearRegression'
    # ### EmissionsTracker ###
    # __tracker = None
    
    def __init__(self, model_name:string):
        
        # General Model information
        self.__model_name = model_name
        self.__framework = ModelFramework.SKLEARN
        
        # Responsible Model Metrics
        self.__emissions = 0.0
        self.__interpretability = 0.0
        self.__robustness = 0.0

        # Responsible Index
        self.__emissions_index = 0
        self.__class_balance_index = 0
        self.__interpretability_index = 0
        self.__robustness_index = 0
        
        # Overall Responsible Index
        self.__model_index = 0.0
        self.__model_accuracy = 0.0 
                
    def get_model_name(self)->string:
        return self.__model_name
    
    def get_framework(self)->ModelFramework:
        return self.__framework
    
    def get_emissions(self)->float:
        return self.__emissions
    
    def get_class_balance(self)->float:
        return self.__class_balance
    
    def get_interpretability(self)->float:
        return self.__interpretability

    def get_emissions_index(self)->float:
        if self.__emissions_index == 0 :
            self.__calculate_emissions_index()
            
        return self.__emissions_index
    
    def get_interpretability_index(self)->float:
        if self.__interpretability_index == 0:
            self.__calculate_interpretability_index()
        
        return self.__interpretability_index
    
    def get_class_balance_index(self)->float:
        if self.__class_balance_index == 0:
            self.__calculate_class_balance_index()
            
        return self.__class_balance_index
    
    def set_model_name(self, model):
        model_name = None
        if hasattr(model, 'state_dict'):
            model_name = 'PyTorch Model'
        elif hasattr(model, 'get_weights'):
            model_name = 'TensorFlow' 
        else:
            model_name = str(type(model))
        self.__model_name = model_name
        
    def set_framework(self, framework):
        self.__framework = framework
                
    def set_emissions(self, emissions):
        self.__emissions = emissions
        
    def set_interpretability(self, interpretability):
        self.__interpretability = interpretability
        
    def set_index_weightage(self, index_weightage):
        self.index_weightage = index_weightage
        
    def set_model_accuracy(self, accuracy):
        self.__model_accuracy = accuracy
        
    def get_model_info(self):
        
        value = json.dumps({"model name": self.__model_name,
                    "framework": self.__framework.value,
                    "model_accuracy": self.__model_accuracy,
                    "emissions": self.__emissions,
                    "interpretability": self.__interpretability,
                    "interpretability index": self.__interpretability_index,
                    "emission index": self.__emissions_index,
                    "model_rai_index": self.__model_index})
        
        return value
    