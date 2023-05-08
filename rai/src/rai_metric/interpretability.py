import numpy as np
import pandas as pd

from core.base_rai_model import ModelType
from core.utility import get_nn_model_layers, get_nn_model_name

TRANSPARENT = 5
BLACKBOX = 0
class Interpretability:

    def __init__(self, model):
        self.__interpretability_index = 0
        self.__interpretability = None
        self.__model = model

    def calculate_interpretability(self, model):
        if str(str(type(model))) is ModelType.DECISION_TREE_CLASSIFIER or \
            str(str(type(model))) is ModelType.DECISION_TREE_REGRESSOR or \
            str(type(model)) is ModelType.LINEAR_REGRESSOR or \
            str(type(model)) is ModelType.LOGISTIC_REGRESSOR or \
            str(type(model)) is ModelType.K_MEANS or \
            str(type(model)) is ModelType.KN_CLASSIFIER or \
            str(type(model)) is ModelType.KN_REGRESSOR:
                
                self.__interpretability = TRANSPARENT

        elif str(type(model)) is ModelType.RANDOM_FOREST_CLASSIFIER or \
            str(type(model)) is ModelType.RANDOM_FOREST_REGRESSOR or \
            str(type(model)) is ModelType.XGBOOST_CLASSIFIER or \
            str(type(model)) is ModelType.XGBOOST_REGRESSOR:
                self.__interpretability = 3 * (1 - 1/len(model.n_estimators))
        elif get_nn_model_name(self.__model) is ModelType.PYTORCH_MODEL or \
            get_nn_model_name(self.__model) is ModelType.TENSORFLOW_MODEL:
             n_layers = len(get_nn_model_layers(self.__model))

             if n_layers is not None:
                  self.__interpretability = 2 * (1 - 1/n_layers)
             else:
                  self.__interpretability = BLACKBOX
        else:
             self.__interpretability = BLACKBOX
    
    def __calculate_interpretability_index(self):
        
        if self.__interpretability >= 0.6:
            self.__interpretability_index = 3
        elif self.__interpretability > 0.4 and self.__interpretability < 0.6:
            self.__interpretability_index = 2
        else:
            self.__interpretability_index = 1

    def get_interpretability(self)->float:
        if self.__interpretability == None:
            self.calculate_interpretability(self.__model)
        return self.__interpretability
    
    def get_interpretability_index(self)->float:
        if self.__interpretability_index == 0:
            self.__calculate_interpretability_index()
        
        return self.__interpretability_index