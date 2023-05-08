from codecarbon import EmissionsTracker
import numpy as np

from core.base_rai_model import ModelType
from core.utility import get_nn_model_name
class Emission:
    def __init__(self):
        self.__emissions = 0.0
        self.__emissions_index = 0.0
        self.__tracker = EmissionsTracker()
        self.energy_consumptions = []
        self.__total_inference_energy = -1
        self.__mean_inference_energy = -1
        self.__total_train_energy = -1

    def start_emissions_tracker(self):
        self.__tracker.start()
    
    def stop_emissions_tracker(self):
        self.__emissions : float = self.__tracker.stop()
        self.energy_consumptions.append(self.__emissions)
    
    def get_emissions_index(self)->float:
        if self.__emissions_index == 0 :
            self.__calculate_emissions_index()
            
        return self.__emissions_index
    
    def get_total_inference_emissions(self)->float:
        self.__total_inference_energy = np.sum(self.energy_consumptions)
        return self.__total_inference_energy
    
    def get_mean_inference_emissions(self)->float:
        self.__mean_inference_energy = np.mean(self.energy_consumptions)
        return self.__mean_inference_energy
    
    def get_training_emissions(self, model)->float:
        if self.__total_train_energy == -1:
            self.estimate_training_emission(model)

        return self.__total_train_energy
    
    def __calculate_emissions_index(self):
        self.__mean_inference_energy = np.mean(self.energy_consumptions)
        if self.__mean_inference_energy <= 500:
            self.__emissions_index = 3
        elif self.__mean_inference_energy > 500 and self.emissions <= 10000:
            self.__emissions_index = 2
        else:
            self.__emissions_index = 1

    def estimate_training_emission(self, model, n_samples=50000.0, n_features=1.0, n_folds=10.0, n_epoch=20.0):
        if self.__mean_inference_energy == -1:
            self.__mean_inference_energy = np.mean(self.energy_consumptions)
       
        if str(type(model)) is ModelType.LINEAR_REGRESSOR:
            n_features = len(model.coef_)
            self.__total_train_energy = self.__mean_inference_energy * (n_features * (n_samples + n_features))
        
        elif str(type(model)) is ModelType.LOGISTIC_REGRESSOR:
            self.__total_train_energy = self.__mean_inference_energy * (n_samples)
        
        #Tree algorithms need revisions
        elif str(type(model)) is ModelType.DECISION_TREE_REGRESSOR or \
            str(type(model)) is ModelType.DECISION_TREE_CLASSIFIER or \
            str(type(model)) is ModelType.RANDOM_FOREST_CLASSIFIER or \
            str(type(model)) is ModelType.RANDOM_FOREST_REGRESSOR or \
            str(type(model)) is ModelType.XGBOOST_CLASSIFIER or \
            str(type(model)) is ModelType.XGBOOST_REGRESSOR:
                n_features = model.n_features_
                self.__total_train_energy = self.__mean_inference_energy * n_samples * n_features * n_folds
        
        elif str(type(model)) is ModelType.K_MEANS:
             n_features = model.cluster_centers_.shape[1]
             n_iter = model.n_iter_
             self.__total_train_energy = self.__mean_inference_energy * n_samples * n_iter

        elif get_nn_model_name(model) is ModelType.PYTORCH_MODEL or get_nn_model_name(model) is ModelType.TENSORFLOW_MODEL:
            self.__total_train_energy = n_samples * n_epoch * self.__mean_inference_energy
       
        # elif str(type(model)) is ModelType.PYTORCH_MODEL:
        #     # Loop over all layers in the model
        #     # Get a list of all trainable weight matrices in the model
        #     # for layer in model.children():
        #     #     # Check if the layer has weight parameters
        #     #     if hasattr(layer, 'weight'):
        #     #         # Append the weight matrix to the list
        #     #         product_sum += layer.weight.size()[0] * (layer.weight.size()[1] ** 2)
        #     self.__total_train_energy = 2 * n_samples * n_epoch * product_sum

        #     #OR

        #     # trainable_weights = [param for param in model.parameters() if param.requires_grad]
        #     # # Get the shapes of all trainable weight matrices
        #     # weight_shapes = [param.shape for param in trainable_weights]

        # elif str(type(model)) is ModelType.TENSORFLOW_MODEL:
            # filter out weight matrices from a list of trainable variables
            # product_sum = 0
            # # Loop over the layers of the model
            # for layer in model.layers:
            #     # Check if the layer has trainable weights
            #     if len(layer.get_weights()) > 0:
            #         # Get the shape of the weight matrix
            #         weight_shape = layer.get_weights()[0].shape
            #         # Add the shape to the list
            #         product_sum += weight_shape[0] * (weight_shape[1] ** 2)
            #self.__total_train_energy =  n_samples * n_epoch * self.__mean_inference_energy

            #OR 

            # weight_vars = [var for var in model.trainable_variables if 'kernel' in var.name]
            # product_sum = 0
            # for weight in weight_vars:
            #     product_sum += weight.shape[0] * (weight.shape[1] ** 2)
            # self.__mean_inference_energy = 2 * n_samples * n_epoch