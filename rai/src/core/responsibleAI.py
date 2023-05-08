from enum import Enum
import json

from core.base_rai_model import BaseRAIModel
from rai_metric.interpretability import Interpretability
from rai_metric.emission import Emission
from rai_metric.robustness import Robustness
from core.utility import read_log, write_log

class ResponsibleMetrics(Enum):
    EMISSIONS = "emissions"
    INTERPRETABILITY = "interpretability"
    ROBUSTNESS = "robustness"
    
class Emissions_Level(Enum):
    """Level of Emissions"""
    LOW = 500
    MEDIUM = 10000

class SensorArgs:
    def __init__(self, sensor, data, no_of_images, concat, order, concat_axis):
        self.sensor = sensor
        self.data = data
        self.no_of_images = no_of_images
        self.concat = concat
        self.order = order
        self.concat_axis = concat_axis


class RAIModels(BaseRAIModel):

    def __init__(self, model, model_args):
        self.__model_list = []
        
        self.__is_addnoise = True

        #mean and stdv of noise distributions
        self.__noise_params = model_args['model_args']['noise_params'] if 'noise_params' in model_args['model_args'] \
              else [[0.5, 0.0], [0.5, 0.2], [0.5, 0.3], [0.5, 0.4]]
        
        self.__sensors = [ model_input['sensor'] for model_input in model_args['model_args'] ]

        self.__noise_id = 0
        self.__sensor_id = 0

        self.__model = model
 
        self.set_model_name(self.__model)

        self.__interpreter = Interpretability(self.__model)
        self.__robuster = Robustness()
        self.__emitter = Emission()

        self.__output = None

        self.__no_predictions = 0
        self.__emission_calc_rate = 20
        self.__direction_id = 0
        self.__directions = ['front', 'right', 'left', 'rear']
        self.__input_dimensions = len(model_args['model_args'][0]['order'].keys())
        
        self.__mean = 0.5
        self.__stddev = 0.0 # modify between 0.1 and 0.5
    
    def parse_model_args(self, model_args):
        camera_data = None
        lidar_data = None
        sensor_data = {}
        for model_input in model_args:
            if model_input.senor == 'camera':
                sensor_data['camera'] = model_input.data

            elif model_input.sensor == 'lidar':
                sensor_data['lidar'] = model_input.data
        return sensor_data
    
    def predict(self, model_args):    
        model_args = [SensorArgs(**arg) for arg in model_args['model_args']]
        # add noise to data from sensors
        noised_inputs = []
        self.__is_addnoise = True
        if self.__is_addnoise:
            model_input = model_args[self.__sensor_id]
            if model_input.sensor == 'camera':
                if self.__directions[self.__direction_id] in model_input.order:
                    noised_inputs.append(self.__robuster.noise_camera_input(model_input, \
                        self.__noise_params[self.__noise_id][0], self.__noise_params[self.__noise_id][1], \
                        self.__directions[self.__direction_id]))
            elif model_input.senor == 'lidar':
                if self.__directions[self.__direction_id] in model_input.order:
                    noised_inputs.append(self.__robuster.noise_lidar_input(model_input, \
                        self.__noise_params[self.__noise_id][0], self.__noise_params[self.__noise_id][1], \
                        self.__directions[self.__direction_id]))
        
        # In case no noise was added to the input
        if len(noised_inputs) == 0:
            noised_inputs.append(model_args[0].data)

        # only estimate emission for a select amount of time due to processng speed issues
        if self.__no_predictions % self.__emission_calc_rate == 0:
            self.__emitter.start_emissions_tracker()
            self.__output = self.__model.forward(*noised_inputs)
            self.__emitter.stop_emissions_tracker()

        else:
            self.__output = self.__model.forward(*noised_inputs)
        
        #keep track of how many predictions have been made so as to know when to 
        #trigger the carbon emission function
        self.__no_predictions += 1

        return self.__output
        

    # call this method after each scenario run is complete. Robustness stats would be logged
    # a new runs starts with the next noise flavour
    def register_model_rai(self, model_accuracy):

        # Increment directions after each run to noise the input from the next camera
        
        print("+++++++++++++++++", self.__input_dimensions)
        
        final_metric_analysis = None
        
        #prediction counter starts from zero for a new run
        self.__no_predictions = 0

        self.__robuster.set_robustness_value(model_accuracy, self.__directions[self.__direction_id], self.__sensors[self.__sensor_id])

        self.__noise_id += 1

        if self.__noise_id == len(self.__noise_params):
            self.__direction_id += 1
            self.__noise_id = 0

            # Check whether all input dimensions have been visited
            if self.__direction_id == self.__input_dimensions:
                self.__sensor_id += 1
                # Check when we have visited all sensors and input dimensions. This means the end of the process
                # Final robustness would now be evaluated if we reached the end of all process.
                # otherwise just increment counters and continue processing.
                if self.__sensor_id == len(self.__sensors):
                    
                    self.__emitter.estimate_training_emission(self.__model)
                    # create dictionary key
                    dict_key = self.__sensors[self.__sensor_id - 1] + '_' + self.__directions[self.__direction_id - 1]

                    # Get metrics as dictionary object
                    final_metric_analysis = self.__estimate_global_metrics(model_accuracy, dict_key)
                    filename = "results/rai_index.json"
                    write_log(filename, final_metric_analysis)
                
                self.__direction_id = 0
                
        

    def __estimate_global_metrics(self, model_accuracy, dict_key="final_result"):

        #Read metrics written to file in order to calculate final robustness value
        # logged_metrics = dict(read_log(filename))

        # for key, value in logged_metrics:
        #     self.__robuster.set_robustness_value(value['model_accuracy'])
        
        value = dict({dict_key : {"model name": self.get_model_name(), # self.__model_name,
                    "model_accuracy": model_accuracy,
                    "interpretability": self.__interpreter.get_interpretability(),
                    "interpretability index": self.__interpreter.get_interpretability_index(),
                    "single_prediction_emission": self.__emitter.get_mean_inference_emissions(),
                    "total_inference_emissions": self.__emitter.get_total_inference_emissions(),
                    "training_emissions": self.__emitter.get_training_emissions(self.__model),
                    "emission index": self.__emitter.get_emissions_index(),
                    "robustness": self.__robuster.get_robustness()
                    }})
        
        return value
    

    def add_model(self, model):
        self.model_list.append(model)
        
    def remove_model(self, modelname):
        self.model_list.remove(modelname)
        
    def list_models(self):
        model_json = ""
        for model in self.model_list:
            model_json += model.get_model_info()
            if model != self.model_list[-1]:
                model_json += ","
                                
            model_json += "\n"
            
        model_json = "[" + model_json + "]"
        
        return model_json
    
    def get_model(self, modelname):
        for model in self.model_list:
            if model.get_model_name() == modelname:
                return model
        return "Model information NOT Found"
    
    def rank_models(self, rank_by = "rai_index"):
        sorted_json = ""
        
        if rank_by == "rai_index":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_model_index(), reverse=True)
        elif rank_by == ResponsibleMetrics.EMISSIONS:
            sorted_models = sorted(self.model_list, key=lambda x: x.get_emissions_index(), reverse=True)
        elif rank_by == ResponsibleMetrics.ROBUSTNESS:
            sorted_models = sorted(self.model_list, key=lambda x: x.robustness_index(), reverse=True)
        elif rank_by == ResponsibleMetrics.INTERPRETABILITY:
            sorted_models = sorted(self.model_list, key=lambda x: x.get_interpretability_index(), reverse=True)
            
        for model in sorted_models:
            sorted_json += model.model_rai_components()
            if(model != sorted_models[-1]):
                sorted_json += ","
            sorted_json += "\n"
            
        sorted_json = "[" + sorted_json + "]"
        return sorted_json
    



# class RAIModels(BaseRAIModel):
#     model_list = []
    
#     def __init__(self, model_args):
#         self.model_list = []
#         self.model = model_args['model']
#         self.model_args = [SensorArgs(**arg) for arg in model_args]
        
#     def add_model(self, model):
#         self.model_list.append(model)
        
#     def remove_model(self, modelname):
#         self.model_list.remove(modelname)
        
#     def list_models(self):
#         model_json = ""
#         for model in self.model_list:
#             model_json += model.get_model_info() 
#             if model != self.model_list[-1]:
#                 model_json += ","
                                
#             model_json += "\n"
            
#         model_json = "[" + model_json + "]"
        
#         return model_json
    
#     def get_model(self, modelname):
#         for model in self.model_list:
#             if model.get_model_name() == modelname:
#                 return model
#         return "Model information NOT Found"
    
#     def rank_models(self, rank_by = "rai_index"):
#         sorted_json = ""
        
#         if rank_by == "rai_index":
#             sorted_models = sorted(self.model_list, key=lambda x: x.get_model_index(), reverse=True)
#         elif rank_by == ResponsibleMetrics.EMISSIONS:
#             sorted_models = sorted(self.model_list, key=lambda x: x.get_emissions_index(), reverse=True)
#         elif rank_by == ResponsibleMetrics.ROBUSTNESS:
#             sorted_models = sorted(self.model_list, key=lambda x: x.robustness_index(), reverse=True)
#         elif rank_by == ResponsibleMetrics.INTERPRETABILITY:
#             sorted_models = sorted(self.model_list, key=lambda x: x.get_interpretability_index(), reverse=True)
            
#         for model in sorted_models:
#             sorted_json += model.model_rai_components()
#             if(model != sorted_models[-1]):
#                 sorted_json += ","
#             sorted_json += "\n"
            
#         sorted_json = "[" + sorted_json + "]"
#         return sorted_json