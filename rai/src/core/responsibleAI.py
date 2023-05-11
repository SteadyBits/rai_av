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

    def __init__(self):
        self.__robuster = Robustness()
        self.__emitter = Emission()

        self.no_predictions = 0
        self.emission_calc_rate = 20

        self.__mean = 0.5
        self.__stddev = 0.0 # modify between 0.1 and 0.5

    def start_emission_tracker(self):
        self.__emitter.start_emissions_tracker()

    def stop_emission_tracker(self):
        self.__emitter.stop_emissions_tracker()

    def perturb_data(self, input_data, sensor_info):
        
        input_to_noise = input_data[sensor_info['id']]
        noised_input = None

        noised_input = self.__robuster.guassian_noise(input_to_noise, sensor_info)
        input_data[sensor_info['id']][1][:, :, :3] = noised_input

        self.no_predictions += 1

        return input_data
    
    # call this method after each scenario run is complete. Robustness stats would be logged
    # a new runs starts with the next noise flavour
    def register_model_rai(self, model_accuracy):

        # Increment directions after each run to noise the input from the next camera
        
        final_metric_analysis = None
        
        #prediction counter starts from zero for a new run
        self.__no_predictions = 0

        self.__robuster.set_robustness_value(model_accuracy)

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