from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

class Robustness:

    def __init__(self):
        self.__robustness_index = None
        self.__robustness_scores = {}
        self.__perturbation_scores = {}

    
    # def noise_camera_input(self, sensor_data, mean=0, stddev=0.0, direction='front'):
    
    #     model_inputs = sensor_data.data
    #     start_idx = sensor_data.order[direction] * sensor_data.data.shape[2]
    #     end_idx = start_idx + sensor_data.data.shape[2]
        
    #     input_image = model_inputs[:, 0, :, start_idx:end_idx]
        
    #     # Generate random noise with the same shape as the input image
    #     noise = np.random.normal(mean, stddev, input_image.shape)
    #     # Add the noise to the input image
    #     input_image += torch.tensor(noise.astype(np.uint8))

    #     model_inputs[:, 0, :, start_idx:end_idx] = input_image

    #     figure = plt.figure(figsize=(10, 8))
    #     cols, rows = 5, 5
    #     for i in range(1, cols * rows + 1):
    #         sample_idx = torch.randint(len(model_inputs), size=(1,)).item()
    #         img = model_inputs[sample_idx]
    #         figure.add_subplot(rows, cols, i)
    #         plt.title("labels")
    #         plt.axis("off")
    #         plt.imshow(img.squeeze(), cmap="gray")
    #     plt.show()

    #     return model_inputs


    def guassian_noise(self, sensor_data, sensor_info, mean=0, stddev=1):
        if 'camera' in sensor_info:
            # Generate random noise with the same shape as the input image
            noise = np.random.normal(mean, stddev, sensor_data.shape)
            # Add the noise to the input image
            sensor_data += noise.astype(np.uint8)

        elif 'lidar' in sensor_info:
            # Generate random noise with the same shape as the input image
            noise = np.random.normal(mean, stddev, sensor_data.shape)
            # Add the noise to the input image
            sensor_data += noise.astype(np.uint8)

        return sensor_data
    
    def noise_lidar_input(self, sensor_data, mean=0, stddev=50, direction='front'):
        
        model_inputs = sensor_data.data
        start_idx = sensor_data.order[direction] * sensor_data.data.shape[2]
        end_idx = start_idx + sensor_data.data.shape[2]
        
        input_image = model_inputs[0, 0, :, start_idx:end_idx]
        
        # Generate random noise with the same shape as the input image
        noise = np.random.normal(mean, stddev, input_image.shape)
        # Add the noise to the input image
        input_image += noise.astype(np.uint8)

        model_inputs[0, 0, :, start_idx:end_idx] = input_image

        return model_inputs

    
    def __calculate_robustness(self):
        
        # Use the average of the ratios between subsequent scores (i.e., adjacent scores)
        print("scores------", self.__perturbation_scores)
        for scores_key, scores in self.__perturbation_scores.items():
            robustness = 0
            prev_score = scores[0]
            for score in scores:
                current_score = score
                epsilon = 1e-9
                robustness += (current_score / (prev_score + epsilon))
                prev_score = current_score

            robustness /= len(scores)
            self.__robustness_scores[scores_key] = robustness

    def __calculate_robustness_index(self):
        
        self.__robustness_index = self.__robustness_scores * 5

        # if self.__robustness >= 0.6:
        #     self.__robustness_index = 3
        # elif self.__robustness > 0.4 and self.__robustness < 0.6:
        #     self.__robustness_index = 2
        # else:
        #     self.__robustness_index = 1

    def get_robustness(self)->float:
        self.__calculate_robustness()
        return self.__robustness_scores
    
    def get_robustness_index(self)->float:
        if self.__robustness_index == None:
            self.__calculate_robustness_index()
        
        return self.__robustness_index
    
    def set_robustness_value(self, value, direction_id, sensor_id):
        score_key = str(sensor_id) + '_' + str(direction_id)
        if score_key not in self.__perturbation_scores:
            self.__perturbation_scores[score_key] = [value]
        else:
            self.__perturbation_scores[score_key].append(value)