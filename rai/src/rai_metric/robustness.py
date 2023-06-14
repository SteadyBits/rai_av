import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

class Robustness:

    def __init__(self):
        self.__robustness_index = None
        self.__robustness_scores = {}
        self.__perturbation_scores = {}

    def add_guassian_noise(self, sensor_data, sensor_info, mean=0, stddev=25):
        if  sensor_info:
            if sensor_info['type'] == 'camera':
                # Generate random noise with the same shape as the input image
                sensor_data = sensor_data[1][:, :, :3]
                noise = np.random.normal(mean, stddev, sensor_data.shape)
                # Add the noise to the input image
                sensor_data = np.add(sensor_data, noise, out=sensor_data, casting="unsafe")
                # Ensure pixel values are clamped within the valid range (0-255)
                sensor_data = np.clip(sensor_data, 0, 255)

            elif sensor_info['type'] == 'lidar':
                # Generate random noise with the same shape as the input image
                sensor_data = sensor_data[1][:, :, :3]
                noise = np.random.normal(mean, stddev, sensor_data.shape)
                # Add the noise to the input image
                sensor_data = np.add(sensor_data, noise, out=sensor_data, casting="unsafe")
                # Ensure pixel values are clamped within the valid range (0-255)
                sensor_data = np.clip(sensor_data, 0, 255)

        return sensor_data
    
    def add_salt_and_pepper_noise(self, sensor_data, sensor_info, probability):
        if  sensor_info:
            if sensor_info['type'] == 'camera':
                # Generate random noise with the same shape as the input image
                sensor_data = sensor_data[1][:, :, :3]
                height, width, _ = sensor_data.shape

                # Generate random noise mask
                mask = np.random.random((height, width)) < probability

                # Add salt and pepper noise
                sensor_data[mask] = np.where(np.random.random(mask.sum()) < 0.5, 0, 255)
            elif sensor_info['type'] == 'lidar':
                # Generate random noise with the same shape as the input image
                sensor_data = sensor_data[1][:, :, :3]
                height, width, _ = sensor_data.shape

                # Generate random noise mask
                mask = np.random.random((height, width)) < probability

                # Add salt and pepper noise
                sensor_data[mask] = np.where(np.random.random(mask.sum()) < 0.5, 0, 255)

        return sensor_info

    def add_occlussion_noise(self, sensor_data, sensor_info, num_vertices=6):
        # Create an empty mask array
        if  sensor_info:
            if sensor_info['type'] == 'camera':
                # Generate random noise with the same shape as the input image
                sensor_data = sensor_data[1][:, :, :3]
                width, height = sensor_data.shape
                mask = np.zeros((height, width), dtype=np.uint8)

                # Generate random polygon vertices
                vertices = []
                for _ in range(num_vertices):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    vertices.append([x, y])

                # Create a polygon mask using fillPoly
                cv2.fillPoly(mask, [np.array(vertices)], 255)
                sensor_data[mask] = 0
            elif sensor_info['type'] == 'lidar':
                # Generate random noise with the same shape as the input image
                sensor_data = sensor_data[1][:, :, :3]
                width, height = sensor_data.shape
                mask = np.zeros((height, width), dtype=np.uint8)

                # Generate random polygon vertices
                vertices = []
                for _ in range(num_vertices):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    vertices.append([x, y])

                # Create a polygon mask using fillPoly
                cv2.fillPoly(mask, [np.array(vertices)], 255)
                sensor_data[mask] = 0

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