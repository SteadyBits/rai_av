from aigovernance import rai_engine
from numpy import np 

img_data = np.array([1,2,3,4,5,6])
#set model arguments
model_args = dict({'model_args': [
    {'sensor':'camera',
     'data': img_data,
     'no_of_images': 3,
     'concat': True,
     'order': {'front': 0, 'left': 1, 'right': 2},
     'concat_axis': -1
     },
     {'sensor': 'lidar',
     'data':img_data,
     'no_of_images': 3,
     'concat': True,
     'order': {'front': 0, 'left': 1, 'right': 2},
     'concat_axis': -1
     }
    ]})