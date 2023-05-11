# def my_function():
#     print('Before code snippet')
#     # Code snippet goes here
#     print('After code snippet')

# def with_code_injection(my_function):
#     def new_function():
#         print('Before injecting code')
#         my_function()
#         print('After injecting code')
#     return new_function

# my_function_with_injection = with_code_injection(my_function)
# my_function_with_injection()
from rai.src.core import responsibleAI

model_args = dict({'model_args': [
    {'sensor':'camera',
     'data': 0,
     'no_of_images': 1,
     'concat': True,
     'order': {'front': 0},
     'concat_axis': -1
     }
    ]})

rai_engine = responsibleAI.RAIModels(None, model_args)
robust = rai_engine
robust.get_robustness()