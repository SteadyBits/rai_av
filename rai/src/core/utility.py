import json
import os
import string

from core.base_rai_model import ModelType

def get_nn_model_name(model)->string:
    model_name = None
    if hasattr(model, 'state_dict'):
        model_name = ModelType.PYTORCH_MODEL
    elif hasattr(model, 'get_weights'):
        model_name = ModelType.TENSORFLOW_MODEL
    
    return model_name

def get_nn_model_layers(model):
    model_name = get_nn_model_name(model)

    if model_name is ModelType.PYTORCH_MODEL:
        return list(model.children())
    elif model_name is ModelType.TENSORFLOW_MODEL:
        return list(model.layers)
    
    return None


def read_log(filepath):
    with open(filepath, 'r') as file:
        # Load the contents of the file into a Python object
        data = json.load(file)
        
    return data

def write_log(filepath, dict_object):
    """
    Append data in JSON format to the end of a JSON file.
    NOTE: Assumes file contains a JSON object (like a Python
    dict) ending in '}'. 
    :param filepath: path to file
    :param data: dict to append
    """

    # Create the directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


    # Create the file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as file:
            # Write an empty json object
            file.write("{}")

    # edit the file in situ - first open it in read/write mode
    with open(filepath, 'r+') as f:

        f.seek(0,2)        # move to end of file
        index = f.tell()    # find index of last byte

        # walking back from the end of file, find the index 
        # of the original JSON's closing '}'

        f.seek(0,2)        # move to end of file
        index = f.tell()    # find index of last byte
        while not f.read().startswith('}'):
            index -= 1
            if index == 0:
                raise ValueError("can't find JSON object in {!r}".format(filepath))
            f.seek(index)

        # starting at the original ending } position, write out
        # the new ending

        f.seek(index)
        if index <= 2:
            # construct JSON fragment as new file ending
            new_ending = json.dumps(dict_object, indent=4)[1:-1] + "}"
            f.write(new_ending)
        else:
            # construct JSON fragment as new file ending
            new_ending = ",\n " + json.dumps(dict_object, indent=4)[1:-1] + "}\n"
            f.write(new_ending)