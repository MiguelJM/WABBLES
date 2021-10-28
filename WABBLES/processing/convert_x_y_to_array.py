import numpy as np

#########################################
#       convert_x_y_to_array            #
#########################################
"""
 This function converts the dataframes that contain the concatenation of the features + the labels into numpy arrays
 x_set is a dataframe with the x features array
 y_set is a dataframe with the y labels array
"""
def convert_x_y_to_array(x_set, y_set):
    # Convert dataframes into np.arrays 
    x_array = x_set.to_numpy()
    y_array = np.reshape(np.array(y_set), (-1, 1))

    data = np.concatenate((x_array, y_array), axis = 1)
    
    return data