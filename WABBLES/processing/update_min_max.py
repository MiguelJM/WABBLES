#########################################
#          update_min_max               #
#########################################
"""
    This function updates the minimum and maximum of an array. (This is used for performance metrics).
"""
def update_min_max(val, minVal, maxVal):
    if(val < minVal):
        minVal = val
    if(val > maxVal):
        maxVal = val
    return minVal, maxVal