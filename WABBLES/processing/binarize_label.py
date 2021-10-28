#########################################
#          binarize_label               #
#########################################
"""
    binarize_label sets the label of a LC to 0 for PC and 1 for NTP  
"""
def binarize_label(label):
    y = 0
    if(label != 0): #0 is PC, 2 turns into 1 is NTP
        y = 1
    return y