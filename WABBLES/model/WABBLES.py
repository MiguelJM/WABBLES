import numpy as np
import time
import math
import random
from WABBLES.processing.color import color

"""    
 This function loads the mapping and enhancement parameters from a file. 
 The files are created using Experiment 58, which enables the user to 
 save the parameters during the Gradient Descent Epochs (GDE) of the training
 process.
"""
def load_WABBLES_parameters(parameters_path, N1, N3, no_features):
    
    t = load_matrix(parameters_path + "t.txt", N1, no_features)
    d = load_vector(parameters_path + "d.txt", N1)
    we = load_vector(parameters_path + "we.txt", N1)
    be = load_vector(parameters_path + "be.txt", N1)
    wh = load_vector(parameters_path + "wh.txt", N3)
    bh = load_vector(parameters_path + "bh.txt", N3)
    return t, d, we, be, wh, bh
  
"""    
 This function saves the mapping and enhancement parameters to a file. 
 The user decides in which Gradient Descent Epochs (GDE) to save the 
 parameters during the training process. Also, the user may save the parameters
 as many times as wished.
"""
def save_WABBLES_parameters(parameters_path, t, d, we, be, wh, bh, first_line):
    save_matrix(t, parameters_path + "t.txt", first_line)
    save_vector(d, parameters_path + "d.txt", first_line)
    save_vector(we, parameters_path + "we.txt", first_line)
    save_vector(be, parameters_path + "be.txt", first_line)
    save_vector(wh, parameters_path + "wh.txt", first_line)
    save_vector(bh, parameters_path + "bh.txt", first_line)
    return
    
""" 
    This function loads an np.array object from a txt file.
"""
def load_vector(path, n):
    parameter = np.asarray(np.zeros((n,1))) # ndarray (n, 1)
    line_index = 0
    
    file = open(path,"r")
    file_lines = file.readlines()

    for line in file_lines:
        if(line[0] == "#" or line[0] == "t"): #Ignore comments and labels
            continue
    
        data = line.split(",")
        parameter[line_index] = float(data[0])
        line_index += 1
        
    file.close() 
    return parameter

""" 
    This function loads an np.array object with matrix dimensions from a txt file.
"""
def load_matrix(path, n, m):
    parameter = np.asarray(np.zeros((n,m))) # ndarray (n, m)
    line_index = 0
    
    file = open(path,"r")
    file_lines = file.readlines()

    for line in file_lines:
        if(line[0] == "#" or line[0] == "t"): #Ignore comments and labels
            continue
    
        data = line.split(",")
        
        for dim in range(len(data)):
            parameter[line_index, dim] = float(data[dim])
            
        line_index += 1
        
    file.close() 
    return parameter

""" 
    This function saves an ndarray in a txt file.
"""
def save_vector(vector, path, first_line):    
    file = open(path,"w+")    
    file.write(first_line)
    
    for i_vector in range(vector.shape[0]):
        file.write(str(vector[i_vector][0]) + "\n")
    
    file.close() 
    return

""" 
    This function saves an ndarray of matrix dimensions in a txt file.
"""
def save_matrix(matrix, path, first_line):    
    file = open(path,"w+")    
    file.write(first_line)
    
    for i in range(matrix.shape[0]):
        string = ""
        for j in range(matrix.shape[1]):
            string = string + str(matrix[i, j]) + ","
        l = len(string) # This is used to remove the last ','
        file.write(string[:l-1] + "\n")
    
    file.close() 
    return

"""    
 This function initializes the enhancement parameters 
 wh are the enhancement weights (randomly initialized)
 bh are the enhancement betas (initialized as 0's) 
"""
def initialize_enhancement_parameters(N3):
    
    wh = np.asarray(np.random.rand(N3, 1)) # ndarray (N3, 1)
    bh = np.asarray(np.zeros((N3, 1))) # ndarray (N3, 1)
    
    return wh, bh
    
"""    
 This function initializes the mapping parameters 
 we are the mapping weights (randomly initialized)
 be are the mapping betas (initialized as 0's) 
 T is the wavelet translation parameter matrix
 D is the wavelet dilation parameter matrix
 levels are number of decomposition levels, and it is set according to the available nodes
"""
def initialize_mapping_parameters(N1, no_features):
    
    we = np.asarray(np.random.rand(N1, 1)) # ndarray (N1, 1)
    be = np.asarray(np.zeros((N1,1))) # ndarray (N1, 1)
    t = np.asarray(np.zeros((N1, no_features))) # ndarray (N1, no_features)
    
    T = []
    D = []
    a_domain = 0
    b_domain = 1
    levels = math.floor(math.log(N1) / math.log(2)); # The num. of decomp levels is set according to the max power of 2 obtainable with the available enhancement nodes
    
    T, D = initialize_t_d(T, D, a_domain, b_domain, levels, no_features)
    
    while( len(T) < N1): #Randomly initialize the missing enhancement nodes
        T.append(np.ones(no_features) * (a_domain+(b_domain-a_domain) * random.random()))
        D.append(0.5 *(b_domain-a_domain) * math.pow(2, -levels))
    
    for i_node in range(0, N1):
        for i_feature in range(0, no_features):
            t[i_node, i_feature] = T[i_node][i_feature]
    
    t = np.array(T) # ndarray (N1, no_features)
    d = np.asarray(np.transpose(np.asmatrix(np.array(D)))) # ndarray (N1, 1)

    return we, be, t, d

"""
 This function initialices the wavelet parameters
 A p point is chosen from [a_domain, b_domain], where [a_domain, b_domain] is the domain of the function to be approximated (see 1992 Zhang, Benveniste)
 p is the center of gravity of [a_domain,b_domain] t_1 = p, d_1 = eps(b_domain-a_domain); this generates subintervals where t_2, d_2 are initialized and so on
 The rest of the nodes are initialized randomly 
"""
def initialize_t_d(T, D, a_domain, b_domain, level, no_features):
     
    p = 0.5 * (a_domain + b_domain) # Translation parameter is the center of gravity of [a, b]
    ts = np.ones(no_features)
    for i in range(no_features):
        ts[i] = p #* random.random() # p by a 0-1 number
    t_i = ts  # For each enhancement node generate n_features t's to be multiplied with each input feature
    d_i = 0.5 * (b_domain - a_domain) # Dilation parameter
    
    T.append(t_i);
    D.append(d_i);
    
    if( level <= 1):
        return T,D
    else:
        initialize_t_d(T, D, a_domain, p, level-1, no_features)
        initialize_t_d(T, D, p, b_domain, level-1, no_features)
    
    return T,D

""" This function creates the enhancement nodes that consist of phi(wh*psi + bh)
"""       
def create_enhancement_nodes(no_neurons, Z, wh, bh, phi_name):    
    H = np.zeros((no_neurons, 1)); # Enhancement nodes initialization ndarray (no_neurons, 1)   
    
    for i_enh_node in range(0, no_neurons): #For each mapping node          
        x = np.sum(Z)
        enhancement_input = linear_regression( wh[i_enh_node, :], x, bh[i_enh_node, 0] ) #wh*[Z_1 + Z_2 + ... + Z_N1] + bh     
        H[i_enh_node, 0] = phi( enhancement_input, phi_name ) # Each mapping node contains phi(sum(we*x + be)) 

    return H



"""
 This function obtains the wavelet transform of the x inputs
 x - a light curve, i.e. one sample at a time
 d - wavelet dilation parameter
 t - wavelet translation parameter
    z = d(x - t)
 returns:
    sum_x_minus_t_square = the square sum of every (x_i - t_i)
    c = ||d * (x - t)||
    psi - the wavelet transform
"""
def radial_wavelet(x, d, t, no_neurons, no_features, wavelet_name):      
    #(x - t)    
    x_minus_t = radial_wavelet_x_minus_t(x, t, no_neurons, no_features)
    #z = d(x - t)
    z = radial_wavelet_z(d, x_minus_t, no_neurons, no_features)
    
    #sum((xi - ti)^2)
    sum_x_minus_t_square = radial_wavelet_sum_x_minus_t_square(x_minus_t, no_neurons)

    #c_j = ||d * (x - t)||  
    c = radial_wavelet_c(z, no_neurons)
    
    psi = radial_wavelet_psi(c, no_neurons, wavelet_name)

    return c, psi, sum_x_minus_t_square

"""
 Calculate (x - t)
"""
def radial_wavelet_x_minus_t(x, t, N1, no_features):    
    #(x - t)    
    x_minus_t = np.zeros((N1, no_features))
    
    for i_neuron in range(N1):        
        x_minus_t[i_neuron, :] = x - t[i_neuron, :] 
        
    return x_minus_t

"""
 Calculate z = d(x - t)
"""
def radial_wavelet_z(d, x_minus_t, N1, no_features):   
    #z = d(x - t)    
    z = np.zeros((N1, no_features)) # ndarray(N1, no_features)
    
    for i_neuron in range(N1):        
        z[i_neuron, :] = d[i_neuron] * x_minus_t[i_neuron, :]  
    
    return z
  
"""
 Calculate sum((xi - ti)^2)
"""
def radial_wavelet_sum_x_minus_t_square(x_minus_t, no_neurons):
    
    sum_x_minus_t_square = np.zeros((no_neurons, 1))
       
    #sum((xi - ti)^2)   
    x_minus_t_square = np.power(x_minus_t, 2)
    sum_x_minus_t_square[:, 0] = np.transpose(sum(np.transpose(x_minus_t_square)))
        
    return sum_x_minus_t_square   
  
"""
 Calculate c_j = ||d * (x - t)|| = sqrt(sum(d * (x - t)))
"""
def radial_wavelet_c(z, no_neurons):  
    c = np.zeros((no_neurons, 1))    
    c[:,0] = np.transpose(np.sqrt(sum(np.transpose(np.power(z, 2)))));
    
    return c


"""
 Calculate psi (the wavelet)
"""
def radial_wavelet_psi(c, no_neurons, wavelet_name): 
    psi = np.zeros((no_neurons, 1))
    
    for j in range(no_neurons):        
        psi[j] = wavelet_function(c[j], wavelet_name) # Apply the wavelet transform to c_j
     
    return psi

"""
 This function applies the CWT to the radial function c
"""
def wavelet_function(c, wavelet_name):
    # Choose a wavelet
    if wavelet_name == 'gausD':
        # First derivative of the Gaussian wavelet (from the detrivative of Gaussian DOG family)
        # -c * exp(-1/2 * c^2)
        res = -c * np.exp(-0.5 * np.power(c, 2));
    elif wavelet_name == 'morlet':
        # Morlet wavelet
        # exp(-1/2 * c^2) * cos(5c)
        res = np.exp(-0.5 * np.power(c, 2)) * math.cos(5 * c);
    elif wavelet_name == 'mexHat':
        # Mexican hat wavelet
        # ( 2 / (sqrt(3) * pi^(1/4) ) * ( 1-c^2 ) * exp(-1/2 * c^2)
        res = ( 2 / ( (math.pi ** (1/4)) * math.sqrt(3) ) * ( 1 - np.power(c, 2) ) * np.exp(-0.5 * np.power(c, 2)) )
    else:
        res = np.nan
        print('error')
    return res

"""
 This function obtains the Z mapping nodes, which are obtained as we_j * psi(c_j) + be_j; for each jth neuron
"""
def create_mapping_nodes(we, psi, be, no_neurons):
    
    Z = np.zeros((no_neurons, 1)); # Mapping nodes initialization ndarray (no_neurons, 1)
        
    for i_map_nodes in range(0, no_neurons): #Obtain the value of each mapping node                   
        Z[i_map_nodes] = linear_regression(we[i_map_nodes, 0], psi[i_map_nodes, 0], be[i_map_nodes, 0]) #we*psi + be
    
    return Z

"""
 This functions applies the linear regression formula: mx + b
"""
def linear_regression(m, x, b):
    res = m*x + b
    return res


"""
 This function obtains the derivative of each trainable parameter
 Inputs sizes
     x (1, N2*N1)
 It returns the following derivatives: 
    we_deriv - from the mapping weights we
    be_deriv - from the mapping betas be
    wh_deriv - from the enhancement weights wh
    bh_deriv - from the enhancement betas bh
    t_deriv - from the wavelet translation parameter
    d_deriv - from the wavelet dilation parameter
    
"""
def calculate_derivatives(err, we, be, wh, bh, Z, H, psi, c, t, d, mapping_inputs, N1, N3, wavelet_name, phi_name):  
    
    we_deriv = we.copy()    
    be_deriv = be.copy()
    t_deriv = t.copy()    
    d_deriv = d.copy()
    wh_deriv = wh.copy()    
    bh_deriv = bh.copy()
    
    
    for i_map in range(0, N1): #For each mapping node        
        be_deriv[i_map, 0] = calculate_be_derivative(N3, wh, Z[i_map, 0], bh, err, phi_name)
        we_deriv[i_map, 0] = psi[i_map, 0] * be_deriv[i_map, 0] # err * (  psi(c_i) * ( 1 + phi'(wh_j*Z_i + bh_j) * wh )  )     
        t_deriv[i_map, :] = calculate_t_derivative(we[i_map, 0], c[i_map, 0], d[i_map, 0], mapping_inputs, t[i_map, :], be_deriv[i_map, 0], psi[i_map, 0], wavelet_name)
        d_deriv[i_map, 0] = calculate_d_derivative(we[i_map, 0], c[i_map, 0], d[i_map, 0], mapping_inputs, t[i_map, :], be_deriv[i_map, 0], psi[i_map, 0], wavelet_name)

        
    for i_enhanc in range(0, N3): #For each enhancement node
        bh_deriv[i_enhanc, 0] = err * phi_derivative(H[i_enhanc, 0], phi_name) # err * phi'(H[i])
        wh_deriv[i_enhanc, 0] = np.sum(Z) * bh_deriv[i_enhanc, 0] # err * Z * phi'(H[i])
    
    return we_deriv, be_deriv, t_deriv, d_deriv, wh_deriv, bh_deriv

"""
 This function is used to calculate the derivative of the be mapping parameter
 be' = err * (1 + phi'(wh_j*Z_i + bh_j) * wh)
""" 
def calculate_be_derivative(N3, wh, Z, bh, err, phi_name):
    be_deriv = 1 

    for i_enhanc in range(0, N3): #For each enhancement node
        Z_i = linear_regression(wh[i_enhanc, 0], Z, bh[i_enhanc, 0]) # wh_j*Z_i + bh_j
        be_deriv += phi_derivative(Z_i, phi_name) * wh[i_enhanc, 0] # (1 + phi'(wh_j*Z_i + bh_j) * wh)
        
    be_deriv = err * be_deriv # err * (1 + phi'(wh_j*Z_i + bh_j) * wh)
    return be_deriv

"""
 This function is used to calculate the derivative of the t translation parameter
     c is now a scalar value (c_j)
""" 
def calculate_t_derivative(we, c, d, x, t, be_deriv, psi, wavelet_name):
    # Choose a wavelet
    if wavelet_name == 'gausD': # First derivative of the Gaussian wavelet        
        t_deriv = gausD_t_derivative(we, c, d, x, t, be_deriv)
    elif wavelet_name == 'morlet': # Morlet wavelet        
        t_deriv = morlet_t_derivative(we, c, d, x, t, be_deriv, psi)
    elif wavelet_name == 'mexHat': # Mexican hat wavelet
        t_deriv = mexHat_t_derivative(we, c, d, x, t, be_deriv, psi)               
    else:
        print('error')
        exit()
    return t_deriv

"""
Derivative of the Gaussian Derivative wavelet w.r.t the translation parameter (t)
     t' = err * we * (e^(-.5 * c^2) * (d^2*(x-t)) ) * (1/c + c) * (1 + phi'_i * wh)
"""
def gausD_t_derivative(we, c, d, x, t, be_deriv):
    t_deriv = we * be_deriv
    t_deriv = t_deriv * (1/c + c)
    t_deriv = t_deriv * (np.exp(-0.5 * np.power(c, 2)))
    t_deriv = t_deriv * (np.power(d, 2) * (x - t))
    
    return t_deriv

"""
Derivative of the morlet wavelet w.r.t the translation parameter (t)
    t' = err * we * ((d^2*(x-t)) ) * (psi(c) + (5/c * e^(-.5 * c^2) * sen(5c))) * (1 + phi'_i * wh)
"""
def morlet_t_derivative(we, c, d, x, t, be_deriv, psi):
    t_deriv = we * be_deriv
    t_deriv = t_deriv * (psi + (5/c)*(np.exp(-0.5 * np.power(c, 2)))*np.sin(5*c))
    t_deriv = t_deriv * (np.power(d, 2) * (x - t))
    
    return t_deriv
               
"""
Derivative of the mexican hat wavelet w.r.t the translation parameter (t)
    t' = err * we * ((d^2*(x-t)) ) * ( ( 4/ (pi^(1/4)*sqrt(3)) ) * e^(-.5 * c^2) + psi(c) ) * (1 + phi'_i * wh)
"""
def mexHat_t_derivative(we, c, d, x, t, be_deriv, psi):
    t_deriv = we * be_deriv
    t_deriv = t_deriv * (( 4 / ( (math.pi ** (1/4)) * math.sqrt(3) ) * np.exp(-0.5 * np.power(c, 2))) + psi)
    t_deriv = t_deriv * (np.power(d, 2) * (x - t))
    
    return t_deriv
               
"""
 This function is used to calculate the derivative of the d dilation parameter
"""
def calculate_d_derivative(we, c, d, x, t, be_deriv, psi, wavelet_name):
    # Choose a wavelet
    if wavelet_name == 'gausD': # First derivative of the Gaussian wavelet        
        d_deriv = gausD_d_derivative(we, c, d, x, t, be_deriv, psi)
    elif wavelet_name == 'morlet': # Morlet wavelet        
        d_deriv = morlet_d_derivative(we, c, d, x, t, be_deriv, psi)
    elif wavelet_name == 'mexHat': # Mexican hat wavelet
        d_deriv = mexHat_d_derivative(we, c, d, x, t, be_deriv, psi)
    else:
        print('error')
        exit()
    return d_deriv

"""
Derivative of the Gaussian Derivative wavelet w.r.t the dilation parameter (d)
    d' = err * -we * (d^2*(x-t)^2) * (1/c * e^(-.5 * c^2) + psi(c)) * (1 + phi'_i * wh)
"""
def gausD_d_derivative(we, c, d, x, t, be_deriv, psi):        
    d_deriv = -we * be_deriv
    d_deriv = d_deriv * (d * np.sum(np.power((x - t), 2)))
    d_deriv = d_deriv * ( (np.exp(-0.5 * np.power(c, 2)) / c)  + psi )
    
    return d_deriv

"""
Derivative of the morlet wavelet w.r.t the dilation parameter (d)
     d' = err * -we * (d^2*(x-t)^2) * (psi(c) + (5/c * e^(-.5 * c^2) * sen(5c))) * (1 + phi'_i * wh)
"""
def morlet_d_derivative(we, c, d, x, t, be_deriv, psi):  
    d_deriv = -we * be_deriv
    d_deriv = d_deriv * (d * np.sum(np.power((x - t), 2))) 
    d_deriv = d_deriv * ( psi + (5/c) * (np.exp(-0.5 * np.power(c, 2))) * (np.sin(5*c)) )
    
    return d_deriv

"""
Derivative of the mexican hat wavelet w.r.t the dilation parameter (d)
     d' = err * -we * (d^2*(x-t)^2) * ( ( 4/ (pi^(1/4)*sqrt(3)) ) * e^(-.5 * c^2) + psi(c) )) * (1 + phi'_i * wh)
"""
def mexHat_d_derivative(we, c, d, x, t, be_deriv, psi):  
    d_deriv = -we * be_deriv
    d_deriv = d_deriv * (d * np.sum(np.power((x - t), 2))) 
    d_deriv = d_deriv * (( 4 / ( (math.pi ** (1/4)) * math.sqrt(3) ) * np.exp(-0.5 * np.power(c, 2))) + psi)
    
    return d_deriv


"""
  This function updates the parameters using their derivatives and the alpha learning rate 
"""
def update_parameters(wh, bh, t, d, we, be, wh_deriv, bh_deriv, t_deriv, d_deriv, we_deriv, be_deriv, alpha):    

    wh = update_parameter(wh, wh_deriv, alpha) 
    bh = update_parameter(bh, bh_deriv, alpha)
    t = update_parameter(t, t_deriv, alpha)
    d = update_parameter(d, d_deriv, alpha)
    we = update_parameter(we, we_deriv, alpha)
    be = update_parameter(be, be_deriv, alpha)    
    
    return wh, bh, t, d, we, be

""" Function to update a single parameter """
def update_parameter(param, param_deriv, alpha):
    param = param - alpha * param_deriv; 
    
    return param



""" PHI functions """

"""
 This function applies the phi function to the enhancement inputs
"""
def phi(x, phi_name):
    if phi_name == 'sigmoid':
        res = sigmoid(x)
    elif phi_name == 'tanh':
        res = tanh(x)
    else:
        res = np.nan
        print('PHI error')
    return res

"""
 This function applies the derivative of the phi function
"""
def phi_derivative(x, phi_name):
    if phi_name == 'sigmoid':
        res = derivative_sigmoid(x)
    elif phi_name == 'tanh':
        res = derivative_tanh(x)
    else:
        res = np.nan
        print('PHI error')
    return res

""" Sigmoid function """
def sigmoid( x ):
    return 1 / (1 + np.exp(-x));

""" Derivative of the sigmoid function """
def derivative_sigmoid( x ):
    return np.exp(-x) / np.power(1 + np.exp(-x), 2);


""" Tanh function """
def tanh( x ):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x));

""" Derivative of the tanh function tanh' = 1 - tanh(x)^2 """
def derivative_tanh( x ):
    return 1 - np.power((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), 2);


""" WABBLES testing process """
def test_model(test_x, test_y, t, d, we, be, wh, bh, N1, N3, wavelet_name, phi_name, class_threshold):
    no_samples_test = test_x.shape[0]
    no_features_test = test_x.shape[1]
    y_test_estimated = np.zeros(no_samples_test)

    for i_test_sample in range(0, no_samples_test): # Subject each sample to the mapping and enhancement nodes     

             ### MAPPING NODES PHASE ############################        
            test_mapping_inputs = np.zeros((1, no_features_test));
            test_mapping_inputs[0, :] = np.copy(test_x[i_test_sample,:]); #Inputs of the mapping nodes  

            test_c, test_psi, test_sum_x_minus_t_square = radial_wavelet(test_mapping_inputs, d, t, N1, no_features_test, wavelet_name)

            test_Z = create_mapping_nodes(we, test_psi, be, N1)

            ### ENHACNEMENT NODES PHASE ############################          
            test_H = create_enhancement_nodes(N3, test_Z, wh, bh, phi_name)

            #### RESULT EVALUATION PHASE            
            y_test_estimated[i_test_sample] = np.sum(test_Z) + np.sum(test_H) #sum(psi(we*x + be)) * sum(phi(wh*[Z_1...Z_N1]+bh))

            del test_H;
            del test_Z;

    # Obtain metrics

    cnt = 0;
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    unknown = 0

    for i in range(0, len(y_test_estimated)):
        if y_test_estimated[i] < class_threshold:
            y_test_estimated[i] = 0
        else:
            y_test_estimated[i] = 1

        if round(y_test_estimated[i]) == test_y[i]:
            cnt = cnt + 1;

        # Check number of correct identifications  test_yy is the correct label, y is the predicted label
        if(test_y[i] == round(y_test_estimated[i]) and round(y_test_estimated[i]) == 0):
            TP += 1
        elif(test_y[i] == round(y_test_estimated[i]) and round(y_test_estimated[i]) == 1):
            TN += 1
        elif(test_y[i] != round(y_test_estimated[i]) and round(y_test_estimated[i]) == 0):
            FP += 1
        elif(test_y[i] != round(y_test_estimated[i]) and round(y_test_estimated[i]) == 1):
            FN += 1
        else:
            unknown += 1

    print("TP, TN, FP, FN:", TP, TN, FP, FN)

    #TestingAccuracy = cnt * 1.0 / test_y.shape[0];

    label = test_y;
    predicted = y_test_estimated;

    acc = 0
    spec = 0
    sen = 0
    prec = 0
    spec = 0
    fscore = 0

    if(TP != 0 or TN != 0 or FP != 0 or FN != 0 ):
        acc = ((TP + TN) / (TP + TN + FP + FN)) * 100

    if(TP != 0 or FP != 0 ):
        prec = ((TP) / (TP + FP)) * 100

    if( TN != 0 or FP != 0 ):
        spec = ((TN) / (TN + FP)) * 100

    if( TP != 0 or FN != 0 ):
        sen = ((TP) / (TP + FN)) * 100

    if(TP != 0 or FP != 0 or FN != 0 ):
        fscore = (TP / (TP + (0.5 * (FP + FN)))) * 100  #Or (2*TP / (2*TP + FP + FN) ; which is the same

    if( TN != 0 or FP != 0 ):
        spec = ((TN) / (TN + FP)) * 100

    TestingAccuracy = acc
    f_score = fscore
    TestingPrecision = prec
    TestingRecall = sen
    TestingSpecificity = spec

    return TestingAccuracy, f_score, TestingPrecision, TestingRecall, TestingSpecificity, TP, TN, FP, FN



"""
    Function that creates the BLS model. It returns the training and test accuracy, training and
    testing time, and testing F-Score.
        bls_train_fscore(train_x, train_y, test_x, test_y, s, C, N1, N2, N3)
        'train_x' and 'test_x' are the entire training data and test data.
        'train_y' and 'test_y' are the entire training labels and test labels.
        's' is the shrinkage parameter for enhancement nodes.
        'C' is the parameter for sparse regularization.
        'N1' is the number of mapped feature nodes.
        'N2' are the groups of mapped features.
        'N3' is the number of enhancement nodes.
        'err_threshold' is the error limit to which the gradient descent will stop
        'gradient_descent_epochs' are the maximum number of epochs to train the enhancement nodes
        'parameters_path' contains a string with the path where the parameters will be saved/loaded
        'class_threshold' is the classification threshold, it can be chosen from 0 to 1, having a value closer to 0 means more priority to class 1 and having a value closer to 1 will prioritize class 0
        'invert_LC' decides if the LC has been turned upsidedown or not

    Randomoly generated weights for the mapped features and enhancement nodes are
    stored in the
    matrices 'we' and 'wh,' respectively.
"""
def wabbles(train_x, train_y, test_x, test_y, N1, N3, err_threshold, gradient_descent_epochs, alpha, wavelet_name, phi_name, class_threshold, writer, parameters_path, load_parameters, train_model, save_parameters, invert_LC):
    
    no_samples = train_x.shape[0]
    no_features = train_x.shape[1]
    last_epoch_saved = 0 #last epoch in which the parameters were saved.
    
## TRAINING PROCESS ####################
    time_start=time.time()
    TrainingAccuracy = 0
    Training_time = 0
    
    print(parameters_path)
    if load_parameters: # Use previously save parameters or initialize them?
        t, d, we, be, wh, bh = load_WABBLES_parameters(parameters_path, N1, N3, no_features)
        print("Mapping and enhancement parameters are loaded.")
    else:
        we, be, t, d = initialize_mapping_parameters(N1, no_features)
        print("Mapping initialization is complete.")

        wh, bh = initialize_enhancement_parameters(N3)
        print("Enhancement initialization is complete.")
    
    if train_model: # Train the model? 
        total_err = 100
        n_epochs = 0
        y_estimated = np.zeros(no_samples)
        last_acc_obtained = 0 #Used for automatic parameter saving

        print("Training the mapping and enhancement nodes.")

        writer.writerow(["", "Gradient Descent Test Data"])
        writer.writerow(["", "Gradient Descent Epoch (GDE)", "Total error", "Accuracy", "F-Score", "Precision", "Recall",
                            "Specificity", "TP", "TN", "FP", "FN"])

        while(total_err > err_threshold and gradient_descent_epochs > n_epochs ): #Error is greater than the threshold and there still are epochs to go

            n_epochs += 1
            total_err = 0

            for i_sample in range(0, no_samples): # Subject each sample to the mapping and enhancement nodes     


                ### MAPPING NODES PHASE ############################        
                mapping_inputs = np.zeros((1, no_features));
                mapping_inputs[0, :] = np.copy(train_x[i_sample,:]); #Inputs of the mapping nodes  

                c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, wavelet_name)

                Z = create_mapping_nodes(we, psi, be, N1)

                ### ENHACNEMENT NODES PHASE ############################          
                H = create_enhancement_nodes(N3, Z, wh, bh, phi_name)

                #### RESULT EVALUATION PHASE            
                y_estimated[i_sample] = np.sum(Z) + np.sum(H) #sum(psi(we*x + be)) * sum(phi(wh*[Z_1...Z_N1]+bh))

                err = y_estimated[i_sample] - train_y[i_sample]
                total_err += abs(err)

                #Update the parameters 
                we_deriv, be_deriv, t_deriv, d_deriv, wh_deriv, bh_deriv = calculate_derivatives(err, we, be, wh, bh, Z, H, psi, c, t, d, mapping_inputs, N1, N3, wavelet_name, phi_name)

                wh, bh, t, d, we, be = update_parameters(wh, bh, t, d, we, be, wh_deriv, bh_deriv, t_deriv, d_deriv, we_deriv, be_deriv, alpha)
                del H;
                del Z;

            total_err = 0.5 * sum(total_err ** 2)

            print(color.BOLD + "\n\n\n__________________________________________________________________________" +
                                "\nTraining epoch " + str(n_epochs) + " error was " + str(total_err) + "." + color.END)

            """
            # Assess the GDE using the test data #
            #
            #
            ######################################
            """
            
            ### Testing Process
            TestingAccuracy, f_score, TestingPrecision, TestingRecall, TestingSpecificity, TP, TN, FP, FN = test_model(test_x, test_y, t, d, we, be, wh, bh, N1, N3, wavelet_name, phi_name, class_threshold)
    
            print("Testing Accuracy is : ", TestingAccuracy, " %")
            print("Testing F-Score is : ", f_score, " %")
            writer.writerow(["", n_epochs, total_err, round(TestingAccuracy, 2),
                round(f_score, 2), round(TestingPrecision, 2), round(TestingRecall, 2), round(TestingSpecificity, 2), round(TP, 2),
                round(TN, 2), round(FP, 2), round(FN, 2)]) 
            
            if save_parameters:
                if last_acc_obtained < TestingAccuracy:  # Save the parameters in this epoch?                    
                    last_epoch_saved = n_epochs
                    last_acc_obtained = TestingAccuracy

                    # This line describes the model with which the model was trained with 
                    parameter_first_line = "#WABBLES V3, N1="+str(N1)+",N3="+str(N3)+",wavelet_name="+wavelet_name+"phi_name="+phi_name+",alpha="+str(alpha)+",classification_threshold="+str(class_threshold)+",GDEs="+str(gradient_descent_epochs)+",inverted_LCs="+str(invert_LC)+",training_error="+str(total_err)+"\n"

                    save_WABBLES_parameters(parameters_path, t, d, we, be, wh, bh, parameter_first_line)      
                    print("Parameters saved to: ", parameters_path)
            """ This GDE ends here. """
        
        
        time_end=time.time()
        Training_time = (time_end - time_start)

        # Training - end

        print("The Total Training Time is : ", Training_time, " seconds");

        ### Training Accuracy
        cnt = 0;
        for i in range(0, len(y_estimated)):
            if y_estimated[i] < class_threshold:
                y_estimated[i] = 0
            else:
                y_estimated[i] = 1

            if round(y_estimated[i]) == train_y[i]:
                cnt = cnt + 1;

        TrainingAccuracy = cnt * 1.0 / train_y.shape[0];

        print("Training Accuracy is : ", TrainingAccuracy * 100, " %");

    """ 
            TESTING PROCESS

    """
    ### Testing Process
    print("---  Final testing process ---")
    time_start=time.time()
    
    # Test the model
    TestingAccuracy, f_score, TestingPrecision, TestingRecall, TestingSpecificity, TP, TN, FP, FN = test_model(test_x, test_y, t, d, we, be, wh, bh, N1, N3, wavelet_name, phi_name, class_threshold)
    
    time_end=time.time()
    Testing_time = time_end - time_start
    print("The Total Testing Time is : ", Testing_time, " seconds");
    print("Testing Accuracy is : ", TestingAccuracy, " %");
    print("Testing F-Score is : ", f_score, " %");

    return TrainingAccuracy, TestingAccuracy, TestingPrecision, TestingRecall, TestingSpecificity, TP, TN, FP, FN, Training_time, Testing_time, f_score, last_epoch_saved, last_acc_obtained;  
##################################################################################################
