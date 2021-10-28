import unittest
import numpy as np
from WABBLES.model.WABBLES import calculate_be_derivative
from WABBLES.model.WABBLES import radial_wavelet
from WABBLES.model.WABBLES import create_enhancement_nodes
from WABBLES.model.WABBLES import create_mapping_nodes
from WABBLES.model.WABBLES import derivative_sigmoid
from WABBLES.model.WABBLES import phi_derivative
from WABBLES.model.WABBLES import linear_regression
from WABBLES.model.WABBLES import radial_wavelet_c
from WABBLES.model.WABBLES import radial_wavelet_psi
from WABBLES.model.WABBLES import radial_wavelet_sum_x_minus_t_square
from WABBLES.model.WABBLES import radial_wavelet_x_minus_t
from WABBLES.model.WABBLES import radial_wavelet_z
from WABBLES.model.WABBLES import initialize_mapping_parameters
from WABBLES.model.WABBLES import sigmoid
from WABBLES.model.WABBLES import update_parameter
from WABBLES.model.WABBLES import calculate_d_derivative
from WABBLES.model.WABBLES import calculate_derivatives
from WABBLES.model.WABBLES import calculate_t_derivative
from WABBLES.model.WABBLES import save_vector   
from WABBLES.model.WABBLES import load_vector
from WABBLES.model.WABBLES import save_matrix
from WABBLES.model.WABBLES import load_matrix
from WABBLES.model.WABBLES import phi

''' WABBLES Unitary tests '''
class TestWABBLES(unittest.TestCase):
    
    def setUp(self): # This code runs itself before each test (can be used for shared initializations)
        #print("Set up.")
        pass # Delete to use
    
    def tearDown(self): # This code runs itself after each test (can be used for shared initializations)
        #print("Tear down.")
        pass # Delete to use
    
    def test_save_and_load_parameters(self):
        
        # Initialize parameters to test on trustable values
        we, be, t, d = initialize_mapping_parameters(10, 20)
        
        # Test vector
        save_vector(be, parameters_test_path+"vector_test.txt", "#test,parameters,example=2\n")
        be_test = load_vector(parameters_test_path+"vector_test.txt", 10)
        self.assertEqual(np.allclose(be, be_test), True) 

        # Test matrix
        save_matrix(t, parameters_test_path+"matrix_test.txt", "#test,parameters,example=2\n")
        t_test = load_matrix(parameters_test_path+"matrix_test.txt", 10, 20)
        self.assertEqual(np.allclose(t, t_test), True) 
        
    def test_radial_wavelet(self):
        no_features = 5 # n  
        N1 = 3 # m

        mapping_inputs = np.zeros((1, no_features)) # ndarray (1, n)
        mapping_inputs[0,:] = np.asarray([1, 2, 3, 4, 5])

        t = np.zeros((N1, no_features)) # ndarray (m, n)
        t[0,:] = np.asarray([0.006, 0.007, 0.008, 0.009, 0.01])
        t[1,:] = np.asarray([0.02, 0.03, 0.04, 0.05, 0.06])
        t[2,:] = np.asarray([0.008, 0.009, 0.01, 0.011, 0.012])

        d = np.zeros((N1, 1)) # ndarray (m, 1)
        d[0] = 0.2
        d[1] = 0.3
        d[2] = 0.4

        c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, 'gausD')
        
        c_ = np.zeros((N1, 1)) # ndarray (m, 1)
        c_[0] = 1.47973417
        c_[1] = 2.19654501
        c_[2] = 2.95785084
        
        psi_ = np.zeros((N1, 1)) # ndarray (m, 1)
        psi_[0] = -0.49512543
        psi_[1] = -0.19680944
        psi_[2] = -0.03725467
        
        sum_x_minus_t_square_ = np.zeros((N1, 1)) # ndarray (m, 1)
        sum_x_minus_t_square_[0] = 54.74033
        sum_x_minus_t_square_[1] = 53.609
        sum_x_minus_t_square_[2] = 54.68051       
        
        self.assertEqual(np.allclose(c_, c), True)
        self.assertEqual(np.allclose(psi_, psi), True)
        self.assertEqual(np.allclose(sum_x_minus_t_square_, sum_x_minus_t_square), True)
    
    def test_radial_wavelet_x_minus_t(self):
        no_features = 5 # n
        no_neurons = 3 # m
        
        t = np.zeros((no_neurons, no_features)) # ndarray (m, n)
        t[0,:] = np.asarray([6, 7, 8, 9, 10])
        t[1,:] = np.asarray([11, 12, 13, 14, 15])
        t[2,:] = np.asarray([16, 17, 18, 19, 20])
        
        x = np.zeros((1, no_features)) # ndarray (1, n)
        x[0,:] = np.asarray([1, 2, 3, 4, 5])
        
        x_minus_t = radial_wavelet_x_minus_t(x, t, no_neurons, no_features) # ndarray (m, n)
        y = np.zeros((no_neurons, no_features)) # ndarray (m, n)
        y[0,:] = np.asarray([-5, -5, -5, -5, -5])
        y[1,:] = np.asarray([-10, -10, -10, -10, -10])
        y[2,:] = np.asarray([-15, -15, -15, -15, -15])
        
        self.assertEqual(np.allclose(x_minus_t, y), True)
  
    def test_radial_wavelet_z(self):
        no_features = 5 # n
        no_neurons = 3 # m        
        
        d = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        d[0] = 1
        d[1] = 2
        d[2] = 3
        
        x_minus_t = np.zeros((no_neurons, no_features)) # ndarray (m, n)
        x_minus_t[0,:] = np.asarray([4, 5, 6, 7, 8])
        x_minus_t[1,:] = np.asarray([9, 10, 11, 12, 13])
        x_minus_t[2,:] = np.asarray([14, 15, 16, 17, 18])
        
        z = radial_wavelet_z(d, x_minus_t, no_neurons, no_features) # ndarray (m, n)
        y = np.zeros((no_neurons, no_features)) # ndarray (m, n)
        y[0,:] = np.asarray([4, 5, 6, 7, 8])
        y[1,:] = np.asarray([18, 20, 22, 24, 26])
        y[2,:] = np.asarray([42, 45, 48, 51, 54])
        
        self.assertEqual(np.allclose(z, y), True)
        
        
    def test_radial_wavelet_sum_x_minus_t_square(self):
        no_features = 5 # n
        no_neurons = 3 # m  
        
        x_minus_t = np.zeros((no_neurons, no_features)) # ndarray (m, n)
        x_minus_t[0,:] = np.asarray([1, 2, 3, 4, 5])
        x_minus_t[1,:] = np.asarray([6, 7, 8, 9, 10])
        x_minus_t[2,:] = np.asarray([11, 12, 13, 14, 15])
        
        sum_x_minus_t_square = radial_wavelet_sum_x_minus_t_square(x_minus_t, no_neurons) # ndarray (m, 1)
        
        y = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        y[0] = 55
        y[1] = 330
        y[2] = 855 
        
        self.assertEqual(np.allclose(sum_x_minus_t_square, y), True)
        
        
    def test_radial_wavelet_c(self):
        no_features = 5 # n
        no_neurons = 3 # m 
        
        z = np.zeros((no_neurons, no_features)) # ndarray (m, n)
        z[0,:] = np.asarray([1, 2, 3, 4, 5])
        z[1,:] = np.asarray([6, 7, 8, 9, 10])
        z[2,:] = np.asarray([11, 12, 13, 14, 15])
        
        c = radial_wavelet_c(z, no_neurons) # ndarray (m, 1)
        
        y = np.asarray(np.transpose(np.asmatrix([7.41, 18.16, 29.24]))) # ndarray (m, 1)
        
        self.assertEqual(np.allclose(c, y, rtol=1e-02), True) 
          
    def test_radial_wavelet_psi(self):
        
        no_features = 5 # n
        no_neurons = 3 # m       
        
        c = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        c[0] = 1
        c[1] = 2
        c[2] = 3 
        
        psi = radial_wavelet_psi(c, no_neurons, 'gausD') # ndarray (m, 1), ndarray (m, 1)
         
        y_gausD = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        y_gausD[0] = -0.6065
        y_gausD[1] = -0.2706
        y_gausD[2] = -0.0333 
        
        self.assertEqual(np.allclose(psi, y_gausD, rtol=1e-03), True) 
        
        psi = radial_wavelet_psi(c, no_neurons, 'morlet') # ndarray (m, 1), ndarray (m, 1)
        
        y_morlet = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        y_morlet[0] = 0.1720
        y_morlet[1] = -0.11355
        y_morlet[2] = -0.00843
        
        self.assertEqual(np.allclose(psi, y_morlet, rtol=1e-02), True)
        
        psi = radial_wavelet_psi(c, no_neurons, 'mexHat') # ndarray (m, 1), ndarray (m, 1)
        
        y_mexHat = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        y_mexHat[0] = 0
        y_mexHat[1] = -0.352173
        y_mexHat[2] = -0.07709
        
        self.assertEqual(np.allclose(psi, y_mexHat, rtol=1e-02), True)
         
    def test_create_mapping_nodes(self):
        
        no_features = 5 # n
        no_neurons = 3 # m       
        
        be = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        be[0] = 1
        be[1] = 2
        be[2] = 3  
        
        we = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        we[0] = 4
        we[1] = 5
        we[2] = 6 
        
        psi = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        psi[0] = 7
        psi[1] = 8
        psi[2] = 9 
        
        y = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        y[0] = 29
        y[1] = 42
        y[2] = 57
        
        Z = create_mapping_nodes(we, psi, be, no_neurons) # ndarray (m, 1)       
        
        self.assertEqual(np.allclose(Z, y), True) 
        
    def test_linear_regression(self):
        
        no_features = 5 # n
        no_neurons = 3 # m       
        
        b = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        b[0] = 1
        b[1] = 2
        b[2] = 3  
        
        m = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        m[0] = 4
        m[1] = 5
        m[2] = 6 
        
        x = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        x[0] = 7
        x[1] = 8
        x[2] = 9 
        
        y = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        y[0] = 29
        y[1] = 42
        y[2] = 57
        
        res = linear_regression(m,x,b)
        
        y = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        y[0] = 29
        y[1] = 42
        y[2] = 57
        
        self.assertEqual(np.allclose(res, y), True) 
        
    def test_calculate_derivatives(self):
        err = 5
        no_features = 5 # n
        N1 = 3 # m
        N3 = 4 

        mapping_inputs = np.zeros((1, no_features)) # ndarray (1, n)
        mapping_inputs[0,:] = np.asarray([1, 2, 3, 4, 5])

        H = np.zeros((N3, 1)) # ndarray (N3, 1)
        H[0] = 5
        H[1] = 6
        H[2] = 7
        H[3] = 8
        
        t = np.zeros((N1, no_features)) # ndarray (m, n)
        t[0,:] = np.asarray([0.006, 0.007, 0.008, 0.009, 0.01])
        t[1,:] = np.asarray([0.02, 0.03, 0.04, 0.05, 0.06])
        t[2,:] = np.asarray([0.008, 0.009, 0.01, 0.011, 0.012])

        d = np.zeros((N1, 1)) # ndarray (m, 1)
        d[0] = 0.2
        d[1] = 0.3
        d[2] = 0.4

        c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, 'gausD')

        we = np.zeros((N1, 1)) # ndarray (N1, 1)
        we[0] = .4
        we[1] = .5
        we[2] = .6

        be = np.zeros((N1, 1)) # ndarray (N1, 1)
        be[0] = .03
        be[1] = .4
        be[2] = .05

        Z = create_mapping_nodes(we, psi, be, N1)

        wh = np.zeros((N3, 1)) # ndarray (N3, 1)
        wh[0] = 0.04
        wh[1] = 0.05
        wh[2] = 0.06
        wh[3] = 0.07

        bh = np.zeros((N3, 1)) # ndarray (N3, 1)
        bh[0] = 0.08
        bh[1] = 0.09
        bh[2] = 0.1
        bh[3] = 0.11
        
        be_deriv_ = np.zeros((N1, 1)) # ndarray (N1, 1)
        be_deriv_[0,0] = 5.2744
        be_deriv_[1,0] = 5.27408
        be_deriv_[2,0] = 5.2743
        
        we_deriv_ = np.zeros((N1, 1)) # ndarray (N1, 1)
        we_deriv_[0,0] = -2.6114
        we_deriv_[1,0] = -1.0379
        we_deriv_[2,0] = -0.1962
        
        t_deriv_ = np.zeros((N1, no_features)) # ndarray (m, n)
        t_deriv_[0,:] = np.asarray([0.0605, 0.1213, 0.1821, 0.2429, 0.3037])
        t_deriv_[1,:] = np.asarray([0.0553, 0.1110, 0.1669, 0.2227, 0.2785])
        t_deriv_[2,:] = np.asarray([0.0208, 0.0418, 0.0628, 0.0838, 0.1048])

        d_deriv_ = np.zeros((N1, 1)) # ndarray (N1, 1)
        d_deriv_[0,:] = 6.2134
        d_deriv_[1,:] = 6.6168
        d_deriv_[2,:] = 2.2839

        bh_deriv_ = np.zeros((N3, 1)) # ndarray (N3, 1)
        bh_deriv_[0] = 0.0332
        bh_deriv_[1] = 0.0123
        bh_deriv_[2] = 0.0045
        bh_deriv_[3] = 0.0016

        wh_deriv_ = np.zeros((N3, 1)) # ndarray (N3, 1)
        wh_deriv_[0] = 0.0054
        wh_deriv_[1] = 0.0019
        wh_deriv_[2] = 0.0007
        wh_deriv_[3] = 0.0003
        
        we_deriv, be_deriv, t_deriv, d_deriv, wh_deriv, bh_deriv = calculate_derivatives(err, we, be, wh, bh, Z, H, psi, c, t, d, mapping_inputs, N1, N3, 'gausD', 'sigmoid')
        
        self.assertEqual(np.allclose(be_deriv, be_deriv_, rtol=1e-03), True) 
        self.assertEqual(np.allclose(we_deriv, we_deriv_, rtol=1e-02), True) 
        self.assertEqual(np.allclose(bh_deriv, bh_deriv_, rtol=1e-01), True)         
        self.assertEqual(np.allclose(d_deriv, d_deriv_, rtol=1e-02), True)         
        self.assertEqual(np.allclose(t_deriv, t_deriv_, rtol=1e-02), True)  
        self.assertEqual(np.allclose(wh_deriv, wh_deriv_, rtol=1e-01), True)       
        
    def test_calculate_be_derivative(self):
        err = 5
        N1 = 3
        N3 = 4
        
        Z = np.zeros((N1, 1)) # ndarray (N1, 1)
        Z[0] = 0.0001
        Z[1] = 0.0002
        Z[2] = 0.0003
        
        wh = np.zeros((N3, 1)) # ndarray (N3, 1)
        wh[0] = 0.04
        wh[1] = 0.05
        wh[2] = 0.06
        wh[3] = 0.07
        
        bh = np.zeros((N3, 1)) # ndarray (N3, 1)
        bh[0] = 0.08
        bh[1] = 0.09
        bh[2] = 0.1
        bh[3] = 0.11
        
        be_deriv = calculate_be_derivative(N3, wh, Z[0, 0], bh, err, 'sigmoid')
          
        y = 5.27428
        
        self.assertEqual(np.allclose(be_deriv, y, rtol=1e-03), True) 
        

    def test_calculate_t_derivative(self):
        err = 5
        no_features = 5 # n
        N1 = 3 # m
        N3 = 4 

        mapping_inputs = np.zeros((1, no_features)) # ndarray (1, n)
        mapping_inputs[0,:] = np.asarray([1, 2, 3, 4, 5])

        t = np.zeros((N1, no_features)) # ndarray (m, n)
        t[0,:] = np.asarray([0.006, 0.007, 0.008, 0.009, 0.01])
        t[1,:] = np.asarray([0.02, 0.03, 0.04, 0.05, 0.06])
        t[2,:] = np.asarray([0.008, 0.009, 0.01, 0.011, 0.012])

        d = np.zeros((N1, 1)) # ndarray (m, 1)
        d[0] = 0.2
        d[1] = 0.3
        d[2] = 0.4

        c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, 'gausD')

        we = np.zeros((N1, 1)) # ndarray (N1, 1)
        we[0] = .4
        we[1] = .5
        we[2] = .6

        be = np.zeros((N1, 1)) # ndarray (N1, 1)
        be[0] = .03
        be[1] = .4
        be[2] = .05

        Z = create_mapping_nodes(we, psi, be, N1)

        wh = np.zeros((N3, 1)) # ndarray (N3, 1)
        wh[0] = 0.04
        wh[1] = 0.05
        wh[2] = 0.06
        wh[3] = 0.07

        bh = np.zeros((N3, 1)) # ndarray (N3, 1)
        bh[0] = 0.08
        bh[1] = 0.09
        bh[2] = 0.1
        bh[3] = 0.11

        be_deriv = np.zeros((N1, 1)) # ndarray (N1, 1)
        be_deriv[0,0] = calculate_be_derivative(N3, wh, Z[0, 0], bh, err, 'sigmoid')
        be_deriv[1,0] = calculate_be_derivative(N3, wh, Z[1, 0], bh, err, 'sigmoid')
        be_deriv[2,0] = calculate_be_derivative(N3, wh, Z[2, 0], bh, err, 'sigmoid')

        t_deriv = calculate_t_derivative(we, c, d, mapping_inputs, t, be_deriv, psi, 'gausD')

        t_deriv_gausD = np.zeros((N1, no_features)) # ndarray (m, n)
        t_deriv_gausD[0,:] = np.asarray([0.0605, 0.1213, 0.1821, 0.2429, 0.3037])
        t_deriv_gausD[1,:] = np.asarray([0.0553, 0.1110, 0.1669, 0.2227, 0.2785])
        t_deriv_gausD[2,:] = np.asarray([0.0208, 0.0418, 0.0628, 0.0838, 0.1048])
        
        # gausD 
        self.assertEqual(np.allclose(t_deriv, t_deriv_gausD, rtol=1e-02), True) 
        
        ############################### morlet #####################################################
        c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, 'morlet')

        Z = create_mapping_nodes(we, psi, be, N1)

        be_deriv = np.zeros((N1, 1)) # ndarray (N1, 1)
        be_deriv[0,0] = calculate_be_derivative(N3, wh, Z[0, 0], bh, err, 'sigmoid')
        be_deriv[1,0] = calculate_be_derivative(N3, wh, Z[1, 0], bh, err, 'sigmoid')
        be_deriv[2,0] = calculate_be_derivative(N3, wh, Z[2, 0], bh, err, 'sigmoid')

        # morlet
        t_deriv = calculate_t_derivative(we, c, d, mapping_inputs, t, be_deriv, psi, 'morlet')

        t_deriv_morlet = np.zeros((N1, no_features)) # ndarray (m, n)
        t_deriv_morlet[0,:] = np.asarray([0.0975, 0.1955, 0.2935, 0.3915, 0.4896])
        t_deriv_morlet[1,:] = np.asarray([-0.0477, -0.095887, -0.14407, -0.1922, -0.2404])
        t_deriv_morlet[2,:] = np.asarray([0.00466, 0.00935, 0.01404, 0.01874, 0.02343])
        
        # morlet 
        self.assertEqual(np.allclose(t_deriv, t_deriv_morlet, rtol=1e-03), True) 
        
        ############################### Mexican hat #####################################################
        c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, 'mexHat')

        Z = create_mapping_nodes(we, psi, be, N1)

        be_deriv = np.zeros((N1, 1)) # ndarray (N1, 1)
        be_deriv[0,0] = calculate_be_derivative(N3, wh, Z[0, 0], bh, err, 'sigmoid')
        be_deriv[1,0] = calculate_be_derivative(N3, wh, Z[1, 0], bh, err, 'sigmoid')
        be_deriv[2,0] = calculate_be_derivative(N3, wh, Z[2, 0], bh, err, 'sigmoid')

        t_deriv = calculate_t_derivative(we, c, d, mapping_inputs, t, be_deriv, psi, 'mexHat')

        t_deriv_mexHat = np.zeros((N1, no_features)) # ndarray (m, n)
        t_deriv_mexHat[0,:] = np.asarray([0.01972, 0.03955, 0.05938, 0.07921, 0.09903])
        t_deriv_mexHat[1,:] = np.asarray([-0.0329, -0.0663, -0.0996, -0.1329, -0.1662])
        t_deriv_mexHat[2,:] = np.asarray([-0.0315, -0.0633, -0.0950, -0.1268, -0.1586])
        
        self.assertEqual(np.allclose(t_deriv, t_deriv_mexHat, rtol=1e-02), True) 
        
    def test_calculate_d_derivative(self):
        err = 5
        no_features = 5 # 
        N1 = 3 #
        N3 = 4 
        
        mapping_inputs = np.zeros((1, no_features)) # ndarray (1, n)
        mapping_inputs[0,:] = np.asarray([1, 2, 3, 4, 5])

        t = np.zeros((N1, no_features)) # ndarray (m, n)
        t[0,:] = np.asarray([0.006, 0.007, 0.008, 0.009, 0.01])
        t[1,:] = np.asarray([0.02, 0.03, 0.04, 0.05, 0.06])
        t[2,:] = np.asarray([0.008, 0.009, 0.01, 0.011, 0.012])

        d = np.zeros((N1, 1)) # ndarray (m, 1)
        d[0] = 0.2
        d[1] = 0.3
        d[2] = 0.4

        c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, 'gausD')

        we = np.zeros((N1, 1)) # ndarray (N1, 1)
        we[0] = .4
        we[1] = .5
        we[2] = .6

        be = np.zeros((N1, 1)) # ndarray (N1, 1)
        be[0] = .03
        be[1] = .4
        be[2] = .05

        Z = create_mapping_nodes(we, psi, be, N1)

        wh = np.zeros((N3, 1)) # ndarray (N3, 1)
        wh[0] = 0.04
        wh[1] = 0.05
        wh[2] = 0.06
        wh[3] = 0.07

        bh = np.zeros((N3, 1)) # ndarray (N3, 1)
        bh[0] = 0.08
        bh[1] = 0.09
        bh[2] = 0.1
        bh[3] = 0.11

        be_deriv = np.zeros((N1, 1)) # ndarray (N1, 1)
        be_deriv[0,0] = calculate_be_derivative(N3, wh, Z[0, 0], bh, err, 'sigmoid')
        be_deriv[1,0] = calculate_be_derivative(N3, wh, Z[1, 0], bh, err, 'sigmoid')
        be_deriv[2,0] = calculate_be_derivative(N3, wh, Z[2, 0], bh, err, 'sigmoid')

        d_deriv = np.zeros((N1, 1)) # ndarray (N1, 1)
        d_deriv[0, 0] = calculate_d_derivative(we[0,0], c[0,0], d[0,0], mapping_inputs, t[0,:], be_deriv[0,0], psi[0,0], 'gausD')
        d_deriv[1, 0] = calculate_d_derivative(we[1,0], c[1,0], d[1,0], mapping_inputs, t[1,:], be_deriv[1,0], psi[1,0], 'gausD')
        d_deriv[2, 0] = calculate_d_derivative(we[2,0], c[2,0], d[2,0], mapping_inputs, t[2,:], be_deriv[2,0], psi[2,0], 'gausD')

        d_deriv_gausD = np.zeros((N1, 1)) # ndarray (N1, 1)
        d_deriv_gausD[0,:] = 6.2134
        d_deriv_gausD[1,:] = 6.6168
        d_deriv_gausD[2,:] = 2.2839 
        
        # gausD
        self.assertEqual(np.allclose(d_deriv, d_deriv_gausD, rtol=1e-02), True) 
        
        ############################### morlet #####################################################
        c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, 'morlet')

        Z = create_mapping_nodes(we, psi, be, N1)

        # morlet
        d_deriv = np.zeros((N1, 1)) # ndarray (N1, 1)
        d_deriv[0, 0] = calculate_d_derivative(we[0,0], c[0,0], d[0,0], mapping_inputs, t[0,:], be_deriv[0,0], psi[0,0], 'morlet')
        d_deriv[1, 0] = calculate_d_derivative(we[1,0], c[1,0], d[1,0], mapping_inputs, t[1,:], be_deriv[1,0], psi[1,0], 'morlet')
        d_deriv[2, 0] = calculate_d_derivative(we[2,0], c[2,0], d[2,0], mapping_inputs, t[2,:], be_deriv[2,0], psi[2,0], 'morlet')

        d_deriv_morlet = np.zeros((N1, 1)) # ndarray (N1, 1)
        d_deriv_morlet[0,:] = -26.8533
        d_deriv_morlet[1,:] = 8.698
        d_deriv_morlet[2,:] = -0.6422
        
        # morlet
        self.assertEqual(np.allclose(d_deriv, d_deriv_morlet, rtol=1e-03), True)
        
        ############################### Mexican hat #####################################################
        c, psi, sum_x_minus_t_square = radial_wavelet(mapping_inputs, d, t, N1, no_features, 'mexHat')

        Z = create_mapping_nodes(we, psi, be, N1)

        d_deriv = np.zeros((N1, 1)) # ndarray (N1, 1)
        d_deriv[0, 0] = calculate_d_derivative(we[0,0], c[0,0], d[0,0], mapping_inputs, t[0,:], be_deriv[0,0], psi[0,0], 'mexHat')
        d_deriv[1, 0] = calculate_d_derivative(we[1,0], c[1,0], d[1,0], mapping_inputs, t[1,:], be_deriv[1,0], psi[1,0], 'mexHat')
        d_deriv[2, 0] = calculate_d_derivative(we[2,0], c[2,0], d[2,0], mapping_inputs, t[2,:], be_deriv[2,0], psi[2,0], 'mexHat')

        d_deriv_mexHat = np.zeros((N1, 1)) # ndarray (N1, 1)
        d_deriv_mexHat[0,:] = -5.432275
        d_deriv_mexHat[1,:] = 6.01425
        d_deriv_mexHat[2,:] = 4.346904
        
        self.assertEqual(np.allclose(d_deriv, d_deriv_mexHat, rtol=1e-03), True)       
    
    def test_create_enhancement_nodes(self):
        
        no_features = 5 # n
        no_neurons = 3 # m              
        
        bh = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        bh[0] = 0.1
        bh[1] = 0.2
        bh[2] = 0.3  
        
        wh = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        wh[0] = 4
        wh[1] = 5
        wh[2] = 6 
        
        x = np.zeros((no_features, 1)) # ndarray (n, 1)
        x[0] = 0.0001
        x[1] = 0.0002
        x[2] = 0.0003
        x[3] = 0.0004
        x[4] = 0.0005
        
        H = create_enhancement_nodes(no_neurons, x, wh, bh, 'sigmoid') # ndarray (m, 1)
        
        y = np.zeros((no_neurons, 1)) # ndarray (m, 1)
        y[0] = 0.5264
        y[1] = 0.5516
        y[2] = 0.5766
        
        self.assertEqual(np.allclose(H, y, rtol=1e-03), True) 
        
    def test_update_parameter(self):
        #Probar en el caso de a y de los normales
        x = np.asarray([1, 2, 3]) #(n,) ndarray
        x_deriv = np.asarray([4, 5, 6]) #(n,) ndarray
        y = np.asarray([-1, -0.5, 0])
        
        self.assertEqual(np.allclose(update_parameter(x, x_deriv, 0.5), y), True) #(n,) ndarray
        
        x = np.transpose(np.asmatrix([1, 2, 3])) #(n,1) matrix
        x_deriv = np.transpose(np.asmatrix([4, 5, 6])) #(n,1) matrix
        y = np.transpose(np.asmatrix([-1, -0.5, 0]))
        
        self.assertEqual(np.allclose(update_parameter(x, x_deriv, 0.5), y), True) #(n,1) matrix
        
        x = np.transpose(np.asmatrix([1, 2, 3])) #(n,1) matrix
        x_deriv = np.asarray(np.transpose(np.asmatrix([4, 5, 6]))) #(n,1) ndarray
        y = np.transpose(np.asmatrix([-1, -0.5, 0]))
        
        self.assertEqual(np.allclose(update_parameter(x, x_deriv, 0.5), y), True) #(n,1) matrix
        
        x = np.asarray(np.transpose(np.asmatrix([1, 2, 3]))) #(n,1) ndarray
        x_deriv = np.asarray(np.transpose(np.asmatrix([4, 5, 6]))) #(n,1) ndarray
        y = np.asarray(np.transpose(np.asmatrix([-1, -0.5, 0])))
        
        self.assertEqual(np.allclose(update_parameter(x, x_deriv, 0.5), y), True) #(n,1) ndarray
   
        
    def test_sigmoid(self):
        self.assertEqual(np.allclose(sigmoid(7), 0.9991, rtol=1e-04), True) # One value
        
        x = sigmoid(np.asarray([1,2,3]))
        y = np.asarray([0.7311, 0.8808, 0.9526])
        
        self.assertEqual(np.allclose(x, y, rtol=1e-03), True) # An ndarray
    
    def test_derivative_sigmoid(self):
        self.assertEqual(np.allclose(derivative_sigmoid(7), 0.00091022, rtol=1e-08), True) # One value
        
        x = derivative_sigmoid(np.asarray([1,2,3]))
        y = np.asarray([0.1966, 0.105, 0.0452])
        
        self.assertEqual(np.allclose(x, y, rtol=1e-03), True) # An ndarray
            
    def test_tanh(self):
        #phi(x, 'tanh') es lo mismo que escribir tanh(x) pero permite probar dos funciones a la vez
        self.assertEqual(np.allclose(phi(7, 'tanh'), 0.9999, rtol=1e-04), True) # One value
        
        x = phi(np.asarray([1,2,3]), 'tanh')
        y = np.asarray([0.76159, 0.9640, 0.99505])
        
        self.assertEqual(np.allclose(x, y, rtol=1e-03), True) # An ndarray
    
    def test_derivative_tanh(self):
        #phi_derivative(x, 'tanh') es lo mismo que escribir derivative_tanh(x) pero permite probar dos funciones a la vez
        y = phi_derivative(-0.5, 'tanh')
        self.assertEqual(np.allclose(y, 0.78644, rtol=1e-03), True) # One value
        
        x = phi_derivative(np.asarray([1,2,3]), 'tanh')
        y = np.asarray([0.41997, 0.07065, 0.00986])
        
        self.assertEqual(np.allclose(x, y, rtol=1e-03), True) # An ndarray

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)