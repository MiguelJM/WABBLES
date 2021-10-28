'''
 This functions Returns a number based on a label
'''
def Set_Av_Class(x):
    ''' Used to choose which integer value will be assigned to this object according to autovetter class
        0 = PC
        1 = AFP
        2 = NTP
        3 = UNK
    '''
    return {
        'PC': 0,
        'AFP': 1,
        'NTP': 2,
        'UNK': 3
    }.get(x, -1)    # -1 is default if x not found


'''
    Light Curve (LC) Class that contains the data structure of a LC
'''
class LightCurve:
    ''' This class contains every component of a light curve that will be needed '''
    kepID = 0 #ID of the planet
    t0 = 0 #Mid transit time
    t_duration = 0 #Duration of the transit
    t_period = 0 #Period of the transit
    av_class = 0 #Class predicted by the autovetter: 0 = PC, 1 = AFP, 2 = NTP, 3 = UNK
    flux = list() #Contains the concatenated raw light flux 
    time = list() #Contains the concatenated raw time lectures
    limb_darkening_model = ""  #Contains the transit signal without noise
    # Simulated noise parameters
    phi = 0
    wave_amp = 0
    wave_perd = 0
    PA = 0
    PW = 0
    sig_tol = 0
    
    def __init__(self, kID, t_0, t_dur, t_per, av_cl, limb_dark, lc_no):
        self.kepID = kID
        self.t0 = t_0
        self.t_duration = t_dur 
        self.t_period = t_per       
        self.av_class = int(av_cl)
        self.limb_darkening_model = limb_dark
        self.time = list()
        self.flux = list()
        print("LC added from {} added with class {} and a limb model {}. {}".format(kID, self.av_class, limb_dark, lc_no))
        
    #Converts class int to string
    def Get_Class(self): 
        x = self.av_class
        return {
            0: 'PC',
            1: 'AFP',
            2: 'NTP',
            3: 'UNK'
        }.get(x, 'UNK')    # UNK is default if x not found
        