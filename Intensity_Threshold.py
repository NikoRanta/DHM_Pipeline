import numpy as np
from numpy import logical_and
import timeit


def Intensity_Threshold(Threshold_Method,input_hologram,lower_bound,upper_bound,Begin_Time,time_slices):
    print(f'[Time Slice #{time_slices+1}] Started Thresholding ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    Locations_of_Interest = logical_and(input_hologram>=lower_bound,input_hologram<=upper_bound)
    Thresholded_Hologram = np.zeros(Locations_of_Interest.shape)
    
    if Threshold_Method == 'Intensity Window - Binary':
        Thresholded_Hologram[Locations_of_Interest] = 1
    if Threshold_Method == 'Intensity Window - Non-Binary':
        Thresholded_Hologram[Locations_of_Interest] = input_hologram[Locations_of_Interest]
    
    return Thresholded_Hologram