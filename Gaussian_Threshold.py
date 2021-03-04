import numpy as np
from numpy import logical_and
def One_Sided_Gaussian(input_hologram,standard_deviation,direction,begin_timer,time_slices):
    print(f'[Time Slice #{time_slices+1}] Started Thresholding ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    holo_mean = np.mean(input_hologram)
    holo_std = np.std(input_hologram)
    upper_cutoff = holo_mean + standard_deviation*holo_std
    lower_cutoff = holo_mean - standard_deviation*holo_std
    Thresholded_Hologram = np.zeros(input_hologram.shape)
    if direction == 'Positive':
        Thresholded_Hologram[input_hologram>=upper_cutoff] = input_hologram[input_hologram>=upper_cutoff]
    if direction == 'Negative':
        Thresholded_Hologram[input_hologram<=lower_cutoff] = input_hologram[input_hologram<=lower_cutoff]
    
    return Thresholded_Hologram
    
def Two_Sided_Gaussian(input_hologram,standard_deviation,begin_timer,time_slices):
    print(f'[Time Slice #{time_slices+1}] Started Thresholding ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    holo_mean = np.mean(input_hologram)
    holo_std = np.std(input_hologram)
    upper_cutoff = holo_mean + standard_deviation*holo_std
    lower_cutoff = holo_mean - standard_deviation*holo_std
    Thresholded_Hologram = np.zeros(input_hologram.shape)
    Thresholded_Hologram[input_hologram>=upper_cutoff] = input_hologram[input_hologram>=upper_cutoff]
    Thresholded_Hologram[input_hologram<=lower_cutoff] = input_hologram[input_hologram<=lower_cutoff]
    
    return Thresholded_Hologram