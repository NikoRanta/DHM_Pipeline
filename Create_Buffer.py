import numpy as np
import timeit

def Buffering_Array(Thresholded_Hologram,Initial_Hologram,Pixel_Size,Buffer_Size,Begin_Time,time_slices):
    buffer_elements_x = np.int(np.floor(Buffer_Size[0]/Pixel_Size[0]))
    buffer_elements_y = np.int(np.floor(Buffer_Size[1]/Pixel_Size[1]))
    buffer_elements_z = np.int(np.floor(Buffer_Size[2]/Pixel_Size[2]))
    threshold_array_time_slice_region_of_interest = np.copy(Thresholded_Hologram)
    Thresholded_Hologram = []
    buffered_array = np.zeros(Initial_Hologram.shape,'<f4')
    threshold_array_time_slice_inside_buffer = np.zeros(Initial_Hologram.shape).astype(int)
    threshold_array_time_slice_inside_buffer[buffer_elements_z:Initial_Hologram.shape[0]-buffer_elements_z,buffer_elements_y:Initial_Hologram.shape[1]-buffer_elements_y,buffer_elements_x:Initial_Hologram.shape[2]-buffer_elements_x] = 1
    threshold_array_time_slice_region_of_interest[threshold_array_time_slice_inside_buffer != 1] = 0

    threshold_array_time_slice_non_zero = np.argwhere(threshold_array_time_slice_region_of_interest!=0)

    print(f'[Time Slice #{time_slices+1}] Buffering the array ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    
    timer_check = 0
    timer_check_check = 0

    for non_zero in range(len(threshold_array_time_slice_non_zero[:,0])):
        if non_zero >0:
            if non_zero%(np.floor(0.25*len(threshold_array_time_slice_non_zero[:,0])))==0:
                if timer_check == 0:
                    #print(f'{np.round((np.floor(0.25*len(threshold_array_time_slice_non_zero[:,0]))/non_zero,1))*25} percent done buffering')
                    timer_check = 1
            if timer_check == 1:
                if non_zero%(np.floor(0.25*len(threshold_array_time_slice_non_zero[:,0])))!=0:
                    timer_check = 0
                
        buffered_array[threshold_array_time_slice_non_zero[:,0][non_zero]-buffer_elements_z:threshold_array_time_slice_non_zero[:,0][non_zero]+buffer_elements_z+1,threshold_array_time_slice_non_zero[:,1][non_zero]-buffer_elements_y:threshold_array_time_slice_non_zero[:,1][non_zero]+buffer_elements_y+1,threshold_array_time_slice_non_zero[:,2][non_zero]-buffer_elements_x:threshold_array_time_slice_non_zero[:,2][non_zero]+buffer_elements_x+1] = Initial_Hologram[threshold_array_time_slice_non_zero[:,0][non_zero]-buffer_elements_z:threshold_array_time_slice_non_zero[:,0][non_zero]+buffer_elements_z+1,threshold_array_time_slice_non_zero[:,1][non_zero]-buffer_elements_y:threshold_array_time_slice_non_zero[:,1][non_zero]+buffer_elements_y+1,threshold_array_time_slice_non_zero[:,2][non_zero]-buffer_elements_x:threshold_array_time_slice_non_zero[:,2][non_zero]+buffer_elements_x+1]

    print(f'[Time Slice #{time_slices+1}] Done buffering the array ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    
    return buffered_array

'''
Thresholding - 2046 (Deriv taken)
Initial - 2048

'''   