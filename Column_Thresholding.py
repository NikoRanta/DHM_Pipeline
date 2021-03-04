import numpy as np
from numpy import logical_and
import timeit
from skimage import io
import os
from os import listdir
from os.path import isfile, join
import sys
from exif import Image



def Locate_Columns(input_hologram,Standard_Deviation_First_Threshold,Begin_Time,time_slices):
    
    #input_hologram = np.array(io.imread(filePath)).astype('<f4')
    #print(f'Loaded hologram ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    input_mean = np.mean(input_hologram)
    input_std = np.std(input_hologram)
    cutoff = input_mean+Standard_Deviation_First_Threshold*input_std

    array_data = np.zeros(input_hologram.shape)

    Locations = np.argwhere(input_hologram>=cutoff)
    for x in range(len(Locations)):
        array_data[:,Locations[x][1]-5:Locations[x][1]+5,Locations[x][2]-5:Locations[x][2]+5] = 1
            
        

    print(f'[Time Slice #{time_slices+1}] Finished Column Thresholding  ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')

    
    return array_data

def Identify_Columns(array_data,input_hologram,Percentage_Cutoff_Column_Threshold,Standard_Deviation_First_Threshold,Minimum_Pixel_Count_For_Column,Begin_Time,time_slices):
    column_data = np.copy(array_data[0,:,:])
    array_data = []

    non_zeros = np.argwhere(column_data!=0)
    blob_identifier = np.zeros((non_zeros.shape[0],3))
    blob_identifier[:,:2] = non_zeros
    non_zeros_remaining = np.copy(non_zeros)
    Locations_To_Still_Check = []
    Locations_Deleted = []

    order_of_checks = np.zeros((8,2))
    order_of_checks[0,:] = np.array([-1,-1])
    order_of_checks[1,:] = np.array([-1,0])
    order_of_checks[2,:] = np.array([-1,1])
    order_of_checks[3,:] = np.array([0,-1])
    order_of_checks[4,:] = np.array([0,1])
    order_of_checks[5,:] = np.array([1,-1])
    order_of_checks[6,:] = np.array([1,0])
    order_of_checks[7,:] = np.array([1,1])
    blob_identification = 0
    blob_identifier[0,2] = 1


    while len(np.argwhere(blob_identifier[:,2] == 0)) > 0:
        blob_identification += 1
        print(f'[Time Slice #{time_slices+1}] Building blob #{blob_identification}  ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
        blob_starting_check_position = np.argwhere(blob_identifier[:,2]==0)[0][0]
        for x in range(8):
            if len(np.argwhere(logical_and(blob_identifier[:,0]==blob_identifier[blob_starting_check_position][0]+order_of_checks[x,0],blob_identifier[:,1]==blob_identifier[blob_starting_check_position][1]+order_of_checks[x,1]))) != 0:
                if blob_identifier[np.argwhere(logical_and(blob_identifier[:,0]==blob_identifier[blob_starting_check_position][0]+order_of_checks[x,0],blob_identifier[:,1]==blob_identifier[blob_starting_check_position][1]+order_of_checks[x,1]))[0][0],2]==0:
                    Locations_To_Still_Check.append(np.argwhere(logical_and(blob_identifier[:,0]==blob_identifier[blob_starting_check_position][0]+order_of_checks[x,0],blob_identifier[:,1]==blob_identifier[blob_starting_check_position][1]+order_of_checks[x,1]))[0][0])
                    blob_identifier[np.argwhere(logical_and(blob_identifier[:,0]==blob_identifier[blob_starting_check_position][0]+order_of_checks[x,0],blob_identifier[:,1]==blob_identifier[blob_starting_check_position][1]+order_of_checks[x,1]))[0][0],2] = blob_identification
        
        while(len(Locations_To_Still_Check)>0):
            for x in range(8):
                if len(np.argwhere(logical_and(blob_identifier[:,0]==blob_identifier[Locations_To_Still_Check[0]][0]+order_of_checks[x,0],blob_identifier[:,1]==blob_identifier[Locations_To_Still_Check[0]][1]+order_of_checks[x,1]))) != 0:
                    if blob_identifier[np.argwhere(logical_and(blob_identifier[:,0]==blob_identifier[Locations_To_Still_Check[0]][0]+order_of_checks[x,0],blob_identifier[:,1]==blob_identifier[Locations_To_Still_Check[0]][1]+order_of_checks[x,1]))[0][0],2]==0:
                        Locations_To_Still_Check.append(np.argwhere(logical_and(blob_identifier[:,0]==blob_identifier[Locations_To_Still_Check[0]][0]+order_of_checks[x,0],blob_identifier[:,1]==blob_identifier[Locations_To_Still_Check[0]][1]+order_of_checks[x,1]))[0][0])
                        blob_identifier[np.argwhere(logical_and(blob_identifier[:,0]==blob_identifier[Locations_To_Still_Check[0]][0]+order_of_checks[x,0],blob_identifier[:,1]==blob_identifier[Locations_To_Still_Check[0]][1]+order_of_checks[x,1]))[0][0],2] = blob_identification
            Locations_To_Still_Check.remove(Locations_To_Still_Check[0])

    blob_library = []
    blob_mean = []
    blob_std = []
    blob_max = []
    column_cutoff_max = []
    Cutoff_Blob_Identifier = np.zeros(input_hologram.shape).astype('<f4')
    Cutoff_Blob_Identifier_Layer = np.zeros((input_hologram.shape[1],input_hologram.shape[2]))
    for x in range(blob_identification+1):
        if x!=0:
            print(f'[Time Slice #{time_slices+1}] Indexing blob #{x}  ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
            blob_library.append(input_hologram[:,blob_identifier[blob_identifier[:,2]==x][:,0].astype(int),blob_identifier[blob_identifier[:,2]==x][:,1].astype(int)])
            blob_mean.append(np.mean(input_hologram[:,blob_identifier[blob_identifier[:,2]==x][:,0].astype(int),blob_identifier[blob_identifier[:,2]==x][:,1].astype(int)]))
            blob_std.append(np.std(input_hologram[:,blob_identifier[blob_identifier[:,2]==x][:,0].astype(int),blob_identifier[blob_identifier[:,2]==x][:,1].astype(int)]))
            blob_max.append(np.max(input_hologram[:,blob_identifier[blob_identifier[:,2]==x][:,0].astype(int),blob_identifier[blob_identifier[:,2]==x][:,1].astype(int)]))
            
            cutoff_max = Percentage_Cutoff_Column_Threshold*np.max((input_hologram[:,blob_identifier[blob_identifier[:,2]==x][:,0].astype(int),blob_identifier[blob_identifier[:,2]==x][:,1].astype(int)]))
            column_cutoff_max.append(cutoff_max)
            
            Cutoff_Blob_Identifier_Layer[blob_identifier[blob_identifier[:,2]==x][:,0].astype(int),blob_identifier[blob_identifier[:,2]==x][:,1].astype(int)] = x


    Cutoff_Blob_Identifier[:,:,:] =  Cutoff_Blob_Identifier_Layer
    Column_Removal_Slice = np.copy(Cutoff_Blob_Identifier[0,:,:])
    Columns_Removed_Check = 0
    for x in range(blob_identification+1):
        if x != 0:
            if np.count_nonzero(Column_Removal_Slice==x) < Minimum_Pixel_Count_For_Column:
                print(f'[Time Slice #{time_slices+1}] Removed column for blob #{x} ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
                Cutoff_Blob_Identifier[Cutoff_Blob_Identifier==x] = 0
                if Columns_Removed_Check == 0:
                    Columns_Removed_Check = 1

    if Columns_Removed_Check == 0:
        print(f'[Time Slice #{time_slices+1}] No columns were removed below a pixel count of {Minimum_Pixel_Count_For_Column} ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    
    
    Blobs_Found = np.zeros(input_hologram.shape).astype('<f4')
    Blobs_Found_Numbers = []
    
    Column_Removal_Slice = np.copy(Cutoff_Blob_Identifier[0,:,:])
    for x in range(blob_identification+1):
        if x!=0:
            if np.count_nonzero(Column_Removal_Slice==x) > 0:
                print(f'[Time Slice #{time_slices+1}] Thresholding Column #{x} ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
                Cutoff_Blob_Identifier_Temp = np.copy(Cutoff_Blob_Identifier)
                Cutoff_Blob_Identifier_Temp[Cutoff_Blob_Identifier_Temp!=x] = 0
                Blobs_Found[logical_and(Cutoff_Blob_Identifier_Temp==x,input_hologram>=column_cutoff_max[x-1])] = input_hologram[logical_and(Cutoff_Blob_Identifier_Temp==x,input_hologram>=column_cutoff_max[x-1])]

    print(f'[Time Slice #{time_slices+1}] Done with unbuffered array ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    Cutoff_Blob_Identifier = []
    Cutoff_Blob_Identifier_Layer = []    
    return Blobs_Found

def Column_Thresholding(input_hologram,Standard_Deviation_First_Threshold,Percentage_Cutoff_Column_Threshold,Minimum_Pixel_Count_For_Column,Begin_Time,time_slices):
    percentage_used = str(int(np.round(Percentage_Cutoff_Column_Threshold*100,2)))
    array_data = Locate_Columns(input_hologram,Standard_Deviation_First_Threshold,Begin_Time,time_slices)
    Blobs_Found = Identify_Columns(array_data,input_hologram,Percentage_Cutoff_Column_Threshold,Standard_Deviation_First_Threshold,Minimum_Pixel_Count_For_Column,Begin_Time,time_slices)
    
    print(f'[Time Slice #{time_slices+1}] Done thresholding ({np.round(timeit.default_timer() - Begin_Time,3)} seconds)')
    return Blobs_Found
