import numpy as np
import timeit
from skimage import io
import os
from os import listdir
from os.path import isfile, join, isdir
import sys
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import *
import time
#from Preview_Window import *
from Column_Thresholding import *
from Derivative import *
from Order_Reconstruction_Files import *
from Intensity_Threshold import *
from Gaussian_Threshold import *
from Create_Buffer import *


    
def PreProcessing_Data(processing_variables):
    #Order of variables
        #Input directory        [0]
        #Deriv status           [1]
        #Deriv Order            [2]
        #Deriv Output           [3]
        #Pixel Size             [4]
        #Buffer Size            [5]
        #Threshold status       [6]
        #Threshold method       [7]
        #Threshold variables    [8]
        #Finished Output        [9]
    #print('success')
    for var in processing_variables:
        print(var)
        pass
        
    
    begin_timer = timeit.default_timer()
    folderPath_Recon_Input = processing_variables[0][0]
    Deriv_Status = processing_variables[0][1]
    Deriv_Order = processing_variables[0][2] #Order of axis is (z,x,y), that is why x uses Deriv_Order[1]
    folderPath_Deriv_Output = processing_variables[0][3]
    Threshold_Status = processing_variables[0][4]
    Pixel_Size = processing_variables[0][5]
    Buffer_Size = processing_variables[0][6]
    Threshold_Method = processing_variables[0][7]
    Threshold_Variables = processing_variables[1]
    Output_Folder = processing_variables[2]
    Edge_Removal = processing_variables[3]
    
    
    Organized_Files = Order_Holograms_After_Reconstruction(folderPath_Recon_Input) #(z,t)
    Shape_Of_Hologram = io.imread(Organized_Files[0,0])
    
    maximum_time_and_z_slices = 5000
    
    #maximum_time_slices_allowed = 100
    maximum_time_slices_allowed = int(np.floor(maximum_time_and_z_slices/Organized_Files.shape[1]))
    
    Number_of_Holograms_Needed = int(np.ceil(Organized_Files.shape[1]/maximum_time_slices_allowed))
    save_points = np.zeros(Number_of_Holograms_Needed)
    
    
    if Number_of_Holograms_Needed > 1:
        for x in range(Number_of_Holograms_Needed-1):
            save_points[x] = save_points[x-1]+maximum_time_slices_allowed
        save_points[Number_of_Holograms_Needed-1] = save_points[Number_of_Holograms_Needed-2] + Organized_Files.shape[1]%maximum_time_slices_allowed
        Finished_Hologram = np.zeros((maximum_time_slices_allowed,Organized_Files.shape[0]-2*Deriv_Order[2],Shape_Of_Hologram.shape[0]-2*Deriv_Order[0],Shape_Of_Hologram.shape[1]-2*Deriv_Order[1]),'<f4')
    if Number_of_Holograms_Needed == 1:
        save_points[0] = Organized_Files.shape[1]%maximum_time_slices_allowed
        Finished_Hologram = np.zeros((Organized_Files.shape[1],Organized_Files.shape[0]-2*Deriv_Order[2],Shape_Of_Hologram.shape[0]-2*Deriv_Order[0],Shape_Of_Hologram.shape[1]-2*Deriv_Order[1]),'<f4')
    Identify_Region_Of_Caring = np.zeros((Organized_Files.shape[0],Shape_Of_Hologram.shape[0],Shape_Of_Hologram.shape[1]),np.int8)
    Identify_Region_Of_Caring[:,Edge_Removal:Shape_Of_Hologram.shape[0]-Edge_Removal,Edge_Removal:Shape_Of_Hologram.shape[1]-Edge_Removal] = 1
    
    holograms_made = 0
    t_slice_counter = 0
    first_slice = 1
    for time_slices in range(Organized_Files.shape[1]):
        print(f'[Time Slice #{time_slices+1}] Starting time slice {time_slices+1} ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
        Initial_Hologram = np.zeros((Organized_Files.shape[0],Shape_Of_Hologram.shape[0],Shape_Of_Hologram.shape[1]),'<f4')
        
        for z_slices in range(Organized_Files.shape[0]):
            Initial_Hologram[Organized_Files.shape[0]-z_slices-1,:,:] = np.array(io.imread(Organized_Files[z_slices,time_slices])).astype('<f4')
        print(f'[Time Slice #{time_slices+1}] Array imported ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
        
        input_hologram = np.copy(Initial_Hologram)
        input_hologram[Identify_Region_Of_Caring == 0 ] = 0
        print(f'[Time Slice #{time_slices+1}] Finished Edge Removal ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
        
        if Deriv_Status == 'Yes':
            print(f'[Time Slice #{time_slices+1}] Taking Derivative ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
            input_hologram = Derivatives(Deriv_Order,input_hologram,begin_timer)
            print(f'[Time Slice #{time_slices+1}] Finished Derivative ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
            
        
        
        if folderPath_Deriv_Output != '(Optional)':
            if time_slices < 9:
                io.imsave(folderPath_Deriv_Output+'0000'+str(time_slices+1)+'.tif',input_hologram)
            if 9<= time_slices<99:
                io.imsave(folderPath_Deriv_Output+'000'+str(time_slices+1)+'.tif',input_hologram)
            if 99<=time_slices<999:
                io.imsave(folderPath_Deriv_Output+'00'+str(time_slices+1)+'.tif',input_hologram)
        print(f'[Time Slice #{time_slices+1}] Saved Derivative ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
        if Threshold_Status == 'Yes':
            if Threshold_Method == 'Column Thresholding':
                print(f'[Time Slice #{time_slices+1}] Starting Threshold Method: {Threshold_Method} ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
                Thresholded_Hologram = Column_Thresholding(input_hologram,Threshold_Variables[0].astype(float),Threshold_Variables[1].astype(float),Threshold_Variables[2].astype(float),begin_timer,time_slices)
            if Threshold_Method == 'Intensity Window - Binary' or Threshold_Method == 'Intensity Window - Non-Binary':
                print(f'[Time Slice #{time_slices+1}] Starting Threshold Method: {Threshold_Method} ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
                Thresholded_Hologram = Intensity_Threshold(Threshold_Method,input_hologram,Threshold_Variables[0].astype(float),Threshold_Variables[1].astype(float),begin_timer,time_slices)
            if Threshold_Method == 'One-sided Gaussian':
                print(f'[Time Slice #{time_slices+1}] Starting Threshold Method: {Threshold_Method} ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
                Thresholded_Hologram = One_Sided_Gaussian(input_hologram,Threshold_Variables[0].astype(float),Threshold_Variables[1].astype(float),begin_timer,time_slices)
            if Threshold_Method == 'Two-sided Gaussian':
                print(f'[Time Slice #{time_slices+1}] Starting Threshold Method: {Threshold_Method} ({np.round(timeit.default_timer() - begin_timer,3)} seconds)',time_slices)
                Thresholded_Hologram = Two_Sided_Gaussian(input_hologram,Threshold_Variables[0].astype(float),begin_timer)
            print(f'[Time Slice #{time_slices+1}] Finished Threshold Method: {Threshold_Method} ({np.round(timeit.default_timer() - begin_timer,3)} seconds)',time_slices)
            if Buffer_Size[0] > 0 or Buffer_Size[1] > 0 or Buffer_Size[2] > 0:
                if Deriv_Status == 'Yes':
                    Initial_Hologram_Size_Reduced = Initial_Hologram[Deriv_Order[2]:Initial_Hologram.shape[0]-Deriv_Order[2],Deriv_Order[0]:Initial_Hologram.shape[1]-Deriv_Order[0],Deriv_Order[1]:Initial_Hologram.shape[2]-Deriv_Order[1]]
                    Finished_Hologram_Time_Slice = np.zeros((input_hologram.shape[0],input_hologram.shape[1],input_hologram.shape[2]),'<f4')
                    Finished_Hologram_Time_Slice = Buffering_Array(Thresholded_Hologram,Initial_Hologram_Size_Reduced,Pixel_Size,Buffer_Size,begin_timer,time_slices)
                if Deriv_Status == 'No':
                    Finished_Hologram_Time_Slice = np.zeros((input_hologram.shape[0],input_hologram.shape[1],input_hologram.shape[2]),'<f4')
                    Finished_Hologram_Time_Slice = Buffering_Array(Thresholded_Hologram,Initial_Hologram,Pixel_Size,Buffer_Size,begin_timer,time_slices)
            if Buffer_Size[0] == 0 and Buffer_Size[1] == 0 and Buffer_Size[2] == 0:
                Finished_Hologram_Time_Slice = np.copy(Thresholded_Hologram)
        if Threshold_Status == 'No':
            Finished_Hologram_Time_Slice = input_hologram
        
        Finished_Hologram[t_slice_counter,:,:,:] = np.copy(Finished_Hologram_Time_Slice)
        Finished_Hologram_Time_Slice = []
        print(f'[Time Slice #{time_slices+1}] Finished Combining Time Slice with Full Array ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
        
        Testing = False
        if Testing:
            #save output
            sys.exit('Successfully Completed Test')
        
        
        if time_slices == save_points[holograms_made]-1:
            if Number_of_Holograms_Needed > 1:
                print(f'[Time Slice #{time_slices+1}] Maximum Time Slices Allowed in One Array - Saving Full Array ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
                io.imsave(Output_Folder+str(Deriv_Order)+'_Derivs_'+Threshold_Method+'_'+Threshold_Variables+'_'+'Slices_'+str(first_slice)+'_to_'+str(time_slices+1)+'.tif',Finished_Hologram)
                print(f'[Time Slice #{time_slices+1}] Finished Saving Full Array ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
        if (time_slices+1)%maximum_time_slices_allowed==0:
            holograms_made += 1
            first_slice += maximum_time_slices_allowed
        
    if Number_of_Holograms_Needed == 1:
        print(f'[Time Slice #{time_slices+1}] Finished Saving Full Array ({np.round(timeit.default_timer() - begin_timer,3)} seconds)')
        io.imsave(Output_Folder+str(Deriv_Order)+' Derivs '+Threshold_Method+' '+Threshold_Variables[0]+' '+Threshold_Variables[1]+' '+Threshold_Variables[2]+'.tif',Finished_Hologram)

if __name__ == '__main__':
    top = tk.Tk()
    
    myframe = Frame(top)
    myframe.pack(fill=BOTH, expand=YES)
    canvas1 = tk.Canvas(myframe, width = 1425, height = 550, relief = 'raised')
    canvas1.pack(fill=BOTH, expand=YES)
    
    

    def Button():
        pass
    def Input_Directory():
        Input_Directory_Text.set('')
        Input_Direction_Chosen = filedialog.askdirectory(parent=top,title='Choose a directory')
        Input_Directory_Text_Entry.insert(END,Input_Direction_Chosen)
    def Output_Directory():
        if Output_Directory_Text != '':
            Output_Directory_Text.set('')
            Output_Directory_Chosen = filedialog.askdirectory(parent=top,title='Choose a directory')
            Output_Directory_Text_Entry.insert(END,Output_Directory_Chosen)
        if Output_Directory_Text == '':
            Output_Directory_Chosen = filedialog.askdirectory(parent=top,title='Choose a directory')
            Output_Directory_Text_Entry.insert(END,Output_Directory_Chosen)
    def Final_Selection_Made():
        processing_variables = []
        processing_variables.append([Input_Directory_Text.get(),Deriv_Menu_Chosen.get(),np.array([int(Deriv_x_Value.get()),int(Deriv_y_Value.get()),int(Deriv_z_Value.get())]),Deriv_Output_Directory_Text.get(),Threshold_Menu_Chosen.get(),np.array(Pixel_Size_Value.get().split(',')).astype(float),np.array(Buffer_Size_Value.get().split(',')).astype(float),Threshold_Method_Selection.get()])
        if Threshold_Menu_Chosen.get() == 'Yes':
            if Threshold_Method_Selection.get() == 'Column Thresholding':
                processing_variables.append(np.array([Column_Finding_Thresholding_Value.get(),Column_Thresholding_Value.get(),Column_Thresholding_Minimum_Size_Value.get()]))
            if Threshold_Method_Selection.get() == 'Intensity Window - Binary':
                processing_variables.append(np.array([Intensity_Window_Lower_Value.get(),Intensity_Window_Upper_Value.get()]))
            if Threshold_Method_Selection.get() == 'Intensity Window - Non-Binary':
                processing_variables.append(np.array([Intensity_Window_Lower_Value.get(),Intensity_Window_Upper_Value.get()]))
            if Threshold_Method_Selection.get() == 'One-sided Gaussian':
                processing_variables.append(np.array([Gaussian_Sided_Value.get(),Gaussian_Sided_Direction.get()]))
            if Threshold_Method_Selection.get() == 'Two-sided Gaussian':
                processing_variables.append(np.array([Gaussian_Sided_Value.get()]))
        if Threshold_Menu_Chosen.get() == 'No':
            processing_variables.append(0)
            processing_variables.append(0)
            processing_variables.append(0)
        processing_variables.append(Output_Directory_Text.get())
        if Edge_Removal_Value.get() != '(Optional)':
            processing_variables.append(np.array(Edge_Removal_Value.get()).astype(int))
        if Edge_Removal_Value.get() == '(Optional)':
            processing_variables.append(0)
            
        
        PreProcessing_Data(processing_variables)
    def Checking_For_Errors():
        if Initializing_Check.get() == 1:
            Errors_Found = np.zeros(18)
            Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Canvas = canvas1.create_window(450,100,window=Incorrect_Reconstruction_Input_Error_Message_Not_Valid)
            canvas1.delete(Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Canvas)
            Incorrect_Reconstruction_Input_Error_Symbol_Canvas = canvas1.create_window(25,77,window=Incorrect_Reconstruction_Input_Error_Symbol)
            canvas1.delete(Incorrect_Reconstruction_Input_Error_Symbol_Canvas)
            Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Canvas = canvas1.create_window(450,100,window=Incorrect_Reconstruction_Input_Error_Message_Missing_Folders)
            canvas1.delete(Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Canvas)
            Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Canvas = canvas1.create_window(450,100,window=Incorrect_Reconstruction_Input_Error_Message_Missing_Files)
            canvas1.delete(Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Canvas)
            Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Check.set(0)
            Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Check.set(0)
            Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check.set(0)
            if not isdir(Input_Directory_Text.get()):
                Errors_Found[0] = 1
                Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Check.set(1)
            if isdir(Input_Directory_Text.get()):
                Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Check.set(0)
                
            if Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Check.get() == 0:
                try:
                    Folder_Names = np.array([(f,float(f.name)) for f in os.scandir(Input_Directory_Text.get())])
                except ValueError:
                    Errors_Found[0] = 1
                    Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Check.set(1)
                else:
                    if len(Folder_Names) == 0:
                        Errors_Found[0] = 1
                        Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Check.set(1)
                    if len(Folder_Names) > 0:
                        Descending_Order = Folder_Names[(-Folder_Names[:,1]).argsort()]
                        Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Check.set(0)
                if Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Check.get() == 0: 
                    Folder_Names = np.array([(f,float(f.name)) for f in os.scandir(Input_Directory_Text.get())])
                    Descending_Order = Folder_Names[(-Folder_Names[:,1]).argsort()]
                    for folders in range(len(Descending_Order)):
                        try:
                            [[onlyfiles.split('.')[0],onlyfiles.split('.')[1]] for onlyfiles in listdir(Descending_Order[folders,0].path) if isfile(join(Descending_Order[folders,0],onlyfiles))]
                        except:
                            Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check.set(1)
                        else:
                            onlyfiles = [[onlyfiles.split('.')[0],onlyfiles.split('.')[1]] for onlyfiles in listdir(Descending_Order[folders,0].path) if isfile(join(Descending_Order[folders,0],onlyfiles))]
                            if len(onlyfiles) == 0:
                                Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check.set(1)
                            if len(onlyfiles)>0:
                                filename_check = np.zeros(len(onlyfiles))
                                filetype_check = np.zeros(len(onlyfiles)).astype(np.unicode_)
                                issues_found = 0
                                for x in range(len(onlyfiles)):
                                    proceed = 0
                                    try:
                                        filename_check[x] = onlyfiles[x][0]
                                    except ValueError:
                                        issues_found = 1
                                        Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check.set(1)
                                        break
                                    if onlyfiles[x][1] != 'tif':
                                        issues_found = 1
                                        Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check.set(1)
                                        break
                                if issues_found==1:
                                    Errors_Found[0] = 1
                                    Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check.set(1)
                                    break
                    if issues_found == 0:     
                        Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check.set(0)
                            
            if Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Check.get() == 1:
                Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Canvas = canvas1.create_window(450,100,window=Incorrect_Reconstruction_Input_Error_Message_Not_Valid)
                Incorrect_Reconstruction_Input_Error_Symbol_Canvas = canvas1.create_window(25,77,window=Incorrect_Reconstruction_Input_Error_Symbol)
            if Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Check.get() == 1:
                Incorrect_Reconstruction_Input_Error_Symbol_Canvas = canvas1.create_window(25,77,window=Incorrect_Reconstruction_Input_Error_Symbol)
                Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Canvas = canvas1.create_window(450,100,window=Incorrect_Reconstruction_Input_Error_Message_Missing_Folders)
            if Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check.get() == 1:
                Incorrect_Reconstruction_Input_Error_Symbol_Canvas = canvas1.create_window(25,77,window=Incorrect_Reconstruction_Input_Error_Symbol)
                Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Canvas = canvas1.create_window(450,100,window=Incorrect_Reconstruction_Input_Error_Message_Missing_Files)
                    
            if Deriv_Menu_Chosen.get() == 'Select One':
                Deriv_Choice_Check.set(1)
                Incorrect_Deriv_Choice_Error_Canvas = canvas1.create_window(25,105+Deriv_Menu_Offset.get()-82,window=Incorrect_Deriv_Choice_Error_Symbol)
                Errors_Found[1] = 1
            
            if Deriv_Menu_Chosen.get() == 'Yes':
                if Deriv_Choice_Check.get() == 1:
                    Deriv_Choice_Check.set(0)
                    Incorrect_Deriv_Choice_Error_Canvas = canvas1.create_window(240,105+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Choice_Error_Symbol)
                    canvas1.delete(Incorrect_Deriv_Choice_Error_Canvas)
                    
                try:
                    np.array(Deriv_x_Value.get()).astype(int)>=0
                except ValueError:
                    Deriv_x_check.set(1)
                else:
                    Deriv_x_check.set(0)
                    
                try:
                    np.array(Deriv_y_Value.get()).astype(int)>=0
                except ValueError:
                    Deriv_y_check.set(1)
                else:
                    Deriv_y_check.set(0)
                    
                try:
                    np.array(Deriv_z_Value.get()).astype(int)>=0
                except ValueError:
                    Deriv_z_check.set(1)
                else:
                    Deriv_z_check.set(0)
                if Deriv_x_check.get() == 1 or Deriv_y_check.get() == 1 or Deriv_z_check.get() == 1:
                    Errors_Found[2] = 1
                    Deriv_Order_Check.set(1)
                    Incorrect_Deriv_Order_Summation_Message_Canvas = canvas1.create_window(710,103+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Summation_Message)
                    canvas1.delete(Incorrect_Deriv_Order_Summation_Message_Canvas)
                    Incorrect_Deriv_Order_Error_Symbol_Canvas = canvas1.create_window(240,103+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Error_Symbol)
                    Incorrect_Deriv_Order_Error_Message_Canvas = canvas1.create_window(710,103+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Error_Message)
                    
                if Deriv_x_check.get() == 0:
                    if Deriv_y_check.get() == 0:
                        if Deriv_z_check.get() == 0:
                            if np.array(Deriv_x_Value.get()).astype(int) + np.array(Deriv_y_Value.get()).astype(int) + np.array(Deriv_z_Value.get()).astype(int) == 0:
                                if Deriv_Order_Check.get() == 0:
                                    Incorrect_Deriv_Order_Error_Symbol_Canvas = canvas1.create_window(240,103+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Error_Symbol)
                                    Incorrect_Deriv_Order_Error_Message_Canvas = canvas1.create_window(710,103+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Error_Message)
                                    canvas1.delete(Incorrect_Deriv_Order_Error_Message_Canvas)
                                    Incorrect_Deriv_Order_Summation_Message_Canvas = canvas1.create_window(710,103+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Summation_Message)
                            if np.array(Deriv_x_Value.get()).astype(int) + np.array(Deriv_y_Value.get()).astype(int) + np.array(Deriv_z_Value.get()).astype(int) > 0:
                                Incorrect_Deriv_Order_Error_Symbol_Canvas = canvas1.create_window(240,132+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Error_Symbol)
                                canvas1.delete(Incorrect_Deriv_Order_Error_Symbol_Canvas)
                                Incorrect_Deriv_Order_Error_Message_Canvas = canvas1.create_window(450,103+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Error_Message)
                                canvas1.delete(Incorrect_Deriv_Order_Error_Message_Canvas)
                                Incorrect_Deriv_Order_Summation_Message_Canvas = canvas1.create_window(710,103+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Order_Summation_Message)
                                canvas1.delete(Incorrect_Deriv_Order_Summation_Message_Canvas)
                                Deriv_Order_Check.set(0)

                                
            if Deriv_Output_Directory_Text.get() == '(Optional)' or Deriv_Output_Directory_Text.get() == '':
                if Deriv_Output_Check.get() == 1:
                    Incorrect_Deriv_Output_Error_Canvas = canvas1.create_window(24,158+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Output_Error_Symbol)
                    canvas1.delete(Incorrect_Deriv_Output_Error_Canvas)
                    Deriv_Output_Check.set(0)
                    Deriv_Output_Hold_Check.set('')
                    Deriv_Output_Prechecked_Check.set(0)
                    
            if Deriv_Output_Directory_Text.get() != '(Optional)' and Deriv_Output_Directory_Text.get() != '':
                if not isdir(Deriv_Output_Directory_Text.get()):
                    Deriv_Output_Check.set(1)
                    Incorrect_Deriv_Output_Error_Canvas = canvas1.create_window(24,158+Deriv_Menu_Offset.get()-82,window=Incorrect_Deriv_Output_Error_Symbol)
                    Incorrect_Deriv_Output_Error_Message_Canvas = canvas1.create_window(450,205,window=Incorrect_Deriv_Output_Error_Message)
                    Deriv_Output_Hold_Check.set('')
                    Deriv_Output_Prechecked_Check.set(0)
                    Errors_Found[3] = 1
                if isdir(Deriv_Output_Directory_Text.get()):
                    if Deriv_Output_Prechecked_Check.get() == 0:
                        Deriv_Output_Hold_Check.set(Deriv_Output_Directory_Text.get())
                        Deriv_Output_Prechecked_Check.set(1)
                        if Deriv_Output_Check.get() == 1:
                            Incorrect_Deriv_Output_Error_Canvas = canvas1.create_window(24,158+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Output_Error_Symbol)
                            canvas1.delete(Incorrect_Deriv_Output_Error_Canvas)
                            Incorrect_Deriv_Output_Error_Message_Canvas = canvas1.create_window(450,205,window=Incorrect_Deriv_Output_Error_Message)
                            canvas1.delete(Incorrect_Deriv_Output_Error_Message_Canvas)
                            Deriv_Output_Check.set(0)
            if Threshold_Menu_Chosen.get() == 'Yes':
                if Threshold_Method_Selection.get() == 'Column Thresholding':
                    Pixel_Buffer_Offset_Tracker.set(2)
                    if Threshold_Method_Check.get() == 1:
                        Threshold_Method_Check.set(0)
                        Incorrect_Threshold_Method_Error_Symbol_Canvas = canvas1.create_window(380,188+Deriv_Menu_Offset.get()-56,window=Incorrect_Threshold_Method_Error_Symbol)
                        canvas1.delete(Incorrect_Threshold_Method_Error_Symbol_Canvas)
                        
                    try:
                        np.array(Column_Finding_Thresholding_Value.get()).astype(int)>0
                    except ValueError:
                        Errors_Found[6] = 1
                        if Threshold_Column_Standard_Deviation_Check.get() == 0:
                            Threshold_Column_Standard_Deviation_Check.set(1)
                            Incorrect_Column_Threshold_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_SD_Symbol)
                            Incorrect_Column_Threshold_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_SD_Message)
                    else:
                        if not np.array(Column_Finding_Thresholding_Value.get()).astype(int)>0:
                            Errors_Found[6] = 1
                            if Threshold_Column_Standard_Deviation_Check.get() == 0:
                                Threshold_Column_Standard_Deviation_Check.set(1)
                                Incorrect_Column_Threshold_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_SD_Symbol)
                                Incorrect_Column_Threshold_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_SD_Message)
                                
                        if np.array(Column_Finding_Thresholding_Value.get()).astype(int)>0:
                            if Threshold_Column_Standard_Deviation_Check.get() == 1:
                                Errors_Found[6] = 0
                                Threshold_Column_Standard_Deviation_Check.set(0)
                                Incorrect_Column_Threshold_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_SD_Symbol)
                                canvas1.delete(Incorrect_Column_Threshold_SD_Symbol_Canvas)
                                Incorrect_Column_Threshold_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_SD_Message)
                                canvas1.delete(Incorrect_Column_Threshold_SD_Message_Canvas)
                    
                    try:
                        1 >= np.array(Column_Thresholding_Value.get()).astype(float) >= 0
                    except ValueError:
                        Errors_Found[7] = 1
                        if Threshold_Column_Percent_Max_Check.get() == 0:
                            Threshold_Column_Percent_Max_Check.set(1)
                            Incorrect_Column_Threshold_Percentage_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_Percentage_Symbol)
                            Incorrect_Column_Threshold_Percentage_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_Percentage_Message)
                    else:
                        if not 1 >= np.array(Column_Thresholding_Value.get()).astype(float) >= 0:
                            Errors_Found[7] = 1
                            if Threshold_Column_Percent_Max_Check.get() == 0:
                                Threshold_Column_Percent_Max_Check.set(1)
                                Incorrect_Column_Threshold_Percentage_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_Percentage_Symbol)
                                Incorrect_Column_Threshold_Percentage_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_Percentage_Message)
                        if 1 >= np.array(Column_Thresholding_Value.get()).astype(float) >= 0:
                            if Threshold_Column_Percent_Max_Check.get() == 1:
                                Errors_Found[7] = 0
                                Threshold_Column_Percent_Max_Check.set(1)
                                Incorrect_Column_Threshold_Percentage_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_Percentage_Symbol)
                                canvas1.delete(Incorrect_Column_Threshold_Percentage_Symbol_Canvas)
                                Incorrect_Column_Threshold_Percentage_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_Percentage_Message)
                                canvas1.delete(Incorrect_Column_Threshold_Percentage_Message_Canvas)
                    try:
                        np.array(Column_Thresholding_Minimum_Size_Value.get()).astype(int)>0
                    except ValueError:
                        Errors_Found[8] = 1
                        if Threshold_Column_Minimum_Pixel_Check.get() == 0:
                            Threshold_Column_Minimum_Pixel_Check.set(1)
                            Incorrect_Column_Minimum_Pixel_Symbol_Canvas = canvas1.create_window(240,301+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Minimum_Pixel_Symbol)
                            Incorrect_Column_Minimum_Pixel_Message_Canvas = canvas1.create_window(710,301+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Minimum_Pixel_Message)
                    else:
                        if not np.array(Column_Thresholding_Minimum_Size_Value.get()).astype(int)>0:
                            Errors_Found[8]=1
                            if Threshold_Column_Minimum_Pixel_Check.get() == 0:
                                Threshold_Column_Minimum_Pixel_Check.set(1)
                                Incorrect_Column_Minimum_Pixel_Symbol_Canvas = canvas1.create_window(240,301+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Minimum_Pixel_Symbol)
                                Incorrect_Column_Minimum_Pixel_Message_Canvas = canvas1.create_window(710,301+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Minimum_Pixel_Message)
                        if np.array(Column_Thresholding_Minimum_Size_Value.get()).astype(int)>0:
                            if Threshold_Column_Minimum_Pixel_Check.get() == 1:
                                Errors_Found[8]=0
                                Threshold_Column_Minimum_Pixel_Check.set(0)
                                Incorrect_Column_Minimum_Pixel_Symbol_Canvas = canvas1.create_window(240,301+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Minimum_Pixel_Symbol)
                                canvas1.delete(Incorrect_Column_Minimum_Pixel_Symbol_Canvas)
                                Incorrect_Column_Minimum_Pixel_Message_Canvas = canvas1.create_window(710,301+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Minimum_Pixel_Message)
                                canvas1.delete(Incorrect_Column_Minimum_Pixel_Message_Canvas)
                    
                if Threshold_Method_Selection.get() == 'Intensity Window - Binary' or Threshold_Method_Selection.get() == 'Intensity Window - Non-Binary':
                    Pixel_Buffer_Offset_Tracker.set(1)
                    if Threshold_Method_Check.get() == 1:
                        Threshold_Method_Check.set(0)
                        Incorrect_Threshold_Method_Error_Symbol_Canvas = canvas1.create_window(380,188+Deriv_Menu_Offset.get()-56,window=Incorrect_Threshold_Method_Error_Symbol)
                        canvas1.delete(Incorrect_Threshold_Method_Error_Symbol_Canvas)
                    try:
                        np.array(Intensity_Window_Lower_Value.get()).astype(float)
                    except ValueError:
                        Errors_Found[11] = 1
                        if Threshold_Intensity_Lower_Bound_Check.get() == 0:
                            Threshold_Intensity_Lower_Bound_Check.set(1)
                            Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Symbol)
                            Incorrect_Intensity_Window_Bound_Lower_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Message)
                    else:
                        if np.array(Intensity_Window_Lower_Value.get()).astype(float)<=0:
                            Errors_Found[11] = 1
                            if Threshold_Intensity_Lower_Bound_Check.get() == 0:
                                Threshold_Intensity_Lower_Bound_Check.set(1)
                                Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Symbol)
                                Incorrect_Intensity_Window_Bound_Lower_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Message)
                        if np.array(Intensity_Window_Lower_Value.get()).astype(float)>0:
                            if Threshold_Intensity_Lower_Bound_Check.get() == 1:
                                Threshold_Intensity_Lower_Bound_Check.set(0)
                                Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Symbol)
                                canvas1.delete(Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas)
                                Incorrect_Intensity_Window_Bound_Lower_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Message)
                                canvas1.delete(Incorrect_Intensity_Window_Bound_Lower_Message_Canvas)
                    try:
                        np.array(Intensity_Window_Upper_Value.get()).astype(float)
                    except ValueError:
                        Errors_Found[12] = 1
                        if Threshold_Intensity_Upper_Bound_Check.get() == 0:
                            Threshold_Intensity_Upper_Bound_Check.set(1)
                            Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Symbol)
                            Incorrect_Intensity_Window_Bound_Upper_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Message)
                    else:
                        if np.array(Intensity_Window_Upper_Value.get()).astype(float)<=0:
                            Errors_Found[12] = 1
                            if Threshold_Intensity_Upper_Bound_Check.get() == 0:
                                Threshold_Intensity_Upper_Bound_Check.set(1)
                                Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Symbol)
                                Incorrect_Intensity_Window_Bound_Upper_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Message)
                        if np.array(Intensity_Window_Upper_Value.get()).astype(float)>0:
                            if Threshold_Intensity_Upper_Bound_Check.get() == 1:
                                Threshold_Intensity_Upper_Bound_Check.set(0)
                                Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Symbol)
                                canvas1.delete(Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas)
                                Incorrect_Intensity_Window_Bound_Upper_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Message)
                    if Threshold_Intensity_Lower_Bound_Check.get() == 0:
                        if Threshold_Intensity_Upper_Bound_Check.get() == 0:
                            if np.array(Intensity_Window_Upper_Value.get()).astype(float) <= np.array(Intensity_Window_Lower_Value.get()).astype(float):
                                Errors_Found[13]=1
                                if Intensity_Window_Valid_Range_Check.get() == 0:
                                    Intensity_Window_Valid_Range_Check.set(1)
                                    if Threshold_Intensity_Upper_Bound_Check.get() == 0:
                                        Threshold_Intensity_Upper_Bound_Check.set(1)
                                        Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Symbol)
                                        Incorrect_Intensity_Window_Bound_Upper_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Message)
                                    if Threshold_Intensity_Lower_Bound_Check.get() == 0:
                                        Threshold_Intensity_Lower_Bound_Check.set(1)
                                        Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Symbol)
                                        Incorrect_Intensity_Window_Bound_Lower_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Message)
                            if np.array(Intensity_Window_Upper_Value.get()).astype(float) > np.array(Intensity_Window_Lower_Value.get()).astype(float):
                                if Intensity_Window_Valid_Range_Check.get() == 1:
                                    if Threshold_Intensity_Upper_Bound_Check.get() == 1:
                                        Threshold_Intensity_Upper_Bound_Check.set(0)
                                        Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Symbol)
                                        canvas1.delete(Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas)
                                        Incorrect_Intensity_Window_Bound_Upper_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Message)
                                        canvas1.delete(Incorrect_Intensity_Window_Bound_Upper_Message_Canvas)
                                    if Threshold_Intensity_Lower_Bound_Check.get() == 1:
                                        Threshold_Intensity_Lower_Bound_Check.set(0)
                                        Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Symbol)
                                        canvas1.delete(Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas)
                                        Incorrect_Intensity_Window_Bound_Lower_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Message)
                                        canvas1.delete(Incorrect_Intensity_Window_Bound_Lower_Message_Canvas)
                if Threshold_Method_Selection.get() == 'One-sided Gaussian':
                    Pixel_Buffer_Offset_Tracker.set(1)
                    if Threshold_Method_Check.get() == 1:
                        Threshold_Method_Check.set(0)
                        Incorrect_Threshold_Method_Error_Symbol_Canvas = canvas1.create_window(380,188+Deriv_Menu_Offset.get()-56,window=Incorrect_Threshold_Method_Error_Symbol)
                        canvas1.delete(Incorrect_Threshold_Method_Error_Symbol_Canvas)
                    try:
                        np.array(Gaussian_Sided_Value.get()).astype(float)
                    except ValueError:
                        Errors_Found[14]=1
                        if Threshold_Gaussian_Standard_Deviation_Check.get() == 0:
                            Threshold_Gaussian_Standard_Deviation_Check.set(1)
                            Incorrect_Gaussian_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Symbol)
                            Incorrect_Gaussian_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Message)
                    else:
                        if np.array(Gaussian_Sided_Value.get()).astype(float)>0:
                            if Threshold_Gaussian_Standard_Deviation_Check.get() == 1:
                                Threshold_Gaussian_Standard_Deviation_Check.set(0)
                                Incorrect_Gaussian_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Symbol)
                                canvas1.delete(Threshold_Gaussian_Standard_Deviation_Check)
                                Incorrect_Gaussian_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Message)
                                canvas1.delete(Incorrect_Gaussian_SD_Message_Canvas)
                        if np.array(Gaussian_Sided_Value.get()).astype(float)<=0:
                            if Threshold_Gaussian_Standard_Deviation_Check.get() == 0:
                                Threshold_Gaussian_Standard_Deviation_Check.set(1)
                                Incorrect_Gaussian_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Symbol)
                                Incorrect_Gaussian_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Message)
                    if Gaussian_Sided_Direction.get() == 'Select One':
                        Errors_Found[15]=1
                        if Threshold_Gaussian_Direction_Check.get() == 0:
                            Threshold_Gaussian_Direction_Check.set(1)
                            Incorrect_Gaussian_Select_Direction_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_Select_Direction_Symbol)
                    if Gaussian_Sided_Direction.get() != 'Select One':
                        if Threshold_Gaussian_Direction_Check.get() == 1:
                            Threshold_Gaussian_Direction_Check.set(0)
                            Incorrect_Gaussian_Select_Direction_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_Select_Direction_Symbol)
                            canvas1.delete(Incorrect_Gaussian_Select_Direction_Symbol_Canvas)
                if Threshold_Method_Selection.get() == 'Two-sided Gaussian':
                    Pixel_Buffer_Offset_Tracker.set(0)
                    if Threshold_Method_Check.get() == 1:
                        Threshold_Method_Check.set(0)
                        Incorrect_Threshold_Method_Error_Symbol_Canvas = canvas1.create_window(380,188+Deriv_Menu_Offset.get()-56,window=Incorrect_Threshold_Method_Error_Symbol)
                        canvas1.delete(Incorrect_Threshold_Method_Error_Symbol_Canvas)
                    try:
                        np.array(Gaussian_Sided_Value.get()).astype(float)
                    except ValueError:
                        Errors_Found[15]=1
                        if Threshold_Gaussian_Standard_Deviation_Check.get() == 0:
                            Threshold_Gaussian_Standard_Deviation_Check.set(1)
                            Incorrect_Gaussian_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Symbol)
                            Incorrect_Gaussian_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Message)
                    else:
                        if np.array(Gaussian_Sided_Value.get()).astype(float)>0:
                            if Threshold_Gaussian_Standard_Deviation_Check.get() == 1:
                                Threshold_Gaussian_Standard_Deviation_Check.set(0)
                                Incorrect_Gaussian_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Symbol)
                                canvas1.delete(Threshold_Gaussian_Standard_Deviation_Check)
                                Incorrect_Gaussian_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Message)
                                canvas1.delete(Incorrect_Gaussian_SD_Message_Canvas)
                        if np.array(Gaussian_Sided_Value.get()).astype(float)<=0:
                            if Threshold_Gaussian_Standard_Deviation_Check.get() == 0:
                                Threshold_Gaussian_Standard_Deviation_Check.set(1)
                                Incorrect_Gaussian_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Symbol)
                                Incorrect_Gaussian_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Message)
                if Threshold_Method_Selection.get() == 'Select One':
                    Errors_Found[5] = 1
                    if Threshold_Method_Check.get() == 0:
                        Threshold_Method_Check.set(1)
                        Incorrect_Threshold_Method_Error_Symbol_Canvas = canvas1.create_window(380,188+Deriv_Menu_Offset.get()-56,window=Incorrect_Threshold_Method_Error_Symbol)
                if Threshold_Method_Selection.get() != 'Select One':
                    Pixel_Buffer_Symbol_Locations = [273,301,328,355]
                    Edge_Removal_Symbol_Locations = [301,238,355,383]
                    Pixel_Offset_Value.set(Pixel_Buffer_Symbol_Locations[Pixel_Buffer_Offset_Tracker.get()]+Deriv_Menu_Offset.get()-56)
                    Buffer_Offset_Value.set(Pixel_Buffer_Symbol_Locations[Pixel_Buffer_Offset_Tracker.get()+1]+Deriv_Menu_Offset.get()-56)
                    Edge_Removal_Offset_Value.set(Pixel_Buffer_Symbol_Locations[Pixel_Buffer_Offset_Tracker.get()+1]+Deriv_Menu_Offset.get()-28)
                    if Pixel_Input_Check.get() == 1:
                        Pixel_Input_Check.set(0)
                        Incorrect_Pixel_Error_Symbol_Canvas = canvas1.create_window(240,+Pixel_Offset_Value.get(),window=Incorrect_Pixel_Error_Symbol)
                        canvas1.delete(Incorrect_Pixel_Error_Symbol_Canvas)
                    if Pixel_Message_Check.get() == 1:
                        Pixel_Message_Check.set(0)
                        Incorrect_Pixel_Format_Display_Text_Canvas = canvas1.create_window(710,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Format_Display_Text)
                        canvas1.delete(Incorrect_Pixel_Format_Display_Text_Canvas)
                    if Buffer_Input_Check.get() == 1:
                        Buffer_Input_Check.set(0)
                        Incorrect_Buffer_Error_Symbol_Canvas = canvas1.create_window(240,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Error_Symbol)
                        canvas1.delete(Incorrect_Buffer_Error_Symbol_Canvas)
                    if Buffer_Message_Check.get() == 1:
                        Buffer_Message_Check.set(0)
                        Incorrect_Buffer_Format_Display_Text_Canvas = canvas1.create_window(710,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Format_Display_Text)
                        canvas1.delete(Incorrect_Buffer_Format_Display_Text_Canvas)
                    try:
                        np.array(Pixel_Size_Value.get().split(',')).astype(float)
                    except ValueError:
                        Errors_Found[9]=1
                        if Pixel_Input_Check.get()==0:
                            Pixel_Input_Check.set(1)
                            Incorrect_Pixel_Error_Symbol_Canvas = canvas1.create_window(240,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Error_Symbol)
                        if Pixel_Message_Check.get()==0:
                            Pixel_Message_Check.set(1)
                            Incorrect_Pixel_Format_Display_Text_Canvas = canvas1.create_window(710,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Format_Display_Text)
                    else:
                        x,y,z = np.array(Pixel_Size_Value.get().split(',')).astype(float)>0
                        if x:
                            if y:
                                if z:
                                    if Pixel_Input_Check.get() == 1:
                                        Pixel_Input_Check.set(0)
                                        Incorrect_Pixel_Error_Symbol_Canvas = canvas1.create_window(240,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Error_Symbol)
                                        canvas1.delete(Incorrect_Pixel_Error_Symbol_Canvas)
                                    if Pixel_Message_Check.get() == 1:
                                        Incorrect_Pixel_Format_Display_Text_Canvas = canvas1.create_window(710,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Format_Display_Text)
                                        canvas1.delete(Incorrect_Pixel_Format_Display_Text_Canvas)
                        if not x:
                            Errors_Found[9]=1
                            if Pixel_Input_Check.get() == 0:
                                Pixel_Input_Check.set(1)
                                Incorrect_Pixel_Error_Symbol_Canvas = canvas1.create_window(240,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Error_Symbol)
                            if Pixel_Message_Check.get() == 0:
                                Pixel_Message_Check.set(1)
                                Incorrect_Pixel_Format_Display_Text_Canvas = canvas1.create_window(710,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Format_Display_Text)
                        if not y:
                            Errors_Found[9]=1
                            if Pixel_Input_Check.get() == 0:
                                Pixel_Input_Check.set(1)
                                Incorrect_Pixel_Error_Symbol_Canvas = canvas1.create_window(240,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Error_Symbol)
                            if Pixel_Message_Check.get() == 0:
                                Pixel_Message_Check.set(1)
                                Incorrect_Pixel_Format_Display_Text_Canvas = canvas1.create_window(710,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Format_Display_Text)
                        if not z:
                            Errors_Found[9]=1
                            if Pixel_Input_Check.get() == 0:
                                Pixel_Input_Check.set(1)
                                Incorrect_Pixel_Error_Symbol_Canvas = canvas1.create_window(240,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Error_Symbol)
                            if Pixel_Message_Check.get() == 0:
                                Pixel_Message_Check.set(1)
                                Incorrect_Pixel_Format_Display_Text_Canvas = canvas1.create_window(710,Pixel_Offset_Value.get(),window=Incorrect_Pixel_Format_Display_Text)
                            
                    try:
                        np.array(Buffer_Size_Value.get().split(',')).astype(float)
                    except ValueError:
                        Errors_Found[10]=1
                        if Buffer_Input_Check.get() == 0:
                            Buffer_Input_Check.set(1)
                            Incorrect_Buffer_Error_Symbol_Canvas = canvas1.create_window(240,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Error_Symbol)
                        if Buffer_Message_Check.get() == 0:
                            Buffer_Message_Check.set(1)
                            Incorrect_Buffer_Format_Display_Text_Canvas = canvas1.create_window(710,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Format_Display_Text)
                    else:
                        x,y,z = np.array(Buffer_Size_Value.get().split(',')).astype(float)>=0
                        if x:
                            if y:
                                if z:
                                    if Buffer_Input_Check.get() == 1:
                                        Buffer_Input_Check.set(0)
                                        Incorrect_Buffer_Error_Symbol_Canvas = canvas1.create_window(240,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Error_Symbol)
                                        canvas1.delete(Incorrect_Buffer_Error_Symbol_Canvas)
                                    if Buffer_Message_Check.get() == 1:
                                        Buffer_Message_Check.set(0)
                                        Incorrect_Buffer_Format_Display_Text_Canvas = canvas1.create_window(710,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Format_Display_Text)
                                        canvas1.delete(Incorrect_Buffer_Format_Display_Text_Canvas)
                        if not x:
                            Errors_Found[10]=1
                            if Buffer_Input_Check.get() == 0:
                                Buffer_Input_Check.set(1)
                                Incorrect_Buffer_Error_Symbol_Canvas = canvas1.create_window(240,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Error_Symbol)
                            if Buffer_Message_Check.get() == 0:
                                Buffer_Message_Check.set(1)
                                Incorrect_Buffer_Format_Display_Text_Canvas = canvas1.create_window(710,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Format_Display_Text)
                        if not y:
                            Errors_Found[10]=1
                            if Buffer_Input_Check.get() == 0:
                                Buffer_Input_Check.set(1)
                                Incorrect_Buffer_Error_Symbol_Canvas = canvas1.create_window(240,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Error_Symbol)
                            if Buffer_Message_Check.get() == 0:
                                Buffer_Message_Check.set(1)
                                Incorrect_Buffer_Format_Display_Text_Canvas = canvas1.create_window(710,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Format_Display_Text)
                        if not z:
                            Errors_Found[10]=1
                            if Buffer_Input_Check.get() == 0:
                                Buffer_Input_Check.set(1)
                                Incorrect_Buffer_Error_Symbol_Canvas = canvas1.create_window(240,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Error_Symbol)
                            if Buffer_Message_Check.get() == 0:
                                Buffer_Message_Check.set(1)
                                Incorrect_Buffer_Format_Display_Text_Canvas = canvas1.create_window(710,Buffer_Offset_Value.get(),window=Incorrect_Buffer_Format_Display_Text)
                        if Edge_Removal_Value.get() != '(Optional)':
                            try:
                                np.array(Edge_Removal_Value.get()).astype(int)>0
                            except ValueError:
                                Errors_Found[17]=1
                                Incorrect_Edge_Removal_Check.set(1)
                                Incorrect_Edge_Removal_Symbol_Canvas = canvas1.create_window(200,Edge_Removal_Offset_Value.get(),window=Incorrect_Edge_Removal_Symbol)
                                Incorrect_Edge_Removal_Value_Message_Canvas = canvas1.create_window(710,Edge_Removal_Offset_Value.get(),window=Incorrect_Edge_Removal_Value_Message)
                            else:
                                if np.array(Edge_Removal_Value.get()).astype(int)>=0:
                                    Incorrect_Edge_Removal_Check.set(0)
                                    Incorrect_Edge_Removal_Symbol_Canvas = canvas1.create_window(200,Edge_Removal_Offset_Value.get(),window=Incorrect_Edge_Removal_Symbol)
                                    canvas1.delete(Incorrect_Edge_Removal_Symbol_Canvas)
                                    Incorrect_Edge_Removal_Value_Message_Canvas = canvas1.create_window(710,Edge_Removal_Offset_Value.get(),window=Incorrect_Edge_Removal_Value_Message)
                                    canvas1.delete(Incorrect_Edge_Removal_Value_Message_Canvas)
                                if np.array(Edge_Removal_Value.get()).astype(int)<0:  
                                    Incorrect_Edge_Removal_Check.set(1)
                                    Incorrect_Edge_Removal_Symbol_Canvas = canvas1.create_window(200,Edge_Removal_Offset_Value.get(),window=Incorrect_Edge_Removal_Symbol)
                                    Incorrect_Edge_Removal_Value_Message_Canvas = canvas1.create_window(710,Edge_Removal_Offset_Value.get(),window=Incorrect_Edge_Removal_Value_Message)
            if Threshold_Menu_Chosen.get() == 'No':
                pass
            if Threshold_Menu_Chosen.get() == 'Select One':
                if Threshold_Choice_Check.get() == 0:
                    Incorrect_Threshold_Choice_Error_Symbol_Canvas = canvas1.create_window(240,188+Deriv_Menu_Offset.get()-56,window=Incorrect_Threshold_Choice_Error_Symbol)
                    Threshold_Choice_Check.set(1)
                    Errors_Found[4]=1
            if Deriv_Output_Directory_Text.get() == '(Optional)' or Deriv_Output_Directory_Text.get() == '':
                if Deriv_Output_Check.get() == 1:
                    Incorrect_Deriv_Output_Error_Canvas = canvas1.create_window(24,158+Deriv_Menu_Offset.get()-56,window=Incorrect_Deriv_Output_Error_Symbol)
                    canvas1.delete(Incorrect_Deriv_Output_Error_Canvas)
                    Deriv_Output_Check.set(0)
                    Deriv_Output_Hold_Check.set('')
                    Deriv_Output_Prechecked_Check.set(0)
            if not isdir(Output_Directory_Text.get()):
                Incorrect_Finished_Output_Symbol_Canvas = canvas1.create_window(25,466,window=Incorrect_Finished_Output_Symbol)
                Incorrect_Finished_Output_Message_Canvas = canvas1.create_window(450,494,window=Incorrect_Finished_Output_Message)
                Errors_Found[16] = 1
            if isdir(Output_Directory_Text.get()):
                Incorrect_Finished_Output_Symbol_Canvas = canvas1.create_window(25,466,window=Incorrect_Finished_Output_Symbol)
                canvas1.delete(Incorrect_Finished_Output_Symbol_Canvas)
                Incorrect_Finished_Output_Message_Canvas = canvas1.create_window(450,494,window=Incorrect_Finished_Output_Message)
                canvas1.delete(Incorrect_Finished_Output_Message_Canvas)
            #Errors_Found[0]=0
            #print(Errors_Found)
            if np.count_nonzero(Errors_Found) == 0:
                if Overal_Entries_Error_Check.get() == 1:
                    Overal_Entries_Error_Check.set(0)
                    Overall_Error_Message_Canvas = canvas1.create_window(450,50,window=Overall_Error_Message)
                    canvas1.delete(Overal_Entries_Error_Check)
                if Final_Selection_Made_Via_Button.get() == 1:
                    if Final_Selection_Avoid_Double_Sending.get() == 'No':
                        #Final_Selection_Avoid_Double_Sending.set('Yes')
                        Final_Selection_Made()
            if np.count_nonzero(Errors_Found) != 0:
                Final_Selection_Made_Via_Button.set(0)
                if Overal_Entries_Error_Check.get() == 0:
                    Overal_Entries_Error_Check.set(1)
                    Overall_Error_Message_Canvas = canvas1.create_window(450,50,window=Overall_Error_Message)
    def Remove_Previous_Thresholding_Errors():
        if Threshold_Choice_Check.get() == 1:
            Incorrect_Threshold_Choice_Error_Symbol_Canvas = canvas1.create_window(240,188+Deriv_Menu_Offset.get()-56,window=Incorrect_Threshold_Choice_Error_Symbol)
            canvas1.delete(Incorrect_Threshold_Choice_Error_Symbol_Canvas)
            Threshold_Choice_Check.set(0)
        if Pixel_Input_Check.get() == 1:
            Pixel_Input_Check.set(0)
            Incorrect_Pixel_Error_Symbol_Canvas = canvas1.create_window(240,328+Deriv_Menu_Offset.get()-56,window=Incorrect_Pixel_Error_Symbol)
            canvas1.delete(Incorrect_Pixel_Error_Symbol_Canvas)
            Incorrect_Pixel_Format_Display_Text_Canvas = canvas1.create_window(710,328+Deriv_Menu_Offset.get()-56,window=Incorrect_Pixel_Format_Display_Text)
            canvas1.delete(Incorrect_Pixel_Format_Display_Text_Canvas)
        if Buffer_Input_Check.get() == 1:
            Buffer_Input_Check.set(0)
            Incorrect_Buffer_Error_Symbol_Canvas = canvas1.create_window(240,355+Deriv_Menu_Offset.get()-56,window=Incorrect_Buffer_Error_Symbol)
            canvas1.delete(Incorrect_Buffer_Error_Symbol_Canvas)
            Incorrect_Buffer_Format_Display_Text_Canvas = canvas1.create_window(710,355+Deriv_Menu_Offset.get()-56,window=Incorrect_Buffer_Format_Display_Text)
            canvas1.delete(Incorrect_Buffer_Format_Display_Text_Canvas)
        if Threshold_Column_Standard_Deviation_Check.get() == 1:
            Threshold_Column_Standard_Deviation_Check.set(0)
            Incorrect_Column_Threshold_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_SD_Symbol)
            canvas1.delete(Incorrect_Column_Threshold_SD_Symbol_Canvas)
            Incorrect_Column_Threshold_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_SD_Message)
            canvas1.delete(Incorrect_Column_Threshold_SD_Message_Canvas)
        if Threshold_Column_Percent_Max_Check.get() == 1:
            Threshold_Column_Percent_Max_Check.set(0)
            Incorrect_Column_Threshold_Percentage_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_Percentage_Symbol)
            canvas1.delete(Incorrect_Column_Threshold_Percentage_Symbol_Canvas)
            Incorrect_Column_Threshold_Percentage_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Threshold_Percentage_Message)
            canvas1.delete(Incorrect_Column_Threshold_Percentage_Message_Canvas)
        if Threshold_Column_Minimum_Pixel_Check.get() == 1:
            Threshold_Column_Minimum_Pixel_Check.set(0)
            Incorrect_Column_Minimum_Pixel_Symbol_Canvas = canvas1.create_window(240,301+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Minimum_Pixel_Symbol)
            canvas1.delete(Incorrect_Column_Minimum_Pixel_Symbol_Canvas)
            Incorrect_Column_Minimum_Pixel_Message_Canvas = canvas1.create_window(710,301+Deriv_Menu_Offset.get()-56,window=Incorrect_Column_Minimum_Pixel_Message)
            canvas1.delete(Incorrect_Column_Minimum_Pixel_Message_Canvas)
        if Threshold_Intensity_Lower_Bound_Check.get() == 1:
            Threshold_Intensity_Lower_Bound_Check.set(0)
            Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Symbol)
            canvas1.delete(Incorrect_Intensity_Window_Bound_Lower_Symbol_Canvas)
            Incorrect_Intensity_Window_Bound_Lower_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Lower_Message)
            canvas1.delete(Incorrect_Intensity_Window_Bound_Lower_Message_Canvas)            
        if Threshold_Intensity_Upper_Bound_Check.get() == 1:
            Threshold_Intensity_Upper_Bound_Check.set(0)
            Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Symbol)
            canvas1.delete(Incorrect_Intensity_Window_Bound_Upper_Symbol_Canvas)
            Incorrect_Intensity_Window_Bound_Upper_Message_Canvas = canvas1.create_window(710,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Intensity_Window_Bound_Upper_Message)
            canvas1.delete(Incorrect_Intensity_Window_Bound_Upper_Message_Canvas)
        if Threshold_Gaussian_Standard_Deviation_Check.get() == 1:
            Threshold_Gaussian_Standard_Deviation_Check.set(0)
            Incorrect_Gaussian_SD_Symbol_Canvas = canvas1.create_window(240,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Symbol)
            canvas1.delete(Incorrect_Gaussian_SD_Symbol_Canvas)
            Incorrect_Gaussian_SD_Message_Canvas = canvas1.create_window(710,247+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_SD_Message)
            canvas1.delete(Incorrect_Gaussian_SD_Message_Canvas)
        if Threshold_Gaussian_Direction_Check.get() == 1:
            Threshold_Gaussian_Direction_Check.set(0)
            Incorrect_Gaussian_Select_Direction_Symbol_Canvas = canvas1.create_window(240,273+Deriv_Menu_Offset.get()-56,window=Incorrect_Gaussian_Select_Direction_Symbol)
            canvas1.delete(Incorrect_Gaussian_Select_Direction_Symbol_Canvas)
        if Incorrect_Edge_Removal_Check.get() == 1:
            Incorrect_Edge_Removal_Symbol_Canvas = canvas1.create_window(25,100,window=Incorrect_Edge_Removal_Symbol)
            Incorrect_Edge_Removal_Value_Message_Canvas = canvas1.create_window(450,100,window=Incorrect_Edge_Removal_Value_Message)
    def Threshold_Menu_Options_Function(*args):
        if Threshold_Previous_Method_Storage.get() != Threshold_Method_Selection.get():
            Threshold_Previous_Method_Storage.set(Threshold_Method_Selection.get())
            Reposition_Errors.set(1)
            Remove_Previous_Thresholding_Errors()
            #Checking_For_Errors()
            if Threshold_Method_Selection.get() != 'Select One':
                original_height.get()
                Threshold_Menu_Drop.set(100)
                #canvas1.config(width=1000,height=original_height.get()+Threshold_Menu_Drop.get())
                #Load_Preview_Hologram_Canvas = canvas1.create_window(331,162+Deriv_Menu_Offset.get(),window=Select_Hologram_To_Preview)
                #Specific_Information_Input_For_Preview_Button_canvas = canvas1.create_window(415,162+Deriv_Menu_Offset.get(),window=Specific_Information_Input_For_Preview_Button)
                
                #Output_Preview_Of_Threshold_Canvas = canvas1.create_window(401,162+Deriv_Menu_Offset.get(),window=Output_Preview_Of_Threshold)
                #Specific_Information_Threshold_Preview_Button_canvas = canvas1.create_window(556,162+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Preview_Button)
                
                if Threshold_Method_Selection.get() != 'Select One':
                    Pixel_Display_Text_ = canvas1.create_window(319,244+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Pixel_Label)
                    Buffer_Display_Text = canvas1.create_window(319,270+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Buffer_Label)
                    Pixel_Display_Entry_Window = canvas1.create_window(453,244+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Pixel_Size_Entry)
                    Buffer_Display_Entry_Window = canvas1.create_window(453,270+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Buffer_Size_Entry)
                    Pixel_Display_Text = canvas1.create_window(319,244+Deriv_Menu_Offset.get(),window=Pixel_Label)
                    Buffer_Display_Text = canvas1.create_window(319,270+Deriv_Menu_Offset.get(),window=Buffer_Label)
                    Pixel_Display_Entry_Window = canvas1.create_window(453,244+Deriv_Menu_Offset.get(),window=Pixel_Size_Entry)
                    Buffer_Display_Entry_Window = canvas1.create_window(453,270+Deriv_Menu_Offset.get(),window=Buffer_Size_Entry)
                    
                
                if Threshold_Method_Selection.get() == 'Column Thresholding':
                    Column_Finding_Thresholding_Label_Canvas = canvas1.create_window(368,190+Deriv_Menu_Offset.get(),window=Column_Finding_Thresholding_Label)
                    Column_FInding_Thresholding_Entry_Canvas = canvas1.create_window(510,190+Deriv_Menu_Offset.get(),window=Column_FInding_Thresholding_Entry)
                    Column_Thresholding_Label_Canvas = canvas1.create_window(368,216+Deriv_Menu_Offset.get(),window=Column_Thresholding_Label)
                    Column_Thresholding_Entry_Canvas = canvas1.create_window(510,216+Deriv_Menu_Offset.get(),window=Column_Thresholding_Entry)
                    Pixel_Display_Text_ = canvas1.create_window(319,270+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Pixel_Label)
                    Buffer_Display_Text = canvas1.create_window(319,296+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Buffer_Label)
                    Pixel_Display_Entry_Window = canvas1.create_window(453,270+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Pixel_Size_Entry)
                    Buffer_Display_Entry_Window = canvas1.create_window(453,296+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Buffer_Size_Entry)
                    Pixel_Display_Text = canvas1.create_window(319,270+Deriv_Menu_Offset.get(),window=Pixel_Label)
                    Buffer_Display_Text = canvas1.create_window(319,296+Deriv_Menu_Offset.get(),window=Buffer_Label)
                    Pixel_Display_Entry_Window = canvas1.create_window(453,270+Deriv_Menu_Offset.get(),window=Pixel_Size_Entry)
                    Buffer_Display_Entry_Window = canvas1.create_window(453,296+Deriv_Menu_Offset.get(),window=Buffer_Size_Entry)
                    Column_Thresholding_Minimum_Size_Label_Canvas = canvas1.create_window(350,244+Deriv_Menu_Offset.get(),window=Column_Thresholding_Minimum_Size_Label)
                    Column_Thresholding_Minimum_Size_Entry_Canvas = canvas1.create_window(493,244+Deriv_Menu_Offset.get(),window=Column_Thresholding_Minimum_Size_Entry)
                    Specific_Information_Threshold_Column_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Column_Standard_Deviation_Button)
                    Specific_Information_Threshold_Column_Percent_Max_Button_Canvas = canvas1.create_window(541,217+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Column_Percent_Max_Button)
                    Specific_Information_Threshold_Column_Minimum_Pixel_Button_Canvas = canvas1.create_window(541,244+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Column_Minimum_Pixel_Button)
                    Specific_Information_Pixel_Size_Button_Canvas = canvas1.create_window(541,271+Deriv_Menu_Offset.get(),window=Specific_Information_Pixel_Size_Button)
                    Specific_Information_Buffer_Size_Button_Canvas = canvas1.create_window(541,298+Deriv_Menu_Offset.get(),window=Specific_Information_Buffer_Size_Button)
                    Edge_Removal_Label_Canvas = canvas1.create_window(301,324+Deriv_Menu_Offset.get(),window=Edge_Removal_Label)
                    Edge_Removal_Entry_Canvas = canvas1.create_window(423,324+Deriv_Menu_Offset.get(),window=Edge_Removal_Entry)
                    Specific_Information_Edge_Removal_Button_Canvas = canvas1.create_window(541,324+Deriv_Menu_Offset.get(),window=Specific_Information_Edge_Removal_Button)
                    
                    Gaussian_Sided_Direction_Label_Canvas = canvas1.create_window(150,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Label)
                    Gaussian_Sided_Direction_Menu_Canvas = canvas1.create_window(303,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Menu,width=101)
                    Gaussian_Sided_Value_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Label)
                    Gaussian_Sided_Value_Entry_Canvas = canvas1.create_window(289,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Entry)
                    Gaussian_Sided_Value_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Label)
                    Gaussian_Sided_Value_Entry_Canvas = canvas1.create_window(289,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Entry)
                    Intensity_Window_Lower_Value_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Label)
                    Intensity_Window_Lower_Value_Canvas = canvas1.create_window(289,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Entry)
                    Intensity_Window_Upper_Value_Label_Canvas = canvas1.create_window(150,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Label)
                    Intensity_Window_Upper_Value_Canvas = canvas1.create_window(289,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Entry)
                    Specific_Information_Threshold_Intensity_Lower_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Intensity_Lower_Button)
                    Specific_Information_Threshold_Intensity_Upper_Button_Canvas = canvas1.create_window(541,217+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Intensity_Upper_Button)
                    Specific_Information_Threshold_Gaussian_Direction_Button_Canvas = canvas1.create_window(541,217++Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Gaussian_Direction_Button)
                    Specific_Information_Threshold_Gaussian_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Gaussian_Standard_Deviation_Button)
                if Threshold_Method_Selection.get() == 'Intensity Window - Binary' or Threshold_Method_Selection.get() == 'Intensity Window - Non-Binary':
                    Intensity_Window_Lower_Value_Label_Canvas = canvas1.create_window(319,190+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Label)
                    Intensity_Window_Lower_Value_Canvas = canvas1.create_window(423,190+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Entry)
                    Intensity_Window_Upper_Value_Label_Canvas = canvas1.create_window(319,216+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Label)
                    Intensity_Window_Upper_Value_Canvas = canvas1.create_window(423,216+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Entry)
                    Specific_Information_Threshold_Intensity_Lower_Button_Canvas = canvas1.create_window(541,190+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Intensity_Lower_Button)
                    Specific_Information_Threshold_Intensity_Upper_Button_Canvas = canvas1.create_window(541,217+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Intensity_Upper_Button)
                    Specific_Information_Pixel_Size_Button_Canvas = canvas1.create_window(541,244+Deriv_Menu_Offset.get(),window=Specific_Information_Pixel_Size_Button)
                    Specific_Information_Buffer_Size_Button_Canvas = canvas1.create_window(541,271+Deriv_Menu_Offset.get(),window=Specific_Information_Buffer_Size_Button)
                    Edge_Removal_Label_Canvas = canvas1.create_window(301,298+Deriv_Menu_Offset.get(),window=Edge_Removal_Label)
                    Edge_Removal_Entry_Canvas = canvas1.create_window(423,298+Deriv_Menu_Offset.get(),window=Edge_Removal_Entry)
                    Specific_Information_Edge_Removal_Button_Canvas = canvas1.create_window(541,298+Deriv_Menu_Offset.get(),window=Specific_Information_Edge_Removal_Button)
                    
                    Gaussian_Sided_Direction_Label_Canvas = canvas1.create_window(150,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Label)
                    Gaussian_Sided_Direction_Menu_Canvas = canvas1.create_window(303,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Menu,width=101)
                    Gaussian_Sided_Value_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Label)
                    Gaussian_Sided_Value_Entry_Canvas = canvas1.create_window(289,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Entry)
                    Gaussian_Sided_Value_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Label)
                    Gaussian_Sided_Value_Entry_Canvas = canvas1.create_window(289,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Entry)
                    Column_Finding_Thresholding_Label_Canvas = canvas1.create_window(368,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Finding_Thresholding_Label)
                    Column_FInding_Thresholding_Entry_Canvas = canvas1.create_window(510,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_FInding_Thresholding_Entry)
                    Column_Thresholding_Label_Canvas = canvas1.create_window(368,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Thresholding_Label)
                    Column_Thresholding_Entry_Canvas = canvas1.create_window(510,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Thresholding_Entry)
                    Column_Thresholding_Minimum_Size_Label_Canvas = canvas1.create_window(350,244+Threshold_Reset_Offset.get(),window=Column_Thresholding_Minimum_Size_Label)
                    Column_Thresholding_Minimum_Size_Entry_Canvas = canvas1.create_window(493,244+Threshold_Reset_Offset.get(),window=Column_Thresholding_Minimum_Size_Entry)
                    Specific_Information_Threshold_Column_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Standard_Deviation_Button)
                    Specific_Information_Threshold_Column_Percent_Max_Button_Canvas = canvas1.create_window(541,217+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Percent_Max_Button)
                    Specific_Information_Threshold_Column_Minimum_Pixel_Button_Canvas = canvas1.create_window(541,244+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Minimum_Pixel_Button)
                    Specific_Information_Threshold_Gaussian_Direction_Button_Canvas = canvas1.create_window(541,217++Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Gaussian_Direction_Button)
                    Specific_Information_Threshold_Gaussian_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Gaussian_Standard_Deviation_Button)
                if Threshold_Method_Selection.get() == 'One-sided Gaussian':
                    Gaussian_Sided_Direction_Label_Canvas = canvas1.create_window(340,216+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Label)
                    Gaussian_Sided_Direction_Menu_Canvas = canvas1.create_window(478,216+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Menu,width=101)
                    Gaussian_Sided_Value_Label_Canvas = canvas1.create_window(319,190+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Label)
                    Gaussian_Sided_Value_Entry_Canvas = canvas1.create_window(423,190+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Entry)
                    Specific_Information_Threshold_Gaussian_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Gaussian_Standard_Deviation_Button)
                    Specific_Information_Threshold_Gaussian_Direction_Button_Canvas = canvas1.create_window(555,217+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Gaussian_Direction_Button)
                    Specific_Information_Pixel_Size_Button_Canvas = canvas1.create_window(541,244+Deriv_Menu_Offset.get(),window=Specific_Information_Pixel_Size_Button)
                    Specific_Information_Buffer_Size_Button_Canvas = canvas1.create_window(541,271+Deriv_Menu_Offset.get(),window=Specific_Information_Buffer_Size_Button)
                    Edge_Removal_Label_Canvas = canvas1.create_window(301,298+Deriv_Menu_Offset.get(),window=Edge_Removal_Label)
                    Edge_Removal_Entry_Canvas = canvas1.create_window(423,298+Deriv_Menu_Offset.get(),window=Edge_Removal_Entry)
                    Specific_Information_Edge_Removal_Button_Canvas = canvas1.create_window(541,298+Deriv_Menu_Offset.get(),window=Specific_Information_Edge_Removal_Button)
                    
                    Intensity_Window_Lower_Value_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Label)
                    Intensity_Window_Lower_Value_Canvas = canvas1.create_window(289,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Entry)
                    Intensity_Window_Upper_Value_Label_Canvas = canvas1.create_window(150,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Label)
                    Intensity_Window_Upper_Value_Canvas = canvas1.create_window(289,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Entry)
                    Column_Finding_Thresholding_Label_Canvas = canvas1.create_window(368,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Finding_Thresholding_Label)
                    Column_FInding_Thresholding_Entry_Canvas = canvas1.create_window(510,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_FInding_Thresholding_Entry)
                    Column_Thresholding_Label_Canvas = canvas1.create_window(368,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Thresholding_Label)
                    Column_Thresholding_Entry_Canvas = canvas1.create_window(510,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Thresholding_Entry)
                    Column_Thresholding_Minimum_Size_Label_Canvas = canvas1.create_window(350,244+Threshold_Reset_Offset.get(),window=Column_Thresholding_Minimum_Size_Label)
                    Column_Thresholding_Minimum_Size_Entry_Canvas = canvas1.create_window(493,244+Threshold_Reset_Offset.get(),window=Column_Thresholding_Minimum_Size_Entry)
                    Specific_Information_Threshold_Column_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Standard_Deviation_Button)
                    Specific_Information_Threshold_Column_Percent_Max_Button_Canvas = canvas1.create_window(541,217+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Percent_Max_Button)
                    Specific_Information_Threshold_Column_Minimum_Pixel_Button_Canvas = canvas1.create_window(541,244+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Minimum_Pixel_Button)
                    Specific_Information_Threshold_Intensity_Lower_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Intensity_Lower_Button)
                    Specific_Information_Threshold_Intensity_Upper_Button_Canvas = canvas1.create_window(541,217+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Intensity_Upper_Button)
                if Threshold_Method_Selection.get() == 'Two-sided Gaussian':
                    Threshold_Menu_Drop.set(Threshold_Menu_Drop.get()-28)
                    Gaussian_Sided_Value_Label_Canvas = canvas1.create_window(319,190+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Label)
                    Gaussian_Sided_Value_Entry_Canvas = canvas1.create_window(423,190+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Entry)
                    Pixel_Display_Text_ = canvas1.create_window(319,216+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Pixel_Label)
                    Buffer_Display_Text = canvas1.create_window(319,244+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Buffer_Label)
                    Pixel_Display_Entry_Window = canvas1.create_window(453,216+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Pixel_Size_Entry)
                    Buffer_Display_Entry_Window = canvas1.create_window(453,244+Threshold_Menu_Drop.get()+Deriv_Menu_Offset.get(),window=Buffer_Size_Entry)
                    Pixel_Display_Text = canvas1.create_window(319,216+Deriv_Menu_Offset.get(),window=Pixel_Label)
                    Buffer_Display_Text = canvas1.create_window(319,244+Deriv_Menu_Offset.get(),window=Buffer_Label)
                    Pixel_Display_Entry_Window = canvas1.create_window(453,216+Deriv_Menu_Offset.get(),window=Pixel_Size_Entry)
                    Buffer_Display_Entry_Window = canvas1.create_window(453,244+Deriv_Menu_Offset.get(),window=Buffer_Size_Entry)
                    Specific_Information_Threshold_Gaussian_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Gaussian_Standard_Deviation_Button)
                    Specific_Information_Pixel_Size_Button_Canvas = canvas1.create_window(541,217+Deriv_Menu_Offset.get(),window=Specific_Information_Pixel_Size_Button)
                    Specific_Information_Buffer_Size_Button_Canvas = canvas1.create_window(541,244+Deriv_Menu_Offset.get(),window=Specific_Information_Buffer_Size_Button)
                    Edge_Removal_Label_Canvas = canvas1.create_window(301,272+Deriv_Menu_Offset.get(),window=Edge_Removal_Label)
                    Edge_Removal_Entry_Canvas = canvas1.create_window(423,272+Deriv_Menu_Offset.get(),window=Edge_Removal_Entry)
                    Specific_Information_Edge_Removal_Button_Canvas = canvas1.create_window(541,272+Deriv_Menu_Offset.get(),window=Specific_Information_Edge_Removal_Button)
                    
                    Gaussian_Sided_Direction_Label_Canvas = canvas1.create_window(150,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Label)
                    Gaussian_Sided_Direction_Menu_Canvas = canvas1.create_window(303,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Menu,width=101)
                    Intensity_Window_Lower_Value_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Label)
                    Intensity_Window_Lower_Value_Canvas = canvas1.create_window(289,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Entry)
                    Intensity_Window_Upper_Value_Label_Canvas = canvas1.create_window(150,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Label)
                    Intensity_Window_Upper_Value_Canvas = canvas1.create_window(289,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Entry)
                    Column_Finding_Thresholding_Label_Canvas = canvas1.create_window(368,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Finding_Thresholding_Label)
                    Column_FInding_Thresholding_Entry_Canvas = canvas1.create_window(510,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_FInding_Thresholding_Entry)
                    Column_Thresholding_Label_Canvas = canvas1.create_window(368,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Thresholding_Label)
                    Column_Thresholding_Entry_Canvas = canvas1.create_window(510,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Thresholding_Entry)
                    Column_Thresholding_Minimum_Size_Label_Canvas = canvas1.create_window(350,244+Threshold_Reset_Offset.get(),window=Column_Thresholding_Minimum_Size_Label)
                    Column_Thresholding_Minimum_Size_Entry_Canvas = canvas1.create_window(493,244+Threshold_Reset_Offset.get(),window=Column_Thresholding_Minimum_Size_Entry)
                    Specific_Information_Threshold_Column_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Standard_Deviation_Button)
                    Specific_Information_Threshold_Column_Percent_Max_Button_Canvas = canvas1.create_window(541,217+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Percent_Max_Button)
                    Specific_Information_Threshold_Column_Minimum_Pixel_Button_Canvas = canvas1.create_window(541,244+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Minimum_Pixel_Button)
                    Specific_Information_Threshold_Intensity_Lower_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Intensity_Lower_Button)
                    Specific_Information_Threshold_Intensity_Upper_Button_Canvas = canvas1.create_window(541,217+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Intensity_Upper_Button)
                    Specific_Information_Threshold_Gaussian_Direction_Button_Canvas = canvas1.create_window(541,217++Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Gaussian_Direction_Button)
            if Threshold_Method_Selection.get() == 'Select One':
                original_height.get()
                Threshold_Menu_Drop.set(0)
                #if Final_Selection_Made_Via_Button.get() == 1:
                #    Final_Selection_Made()
                
                Load_Preview_Hologram_Canvas = canvas1.create_window(331,188+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Select_Hologram_To_Preview)
                Output_Preview_Of_Threshold_Canvas = canvas1.create_window(465,188+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Output_Preview_Of_Threshold)
                Intensity_Window_Lower_Value_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Label)
                Intensity_Window_Lower_Value_Canvas = canvas1.create_window(289,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Lower_Value_Entry)
                Intensity_Window_Upper_Value_Label_Canvas = canvas1.create_window(150,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Label)
                Intensity_Window_Upper_Value_Canvas = canvas1.create_window(289,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Intensity_Window_Upper_Value_Entry)
                Gaussian_Sided_Direction_Label_Canvas = canvas1.create_window(150,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Label)
                Gaussian_Sided_Direction_Menu_Canvas = canvas1.create_window(303,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Direction_Menu,width=101)
                Gaussian_Sided_Value_Label_Canvas = canvas1.create_window(150,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Label)
                Gaussian_Sided_Value_Entry_Canvas = canvas1.create_window(289,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Gaussian_Sided_Value_Entry)
                Column_Finding_Thresholding_Label_Canvas = canvas1.create_window(368,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Finding_Thresholding_Label)
                Column_FInding_Thresholding_Entry_Canvas = canvas1.create_window(510,216+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_FInding_Thresholding_Entry)
                Column_Thresholding_Label_Canvas = canvas1.create_window(368,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Thresholding_Label)
                Column_Thresholding_Entry_Canvas = canvas1.create_window(510,244+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Column_Thresholding_Entry)
                Column_Thresholding_Minimum_Size_Label_Canvas = canvas1.create_window(350,244+Threshold_Reset_Offset.get(),window=Column_Thresholding_Minimum_Size_Label)
                Column_Thresholding_Minimum_Size_Entry_Canvas = canvas1.create_window(493,244+Threshold_Reset_Offset.get(),window=Column_Thresholding_Minimum_Size_Entry)
                Pixel_Display_Text_ = canvas1.create_window(150,270+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Pixel_Label)
                Buffer_Display_Text = canvas1.create_window(150,296+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Buffer_Label)
                Pixel_Display_Entry_Window = canvas1.create_window(319,270+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Pixel_Size_Entry)
                Buffer_Display_Entry_Window = canvas1.create_window(319,296+Threshold_Reset_Offset.get()+Deriv_Menu_Offset.get(),window=Buffer_Size_Entry)
                Pixel_Display_Text = canvas1.create_window(150,270+Threshold_Reset_Offset.get(),window=Pixel_Label)
                Buffer_Display_Text = canvas1.create_window(150,296+Threshold_Reset_Offset.get(),window=Buffer_Label)
                Pixel_Display_Entry_Window = canvas1.create_window(319,270+Threshold_Reset_Offset.get(),window=Pixel_Size_Entry)
                Buffer_Display_Entry_Window = canvas1.create_window(319,296+Threshold_Reset_Offset.get(),window=Buffer_Size_Entry)
                Specific_Information_Threshold_Column_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Standard_Deviation_Button)
                Specific_Information_Threshold_Column_Percent_Max_Button_Canvas = canvas1.create_window(541,217+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Percent_Max_Button)
                Specific_Information_Threshold_Column_Minimum_Pixel_Button_Canvas = canvas1.create_window(541,244+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Minimum_Pixel_Button)
                Specific_Information_Pixel_Size_Button_Canvas = canvas1.create_window(541,271+Threshold_Reset_Offset.get(),window=Specific_Information_Pixel_Size_Button)
                Specific_Information_Buffer_Size_Button_Canvas = canvas1.create_window(541,298+Threshold_Reset_Offset.get(),window=Specific_Information_Buffer_Size_Button)
                Specific_Information_Threshold_Intensity_Lower_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Intensity_Lower_Button)
                Specific_Information_Threshold_Intensity_Upper_Button_Canvas = canvas1.create_window(541,217+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Intensity_Upper_Button)
                Specific_Information_Threshold_Gaussian_Direction_Button_Canvas = canvas1.create_window(541,217++Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Gaussian_Direction_Button)
                Specific_Information_Threshold_Gaussian_Standard_Deviation_Button_Canvas = canvas1.create_window(541,190+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Gaussian_Standard_Deviation_Button)
                Specific_Information_Threshold_Column_Minimum_Pixel_Button_Canvas = canvas1.create_window(541,244+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Column_Minimum_Pixel_Button)
                Specific_Information_Input_For_Preview_Button_canvas = canvas1.create_window(415,162+Threshold_Reset_Offset.get(),window=Specific_Information_Input_For_Preview_Button)
                Specific_Information_Threshold_Preview_Button_canvas = canvas1.create_window(548,162+Threshold_Reset_Offset.get(),window=Specific_Information_Threshold_Preview_Button)
                Edge_Removal_Label_Canvas = canvas1.create_window(301,272+Threshold_Reset_Offset.get(),window=Edge_Removal_Label)
                Edge_Removal_Entry_Canvas = canvas1.create_window(423,272+Threshold_Reset_Offset.get(),window=Edge_Removal_Entry)
                Specific_Information_Edge_Removal_Button_Canvas = canvas1.create_window(541,272+Threshold_Reset_Offset.get(),window=Specific_Information_Edge_Removal_Button)
                
    def Load_Preview_Hologram():
        pass
    def thresholding_menu_display_function(*args):
        if Threshold_Prior_Selection_Check.get() != Threshold_Menu_Chosen.get():
            Threshold_Prior_Selection_Check.set(Threshold_Menu_Chosen.get())
            Remove_Previous_Thresholding_Errors()
            if Threshold_Menu_Chosen.get() == 'Yes':
            #if Threshold_Menu_Chosen.get() != 'No':
                Threshold_Method_Selection.set(Threshold_Menu_Chosen_Memory.get())
                Threshold_Menu_Yes_Check.set('Yes')
                Threshold_Display_Methods_Menu = canvas1.create_window(490,130+Deriv_Menu_Offset.get(),window=Threshold_Method_Menu,width=205)
                Specific_Information_Threshold_Method_Button_Canvas = canvas1.create_window(599,130+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Method_Button)
            if Threshold_Menu_Chosen.get() != 'Yes':
                if Threshold_Menu_Yes_Check.get() == 'Yes':
                    Threshold_Menu_Chosen_Memory.set(Threshold_Method_Selection.get())
                    Threshold_Method_Selection.set('Select One')
                    Threshold_Menu_Options_Function()
                    Threshold_Display_Methods_Menu = canvas1.create_window(455,130+Deriv_Menu_Offset.get()+1000,window=Threshold_Method_Menu,width=205)
                    Specific_Information_Threshold_Method_Button_Canvas = canvas1.create_window(580,130+Deriv_Menu_Offset.get()+1000,window=Specific_Information_Threshold_Method_Button)
                    Threshold_Menu_Yes_Check.set('No')
    def Derivative_Variable_Display(*args):
        if Deriv_Menu_Chosen.get() == 'Yes':
            Deriv_Menu_Yes_Check.set('Yes')
            Deriv_x_Label_Canvas = canvas1.create_window(294,156,window=Deriv_x_Label)
            Deriv_x_Entry_Canvas = canvas1.create_window(344,156,window=Deriv_x_Entry)
            Deriv_y_Label_Canvas = canvas1.create_window(394,156,window=Deriv_y_Label)
            Deriv_y_Entry_Canvas = canvas1.create_window(444,156,window=Deriv_y_Entry)
            Deriv_z_Label_Canvas = canvas1.create_window(494,156,window=Deriv_z_Label)
            Deriv_z_Entry_Canvas = canvas1.create_window(544,156,window=Deriv_z_Entry)
            Deriv_Output_Directory_Button_Canvas = canvas1.create_window(150,182,window=Deriv_Output_Directory_Button)
            Deriv_Output_Directory_Entry_Canvas = canvas1.create_window(532,182,window=Deriv_Output_Directory_Text_Entry)
            Specific_Information_Deriv_Output_Directory_Button_Canvas = canvas1.create_window(813,182,window=Specific_Information_Deriv_Output_Directory_Button)
            Specific_Information_Deriv_Order_Button_Canvas = canvas1.create_window(570,157,window=Specific_Information_Deriv_Order_Button)
            
        if Deriv_Menu_Chosen.get() != 'Yes':
            if Deriv_Menu_Yes_Check.get() == 'Yes':
                Deriv_Menu_Yes_Check.set('No')
                Deriv_x_Label_Canvas = canvas1.create_window(294,156+1000,window=Deriv_x_Label)
                Deriv_x_Entry_Canvas = canvas1.create_window(344,156+1000,window=Deriv_x_Entry)
                Deriv_y_Label_Canvas = canvas1.create_window(394,156+1000,window=Deriv_y_Label)
                Deriv_y_Entry_Canvas = canvas1.create_window(444,156+1000,window=Deriv_y_Entry)
                Deriv_z_Label_Canvas = canvas1.create_window(494,156+1000,window=Deriv_z_Label)
                Deriv_z_Entry_Canvas = canvas1.create_window(544,156+1000,window=Deriv_z_Entry)
                Deriv_Output_Directory_Button_Canvas = canvas1.create_window(150,182+1000,window=Deriv_Output_Directory_Button)
                Deriv_Output_Directory_Entry_Canvas = canvas1.create_window(545,182+1000,window=Deriv_Output_Directory_Text_Entry)
                Specific_Information_Deriv_Output_Directory_Button_Canvas = canvas1.create_window(813,156+1000,window=Specific_Information_Deriv_Output_Directory_Button)
                Specific_Information_Deriv_Order_Button_Canvas = canvas1.create_window(580,129+1000,window=Specific_Information_Deriv_Order_Button)
    def Deriv_Output_Directory():
        if Deriv_Output_Directory_Text != '':
            Deriv_Output_Directory_Text.set('')
            Deriv_Output_Directory_Chosen = filedialog.askdirectory(parent=top,title='Choose a directory')
            Deriv_Output_Directory_Text_Entry.insert(END,Deriv_Output_Directory_Chosen)
        if Deriv_Output_Directory_Text == '':
            Deriv_Output_Directory_Chosen = filedialog.askdirectory(parent=top,title='Choose a directory')
            Deriv_Output_Directory_Text_Entry.insert(END,Deriv_Output_Directory_Chosen)    
    def Final_Selection_Made_Button_Check():
        Final_Selection_Made_Via_Button.set(1)
        #Initializing_Check.set(1)
        Checking_For_Errors()
    def Load_Preview_Hologram():
        pass
    def Threshold_And_View():
        pass
    
    General_Info_Frame = Frame(top,bg='white',width=500,height=190)
    Specific_Info_Frame = Frame(top,bg='white',width=500,height=190)
    General_Info_Frame_Canvas = canvas1.create_window(1125,125,window=General_Info_Frame)
    Specific_Info_Frame_Canvas = canvas1.create_window(1125,350,window=Specific_Info_Frame)
    General_Info_Frame_Title = Label(top,text='General Information')
    Specific_Info_Frame_Tile = Label(top,text='Specific Function Information')
    General_Info_Frame_Title_Canvas = canvas1.create_window(1125,20,window=General_Info_Frame_Title)
    Specific_Info_Frame_Tile_Canvas = canvas1.create_window(1125,245,window=Specific_Info_Frame_Tile)
    
    

    
    Pixel_Size_Value = StringVar()
    Pixel_Size_Value.set('0.1285,0.1285,1')
    Buffer_Size_Value = StringVar()
    Buffer_Size_Value.set('1,1,2')
    Overall_Error_Message = Label(top,text='Missing Correct Inputs(*), Please Correct and Try Again')
    Error_Message_Cover = Label(top,text='',width=50)
    Error_Message_Check = IntVar()
    Incorrect_Entries_Error_Symbol = Label(top,text='*')
    Incorrect_Reconstruction_Input_Error_Symbol = Label(top,text='*')
    Incorrect_Reconstruction_Input_Error_Message_Not_Valid = Label(top,text='Reconstruction Directory Is Not A Valid Directory')
    Incorrect_Reconstruction_Input_Error_Message_Missing_Folders = Label(top,text='Reconstruction Directory Does Not Contain Valid Folders')
    Incorrect_Reconstruction_Input_Error_Message_Missing_Files = Label(top,text='Reconstruction Directory Contains Invalid Holograms')
    Incorrect_Output_Error_Label = Label(top,text='*')
    Incorrect_Deriv_Choice_Error_Symbol = Label(top,text='*')
    Incorrect_Deriv_Order_Error_Symbol = Label(top,text='*')
    Incorrect_Deriv_Order_Error_Message = Label(top,text='Positive Integers Only (>0)')
    Incorrect_Deriv_Output_Error_Symbol = Label(top,text='*')
    Incorrect_Deriv_Output_Error_Message = Label(top,text='Deriv Output Is Not a Valid Directory')
    Incorrect_Threshold_Choice_Error_Symbol = Label(top,text='*')
    Incorrect_Threshold_Method_Error_Symbol = Label(top,text='*')
    Incorrect_Pixel_Error_Symbol = Label(top,text='*')
    #Incorrect_Pixel_Error_Message = Label(top,text='Pixels Must Be Positive Integers Only (>0)')
    Incorrect_Buffer_Error_Symbol = Label(top,text='*')
    #Incorrect_Buffer_Error_Message = Label(top,text='Buffers Must Be Positive Integers Only (>0)')
    Incorrect_Input_Preview_Hologram_Label = Label(top,text='*')
    Incorrect_Input_Preview_Hologram_Label_Text = Label(top,text='No Hologram Loaded')
    Incorrect_Column_Threshold_SD_Symbol = Label(top,text='*')
    Incorrect_Column_Threshold_SD_Message = Label(top,text='Must Be Positive Integers Only (>0)')
    Incorrect_Column_Threshold_Percentage_Symbol = Label(top,text='*')
    Incorrect_Column_Threshold_Percentage_Message = Label(top,text='Must Be Positive Integers Only (>0)')
    Incorrect_Column_Minimum_Pixel_Symbol = Label(top,text='*')
    Incorrect_Column_Minimum_Pixel_Message = Label(top,text='Must Be Positive Integers Only (>0)')
    Incorrect_Intensity_Window_Bound_Lower_Symbol = Label(top,text='*')
    Incorrect_Intensity_Window_Bound_Lower_Message = Label(top,text='Must Be Positive Integers Only (>0)')
    Incorrect_Intensity_Window_Bound_Upper_Symbol = Label(top,text='*')
    Incorrect_Intensity_Window_Bound_Upper_Message = Label(top,text='Must Be Positive Integers Only (>0)')
    Incorrect_Gaussian_SD_Symbol = Label(top,text='*')
    Incorrect_Gaussian_SD_Message = Label(top,text='Must Be Positive Integers Only (>0)')
    Incorrect_Gaussian_Select_Direction_Symbol = Label(top,text='*')
    Incorrect_Finished_Output_Symbol = Label(top,text='*')
    Incorrect_Finished_Output_Message = Label(top,text='Finished Output Is Not a Valid Directory')
    Incorrect_Deriv_Order_Summation_Message = Label(top,text='Sum of Orders Must Be Greater Than Zero')
    Incorrect_Edge_Removal_Value_Message = Label(top,text='Must Be Positive Integers Only (>0)')
    Incorrect_Edge_Removal_Symbol = Label(top,text='*')
    Incorrect_Edge_Removal_Check = IntVar()
    Edge_Removal_Offset_Value = IntVar()
    
    Recon_Input_Check = IntVar()
    Recon_Input_Hold_Check = StringVar()
    Recon_Input_Prechecked_Check = IntVar()
    Deriv_Choice_Check = IntVar()
    Deriv_Order_Check = IntVar()
    Deriv_x_check = IntVar()
    Deriv_y_check = IntVar()
    Deriv_z_check = IntVar()
    Deriv_Order_Chain_Check = IntVar()
    Deriv_Output_Check = IntVar()
    Deriv_Output_Hold_Check = StringVar()
    Deriv_Output_Prechecked_Check = IntVar()
    Threshold_Choice_Check = IntVar()
    Threshold_Method_Check = IntVar()
    Threshold_Column_Standard_Deviation_Check = IntVar()
    Threshold_Column_Percent_Max_Check = IntVar()
    Threshold_Column_Minimum_Pixel_Check = IntVar()
    Threshold_Intensity_Lower_Bound_Check = IntVar()
    Threshold_Intensity_Upper_Bound_Check = IntVar()
    Threshold_Intensity_Valid_Range_Check = IntVar()
    Threshold_Gaussian_Standard_Deviation_Check = IntVar()
    Threshold_Gaussian_Direction_Check = IntVar()
    Pixel_Input_Check = IntVar()
    Pixel_x_Check = IntVar()
    Pixel_y_Check = IntVar()
    Pixel_z_Check = IntVar()
    Buffer_Input_Check = IntVar()
    Buffer_x_Check = IntVar()
    Pixel_Buffer_Offset_Tracker = IntVar()
    Pixel_Offset_Value = IntVar()
    Buffer_Offset_Value = IntVar()
    Move_Pixel_Buffer_Offset = IntVar()
    Pixel_Buffer_Offset_Tracker_Temp_Storage = IntVar()
    Threshold_Method_Temp_Storage = StringVar()
    Threshold_Previous_Method_Storage = StringVar()
    Overal_Entries_Error_Check = IntVar()
    Pixel_Message_Check = IntVar()
    Buffer_Message_Check = IntVar()
    Incorrect_Reconstruction_Input_Error_Message_Not_Valid_Check = IntVar()
    Incorrect_Reconstruction_Input_Error_Message_Missing_Folders_Check = IntVar()
    Incorrect_Reconstruction_Input_Error_Message_Missing_Files_Check = IntVar()
    
    
    
    
    Specific_Information_Recon_Input_Directory_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Deriv_Use_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Deriv_Order_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Deriv_Output_Directory_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Use_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Method_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Input_For_Preview_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Preview_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Intensity_Lower_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Intensity_Upper_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Gaussian_Standard_Deviation_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Gaussian_Direction_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Column_Standard_Deviation_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Column_Percent_Max_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Threshold_Column_Minimum_Pixel_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Pixel_Size_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Buffer_Size_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Finished_Output_Button = tk.Button(top,text='?',command=Button)
    Specific_Information_Edge_Removal_Button = tk.Button(top,text='?',command=Button)
    amplitude_phase_chosen = StringVar()
    amplitude_phase_chosen.set('Select One')
    Amplitude_Phase_Choice_input = OptionMenu(top,amplitude_phase_chosen,'Select One','Amplitude','Phase')
    Input_Directory_Text = StringVar()
    Input_Directory_Text.set('(Required)')
    #Input_Directory_Text.set('D:/DHM_Analysis/Delete')
    #Input_Directory_Text.set('H:/laser-lasso_backup/elec/2019.08.31_08.00.01.92/bug_of_interest/Phase/')
    Input_Directory_Button = tk.Button(top,text='Select Reconstruction Input Directory',command=Input_Directory,width=29)
    Output_Directory_Text = StringVar()
    Output_Directory_Text.set('(Required)')
    Output_Directory_Button = tk.Button(top,text='Select Finished Output Directory',command=Output_Directory,width=29)
    Input_Directory_Text_Entry = Entry(top,textvariable=Input_Directory_Text,width=90)
    Output_Directory_Text_Entry = Entry(top,textvariable=Output_Directory_Text,width=90)
    Deriv_Menu_Chosen = StringVar()
    Deriv_Menu_Chosen.set('Select One')
    Deriv_Menu_Chosen.trace('w',Derivative_Variable_Display)
    Deriv_Menu_Yes_Check = StringVar()
    Deriv_Menu_Yes_Check.set('No')
    Threshold_Menu_Chosen = StringVar()
    Threshold_Menu_Chosen.set('Select One')
    Threshold_Menu_Chosen.trace('w',thresholding_menu_display_function)
    Threshold_Menu_Chosen_Memory = StringVar()
    Threshold_Menu_Chosen_Memory.set('Select One')
    Threshold_Menu_Yes_Check = StringVar()
    Threshold_Menu_Yes_Check.set('No')
    Deriv_Menu = OptionMenu(top,Deriv_Menu_Chosen,'Select One','Yes','No')
    Threshold_Menu = OptionMenu(top,Threshold_Menu_Chosen,'Select One','Yes','No')
    Pixel_Size_Entry = Entry(top,textvariable=Pixel_Size_Value,width=20)
    Buffer_Size_Entry = Entry(top,textvariable=Buffer_Size_Value,width=20)
    Pixel_Label = Label(top,text='Pixel Size (microns):')
    Buffer_Label = Label(top,text='Buffer Size (microns):')
    Method_Choice_Label = Label(top,text='Amplitude or Phase?')
    Deriv_Choice_Label = Label(top,text='Use Deriv Method?')
    Threshold_Menu_Label = Label(top,text='Threshold Choice')
    Finish_Selections_Button = tk.Button(top,text='Start Pre-processing',command=Final_Selection_Made_Button_Check,width=30)
    Incorrect_Pixel_Format_Display_Text = Label(top,text='Correct format is "x,y,z" with positive numbers only  (>0)')
    Incorrect_Buffer_Format_Display_Text = Label(top,text='Correct format is "x,y,z" with positive numbers only (>0)')
    Threshold_Method_Selection = StringVar()
    Threshold_Method_Selection.set('Select One')
    Threshold_Method_Menu = OptionMenu(top,Threshold_Method_Selection,'Select One','Column Thresholding','Intensity Window - Binary','Intensity Window - Non-Binary','One-sided Gaussian','Two-sided Gaussian')
    Threshold_Method_Selection.trace('w',Threshold_Menu_Options_Function)
    #Display_Preview_Settings = tk.Button(top,text='Show Preview Settings',width=20,command=Preview_Settings)
    Intensity_Window_Valid_Range_Check = IntVar()
    Intensity_Window_Lower_Value = StringVar()
    Intensity_Window_Lower_Value.set('0')
    Intensity_Window_Lower_Value_Label = Label(top,text='Lower Bound:')
    Intensity_Window_Lower_Value_Entry = Entry(top,textvariable=Intensity_Window_Lower_Value,width=10)
    Intensity_Window_Upper_Value = StringVar()
    Intensity_Window_Upper_Value.set('0')
    Intensity_Window_Upper_Value_Label = Label(top,text='Upper Bound:')
    Intensity_Window_Upper_Value_Entry = Entry(top,textvariable=Intensity_Window_Upper_Value,width=10)
    Gaussian_Sided_Direction = StringVar()
    Gaussian_Sided_Direction.set('Select One')
    Gaussian_Sided_Direction_Label = Label(top,text='Select Direction of Thresholding:')
    Gaussian_Sided_Direction_Menu = OptionMenu(top,Gaussian_Sided_Direction,'Select One','Positive','Negative')
    Gaussian_Sided_Value = StringVar()
    Gaussian_Sided_Value.set(0)
    Gaussian_Sided_Value_Label = Label(top,text='Standard Deviations:')
    Gaussian_Sided_Value_Entry = Entry(top,textvariable=Gaussian_Sided_Value,width=10)
    Select_Hologram_To_Preview = tk.Button(top,text='Input Preview Hologram',command=Load_Preview_Hologram,width=20)
    Output_Preview_Of_Threshold = tk.Button(top,text='Open Window to Preview Thresholding Outcomes',command=Threshold_And_View,width=40)
    original_height = DoubleVar()
    original_height.set(400)
    Threshold_Menu_Drop = DoubleVar()
    Threshold_Menu_Drop.set(0)
    Threshold_Reset_Offset = DoubleVar()
    Threshold_Reset_Offset.set(1000)
    Deriv_Menu_Offset = DoubleVar()
    Deriv_Menu_Offset.set(112)
    Deriv_Reset_Offset = DoubleVar()
    Deriv_Reset_Offset.set(1000)
    Deriv_x_Label = Label(top,text='Order of dx:')
    Deriv_x_Value = StringVar()
    Deriv_x_Value.set('0')
    Deriv_x_Entry = Entry(top,textvariable=Deriv_x_Value,width=3)
    Deriv_y_Label = Label(top,text='Order of dy:')
    Deriv_y_Value = StringVar()
    Deriv_y_Value.set('0')
    Deriv_y_Entry = Entry(top,textvariable=Deriv_y_Value,width=3)
    Deriv_z_Label = Label(top,text='Order of dz:')
    Deriv_z_Value = StringVar()
    Deriv_z_Value.set('1')
    Deriv_z_Entry = Entry(top,textvariable=Deriv_z_Value,width=3)
    Column_Finding_Thresholding_Label = Label(top,text='Initial Standard Deviation to Find Columns:')
    Column_Finding_Thresholding_Value = StringVar()
    Column_Finding_Thresholding_Value.set('7')
    Column_FInding_Thresholding_Entry = Entry(top,textvariable=Column_Finding_Thresholding_Value,width=5)
    Column_Thresholding_Label = Label(top,text='Percentage of Max Value in Each Column:')
    Column_Thresholding_Value = StringVar()
    Column_Thresholding_Value.set('1.0')
    Column_Thresholding_Entry = Entry(top,textvariable=Column_Thresholding_Value,width=5)
    Column_Thresholding_Minimum_Size_Label = Label(top,text='Minimum Pixel Count for Columns:')
    Column_Thresholding_Minimum_Size_Value = StringVar()
    Column_Thresholding_Minimum_Size_Value.set('150')
    Column_Thresholding_Minimum_Size_Entry = Entry(top,textvariable=Column_Thresholding_Minimum_Size_Value,width=10)
    Reposition_Errors = IntVar()
    Final_Selection_Made_Via_Button = IntVar()
    Threshold_Prior_Selection_Check = StringVar()
    Threshold_Prior_Selection_Check.set('Select One')
    Final_Selection_Avoid_Double_Sending = StringVar()
    Final_Selection_Avoid_Double_Sending.set('No')
    Initializing_Check = IntVar()
    Edge_Removal_Value = StringVar()
    Edge_Removal_Value.set('(Optional)')
    Edge_Removal_Label = Label(top,text='Number of Edge Pixels to Ignore:')
    Edge_Removal_Entry = Entry(top,textvariable=Edge_Removal_Value,width=10)
    
    Input_Directory_Display_Text = canvas1.create_window(150,75,window=Input_Directory_Button)
    Input_Directory_Display_Entry_Window = canvas1.create_window(532,74,window=Input_Directory_Text_Entry)
    Specific_Information_Recon_Input_Directory_Button_Canvas = canvas1.create_window(813,74,window=Specific_Information_Recon_Input_Directory_Button)
    Deriv_Display_Text = canvas1.create_window(150,128,window=Deriv_Choice_Label)
    Deriv_Display_Menu = canvas1.create_window(304,128,window=Deriv_Menu,width=101)
    Deriv_Display_question_mark_Button_Canvas = canvas1.create_window(363,128,window=Specific_Information_Deriv_Use_Button)
    
    Deriv_Output_Directory_Text = StringVar()
    Deriv_Output_Directory_Text.set('(Optional)')
    Deriv_Output_Directory_Button = tk.Button(top,text='Select Derivative Output Directory',command=Deriv_Output_Directory,width=29)
    Deriv_Output_Directory_Text_Entry = Entry(top,textvariable=Deriv_Output_Directory_Text,width=90)
    Threshold_Display_Text = canvas1.create_window(150,130+Deriv_Menu_Offset.get(),window=Threshold_Menu_Label)
    Threshold_Display_Menu = canvas1.create_window(304,130+Deriv_Menu_Offset.get(),window=Threshold_Menu,width=101)
    Specific_Information_Threshold_Use_Button_Canvas = canvas1.create_window(362,130+Deriv_Menu_Offset.get(),window=Specific_Information_Threshold_Use_Button)
    Output_Directory_Button_Canvas = canvas1.create_window(150,466,window=Output_Directory_Button)
    Output_Directory_Text_Entry_Canvas = canvas1.create_window(532,466,window=Output_Directory_Text_Entry)
    Specific_Information_Finished_Output_Button_Canvas = canvas1.create_window(813,466,window=Specific_Information_Finished_Output_Button)
    Finished_Selection_Display_Button = canvas1.create_window(385,520,window=Finish_Selections_Button)
    
    
    Debugg_Mode = False
    
    if Debugg_Mode:

        def button_mover(*args):
            #moving_test_icon_canvas = canvas1.create_window(gapp_displayx.get(),gapp_displayy.get(),window=moving_test_icon)
            test_canvas = canvas1.create_window(gapp_displayx.get(),gapp_displayy.get(),window=Incorrect_Deriv_Choice_Error_Symbol)
            test_canvas1 = canvas1.create_window(gapp_displayx1.get(),gapp_displayy1.get(),window=test1)
        
        test = Label(top,text='Error message test for placing in specific function information window',bg='white')
        test_canvas = canvas1.create_window(1125,400,window=Incorrect_Deriv_Choice_Error_Symbol)
        gapp_numberx = IntVar()
        gapp_numberx.set(1125)
        gapp_numberx.trace('w',button_mover)
        gapp_numbery = IntVar()
        gapp_numbery.set(350)
        gapp_numbery.trace('w',button_mover)
        gapp_displayx = Entry(top,textvariable=gapp_numberx,width=5)
        gapp_displayx_canvas = canvas1.create_window(50,400,window=gapp_displayx)
        gapp_displayy = Entry(top,textvariable=gapp_numbery,width=5)
        gapp_displayy_canvas = canvas1.create_window(100,400,window=gapp_displayy)
        test_icon_width=IntVar()
        test_icon_width.set(0)
        #moving_test_icon = tk.Button(top,text='?',width=test_icon_width.get())
        #moving_test_icon_canvas = canvas1.create_window(gapp_displayx.get(),gapp_displayy.get(),window=moving_test_icon)
        
        #test1 = Label(top,text='Error message test for placing in specific function information window',bg='white')
        test1 = Label(top,text='Correct format is "x,y,z" with positive numbers only  (>0)')
        test_canvas = canvas1.create_window(1125,450,window=test1)
        gapp_numberx1 = IntVar()
        gapp_numberx1.set(1125)
        gapp_numberx1.trace('w',button_mover)
        gapp_numbery1 = IntVar()
        gapp_numbery1.set(400)
        gapp_numbery1.trace('w',button_mover)
        gapp_displayx1 = Entry(top,textvariable=gapp_numberx1,width=5)
        gapp_displayx_canvas1 = canvas1.create_window(50,450,window=gapp_displayx1)
        gapp_displayy1 = Entry(top,textvariable=gapp_numbery1,width=5)
        gapp_displayy_canvas1 = canvas1.create_window(100,450,window=gapp_displayy1)
        #moving_test_icon1 = tk.Button(top,text='?',width=test_icon_width1.get())
        #moving_test_icon_canvas1 = canvas1.create_window(gapp_displayx1.get(),gapp_displayy1.get(),window=moving_test_icon)
        test_icon_width1=IntVar()
        test_icon_width1.set(0)
    
    Initializing_Check.set(1)
    top.title("DHM Pipeline")
    w = 1425 # width for the Tk root
    h = 550 # height for the Tk root

    # get screen width and height
    ws = top.winfo_screenwidth() # width of the screen
    hs = top.winfo_screenheight() # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)

    # set the dimensions of the screen 
    # and where it is placed
    top.geometry('%dx%d+%d+%d' % (w, h, x, y))
    top.mainloop()