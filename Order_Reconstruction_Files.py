import os
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage import io
import timeit


def Order_Holograms_After_Reconstruction(folderPath):
    
    File_Names = np.array([(f,float(f.name)) for f in os.scandir(folderPath)])
    #Descending_Order = File_Names[(-File_Names[:,1]).argsort()]
    Ascending_Order = File_Names[File_Names[:,1].argsort()]
    file_names_combined = np.zeros((len(Ascending_Order),len(listdir(Ascending_Order[0,0].path)))).astype(np.unicode_)
    Organized_Files = []
    for x in range(len(Ascending_Order)):
        onlyfiles = [onlyfiles for onlyfiles in listdir(Ascending_Order[x,0].path) if isfile(join(Ascending_Order[x,0],onlyfiles))]
        z_slice_time_stamp_names = np.array([onlyfiles[x].split('.') for x in range(len(onlyfiles))])
        file_names_combined[x] = np.array([z_slice_time_stamp_names[z_slice_time_stamp_names[:,0].astype(np.float).argsort()][x][0]+'.'+z_slice_time_stamp_names[z_slice_time_stamp_names[:,0].astype(np.float).argsort()][x][1] for x in range(len(onlyfiles))])
        
    for x in range(len(file_names_combined[:,0])):
        for y in range(len(file_names_combined[0,:])):
            Organized_Files.append(Ascending_Order[x,0].path+'/'+str(file_names_combined[x,y]))
    Organized_Files = np.array(Organized_Files).reshape((len(file_names_combined[:,0]),len(file_names_combined[0,:])))
    
    return Organized_Files#,Descending_Order

#A =  Order_Holograms_After_Reconstruction('H:/laser-lasso_backup/elec/2019.08.31_08.00.01.92/bug_of_interest/Phase/')
#A = Order_Holograms_After_Reconstruction('F:/CURRENT_TEST_SET_(12.04.20)_2019.05.01_15-21/Niko_Test/Phase')

'''
folderPath = 'D:/DHM_Analysis/Delete/'
try:
    File_Names = np.array([(f,float(f.name)) for f in os.scandir(folderPath)])
except ValueError:
    print('Bad Folder Names')
else:
    print(File_Names)
    print('Valid Folder Names')

filePath = 'D:/DHM_Analysis/00001.tif'
'''
'''
try:
    image = io.imread(filePath)
except:
    print('Invalid Hologram File')
else:
    print('Valid Hologram')
'''

'''
File_Names = np.array([(f,float(f.name)) for f in os.scandir(folderPath)])
Descending_Order = File_Names[(-File_Names[:,1]).argsort()]
try:
    [[onlyfiles.split('.')[0],onlyfiles.split('.')[1]] for onlyfiles in listdir(Descending_Order[0,0].path) if isfile(join(Descending_Order[0,0],onlyfiles))]
except:
    print('Invalid Hologram File')
else:
    O = [[onlyfiles.split('.')[0],onlyfiles.split('.')[1]] for onlyfiles in listdir(Descending_Order[0,0].path) if isfile(join(Descending_Order[0,0],onlyfiles))]
    if len(O) == 0:
        print('No Holograms Present')
    if len(O)>0:
        print('Valid Hologram')
        A = np.zeros(len(O))
        B = np.zeros(len(O)).astype(np.unicode_)
        issues_found = 0
        start = timeit.default_timer()
        for x in range(len(O)):
            proceed = 0
            try:
                A[x] = O[x][0]
            except ValueError:
                issues_found = 1
                print('Invalid File Name Found')
                break
            if O[x][1] != 'tif':
                issues_found = 1
                print('Invalid File Type Found')
                break
        if issues_found == 0:
            print('All Valid Files Found')
        end = timeit.default_timer()
        print(end-start)
'''