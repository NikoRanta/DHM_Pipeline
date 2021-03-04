from skimage import io
import numpy as np
import timeit
    
def Derivatives(order,F,begin_timer):
    deriv_axis = []
    
    dF = np.zeros(F.shape)
    First_Derivative_Check = 0
    for derivatives in range(3):
        order_axis = np.zeros(3)
        if derivatives == 0:
            order_axis[1] = 1
        if derivatives == 1:
            order_axis[2] = 1
        if derivatives == 2:
            order_axis[0] = 1
        if order[derivatives] > 0:
            for derivs in range(order[derivatives]):
                if First_Derivative_Check != 0:
                    Upper = F[0:int(dF.shape[0]-2*order_axis[0]),0:int(dF.shape[1]-2*order_axis[1]),0:int(dF.shape[2]-2*order_axis[2])]
                    Middle = F[int(1*order_axis[0]):int(dF.shape[0]-1*order_axis[0]),int(1*order_axis[1]):int(dF.shape[1]-1*order_axis[1]),int(1*order_axis[2]):int(dF.shape[2]-1*order_axis[2])]
                    Lower = F[int(2*order_axis[0]):dF.shape[0],int(2*order_axis[1]):dF.shape[1],int(2*order_axis[2]):dF.shape[2]]
                    dF = -Upper+Middle+Lower
                if First_Derivative_Check == 0:
                    First_Derivative_Check = 1
                    Upper = F[0:int(F.shape[0]-2*order_axis[0]),0:int(F.shape[1]-2*order_axis[1]),0:int(F.shape[2]-2*order_axis[2])]
                    Middle = F[int(1*order_axis[0]):int(F.shape[0]-1*order_axis[0]),int(1*order_axis[1]):int(F.shape[1]-1*order_axis[1]),int(1*order_axis[2]):int(F.shape[2]-1*order_axis[2])]
                    Lower = F[int(2*order_axis[0]):F.shape[0],int(2*order_axis[1]):F.shape[1],int(2*order_axis[2]):F.shape[2]]
                    dF = -Upper+Middle+Lower


            c5 = np.array([-3,12,17,12,-3])/35
            c7 = np.array([-2,3,6,7,6,3,-2])/21
            for smooth in range(order[derivatives]+1):
                if deriv_axis == 'x':
                    dF[2,:,:] = (dF[1,:,:]+dF[2,:,:]+dF[3,:,:])/3
                    dF[dF.shape[0]-3,:,:] = (dF[dF.shape[0]-4,:,:]+dF[dF.shape[0]-3,:,:]+dF[dF.shape[0]-2,:,:])/3
                    dF[3,:,:] = c5[0]*dF[1,:,:]+c5[1]*dF[2,:,:]+c5[2]*dF[3,:,:]+c5[3]*dF[4,:,:]+c5[4]*dF[5,:,:]
                    dF[dF.shape[0]-4,:,:] = c5[0]*dF[dF.shape[0]-6,:,:]+c5[1]*dF[dF.shape[0]-5,:,:]+c5[2]*dF[dF.shape[0]-4,:,:]+c5[3]*dF[dF.shape[0]-3,:,:]+c5[4]*dF[dF.shape[0]-2,:,:]
                    dF[4:dF.shape[0]-4,:,:] =  c7[0]*dF[1:dF.shape[0]-7,:,:]+c7[1]*dF[2:dF.shape[0]-6,:,:]+c7[2]*dF[3:dF.shape[0]-5,:,:]+c7[3]*dF[4:dF.shape[0]-4,:,:]+c7[4]*dF[5:dF.shape[0]-3,:,:]+c7[5]*dF[6:dF.shape[0]-2,:,:]+c7[6]*dF[7:dF.shape[0]-1,:,:]
                if deriv_axis == 'y':
                    dF[:,2,:] = (dF[:,1,:]+dF[:,2,:]+dF[:,3,:])/3
                    dF[:,dF.shape[1]-3,:] = (dF[:,dF.shape[1]-4,:]+dF[:,dF.shape[1]-3,:]+dF[:,dF.shape[1]-2,:])/3
                    dF[:,3,:] = c5[0]*dF[:,1,:]+c5[1]*dF[:,2,:]+c5[2]*dF[:,3,:]+c5[3]*dF[:,4,:]+c5[4]*dF[:,5,:]
                    dF[:,dF.shape[1]-4,:] = c5[0]*dF[:,dF.shape[1]-6,:]+c5[1]*dF[:,dF.shape[1]-5,:]+c5[2]*dF[:,dF.shape[1]-4,:]+c5[3]*dF[:,dF.shape[1]-3,:]+c5[4]*dF[:,dF.shape[1]-2,:]
                    dF[:,4:dF.shape[1]-4,:] =  c7[0]*dF[:,1:dF.shape[1]-7,:]+c7[1]*dF[:,2:dF.shape[1]-6,:]+c7[2]*dF[:,3:dF.shape[1]-5,:]+c7[3]*dF[:,4:dF.shape[1]-4,:]+c7[4]*dF[:,5:dF.shape[1]-3,:]+c7[5]*dF[:,6:dF.shape[1]-2,:]+c7[6]*dF[:,7:dF.shape[1]-1,:]
                if deriv_axis == 'z':
                    dF[:,:,2] = (dF[:,:,1]+dF[:,:,2]+dF[:,:,3])/3
                    dF[:,:,dF.shape[2]-3] = (dF[:,:,dF.shape[2]-4]+dF[:,:,dF.shape[2]-3]+dF[:,:,dF.shape[2]-2])/3
                    dF[:,:,3] = c5[0]*dF[:,:,1]+c5[1]*dF[:,:,2]+c5[2]*dF[:,:,3]+c5[3]*dF[:,:,4]+c5[4]*dF[:,:,5]
                    dF[:,:,dF.shape[2]-4] = c5[0]*dF[:,:,dF.shape[2]-6]+c5[1]*dF[:,:,dF.shape[2]-5]+c5[2]*dF[:,:,dF.shape[2]-4]+c5[3]*dF[:,:,dF.shape[2]-3]+c5[4]*dF[:,:,dF.shape[2]-2]
                    dF[:,:,4:dF.shape[2]-4] =  c7[0]*dF[:,:,1:dF.shape[2]-7]+c7[1]*dF[:,:,2:dF.shape[2]-6]+c7[2]*dF[:,:,3:dF.shape[2]-5]+c7[3]*dF[:,:,4:dF.shape[2]-4]+c7[4]*dF[:,:,5:dF.shape[2]-3]+c7[5]*dF[:,:,6:dF.shape[2]-2]+c7[6]*dF[:,:,7:dF.shape[2]-1]
    return dF
    
'''
order = np.array([1,0,0])
F = np.random.randint(1,10,(50,2048,2048))
begin_timer = timeit.default_timer()
F = Derivatives(order,F,begin_timer)
end_timer = timeit.default_timer()
print(end_timer-begin_timer)
'''