import glob
from typing import final
import cv2
import numpy as np

from scipy import optimize

def v(H,i,j):


    v = np.array([[H[0][i]*H[0][j]],
                  [H[0][i]*H[1][j] + H[0][j]*H[1][i]],
                  [H[1][i]*H[1][j]],
                  [H[2][i]*H[0][j] + H[0][i]*H[2][j]],
                  [H[2][i]*H[1][j] + H[1][i]*H[2][j]],
                  [H[2][i]*H[2][j]]])


    return np.reshape(v,(6,1))

def get_K(all_Homographies):

    V_stacked=np.ones([1,6])

    for homography in all_Homographies:

        V = np.vstack([v(homography,0,1).T,
                      (v(homography,0,0)-v(homography,1,1)).T])   ### according to section 3.1 in paper
        V_stacked=np.concatenate([V_stacked,V],axis=0)

    V_stacked=V_stacked[1:]

    ##Finding the SVD to get b

    U,S,V_final=np.linalg.svd(V_stacked)


    b=V_final[np.argmin(S)]

    ##### Constructing the K Matrix from formulas from Appendix B#################

    B11 = b[0]
    B12 = b[1]
    B13 = b[3]

    B21 = b[1]
    B22 = b[2]
    B23 = b[4]
    
    B31 = b[3]
    B23 = b[4]
    B33 = b[5]

    


    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    lam = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11



    alpha = np.sqrt(lam/B11)
    beta = np.sqrt(lam*B11 /(B11*B22 - B12**2))
    gamma = -1*B12*(alpha**2)*beta/lam
    u0 = gamma*v0/beta -B13*(alpha**2)/lam
    K = np.array([[alpha, gamma, u0],[0, beta, v0],[0, 0, 1]])




    return K

def getRotationMatrices(K,all_Homographies):

    K_inverse=np.linalg.inv(K)

    all_extrensic_matrices=[]


    for homography in all_Homographies:

        h1=homography[:,0]
        h2=homography[:,1]
        h3=homography[:,2]

        r1=np.matmul(K_inverse,h1)
        r2=np.matmul(K_inverse,h2)

        norm1=np.linalg.norm(r1)
        lamd=1/norm1

        r1=lamd*r1
        r2=lamd*r2

        r3=np.cross(r1,r2)

        t=lamd*np.matmul(K_inverse,h3)

    

        ####################

        Q=np.array([r1,r2,r3]).T

        ### This is done so at to convert the R into a proper rotation matrix######

        u,s,v =np.linalg.svd(Q)



        R = np.matmul(u,v)


        extrensic_matrix=np.hstack((R,t.reshape(3,1)))

        all_extrensic_matrices.append(extrensic_matrix)

    all_extrensic_matrices=np.array(all_extrensic_matrices)


    return all_extrensic_matrices
        


def getProjectionError(K,all_homographies,all_extrensic_matrics,image_points,world_points,k1,k2):


    error=[]

    projected_points=[]

    u0=K[0,2]
    v0=K[1,2]

    for idx,homography in enumerate(all_homographies):

        image_projected_points=[]

        er=0
        extrensic_matrix=all_extrensic_matrics[idx]
        # mat=np.matmul(K,extrensic_matrix)


        for pt,world_pt in zip(image_points[idx],world_points):


            pt_mat=np.array([[world_pt[0]],[world_pt[1]],[0],[1]])

            ###According to the pinhole model
            x_y_arr=np.matmul(extrensic_matrix,pt_mat)

            x_y_arr=x_y_arr/x_y_arr[2]

            x,y=x_y_arr[0],x_y_arr[1]

            final_mat=np.matmul(K,x_y_arr)
            final_mat=final_mat/final_mat[2]

            u,v=final_mat[0],final_mat[1]

            squared=x**2+y**2

            u_dash=u+(u-u0)*(k1*squared+k2*(squared**2))
            v_dash=v+(v-v0)*(k1*squared+k2*(squared**2))

            image_projected_points.append([u_dash,v_dash])

            # print(pt[0][0])



            er = er + np.sqrt((pt[0][0]-u_dash)**2 + (pt[0][1]-v_dash)**2)


            

        error.append(er/54)
        projected_points.append(image_projected_points)



    error=np.array(error)
    

        

    
    return error,projected_points

    




def getHomographies(img_list,world_points):

    all_Homographies=[]
    all_corner_points=[]

    for image in img_list:

        grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        retval,corners=cv2.findChessboardCorners(grayscale,(9,6),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        

        all_corner_points.append(corners)


        four_point_matrix=[]


        for i in range(54):

            cv2.circle(image,(int(corners[i][0][0]),int(corners[i][0][1])),10,[255,0,0],-1) ## corresponding world points for i=0 is (21.5,21.5)

            # if i in [0,8,53,45]:
            four_point_matrix.append(corners[i][0])

        four_point_matrix=np.array(four_point_matrix)

        # four_point_matrix[[2,3]]=four_point_matrix[[3,2]]


        H_image_world,_=cv2.findHomography(world_points,four_point_matrix)
        

        all_Homographies.append(H_image_world)



    all_Homographies=np.array(all_Homographies)

    # print(all_corner_points.shape)







    return all_Homographies,all_corner_points

def optimizationFunction(init_parameters,K,all_Homographies,all_extrensic_matrices,all_corner_points,all_world_points):

    K=np.zeros(shape=(3,3))
    K[0, 0], K[0, 1], K[0, 2], K[1, 1], K[1, 2],k1,k2=init_parameters
    K[2,2]=1

    u0,v0=init_parameters[2],init_parameters[4]

    projection_error=[]



    for idx,extresnic_matrix in enumerate(all_extrensic_matrices):

        for pt,worldpt in zip(all_corner_points[idx],all_world_points):

            pt_mat=np.array([[worldpt[0]],[worldpt[1]],[0],[1]])

            ###According to the pinhole model
            x_y_arr=np.matmul(extresnic_matrix,pt_mat)

            x_y_arr=x_y_arr/x_y_arr[2]

            x,y=x_y_arr[0],x_y_arr[1]

            final_mat=np.matmul(K,x_y_arr)
            final_mat=final_mat/final_mat[2]

            u,v=final_mat[0],final_mat[1]

            squared=x**2+y**2

            u_dash=u+(u-u0)*(k1*squared+k2*(squared**2))
            v_dash=v+(v-v0)*(k1*squared+k2*(squared**2))

            projection_error.append(pt[0][0]-u_dash)
            projection_error.append(pt[0][1]-v_dash)


    projection_error=np.array(projection_error).flatten()

    # print(projection_error.shape)
    # print('Optimizing on parameters ...')

    
    return projection_error


def reprojectAllImages(img_list,projected_points,all_corner_points):

    for i in range(len(img_list)):

        image=img_list[i]

        for projected_point,pt in zip(projected_points[i],all_corner_points[i]):


            x,y=int(pt[0][0]),int(pt[0][1])
            x_rp,y_rp=int(projected_point[0]),int(projected_point[1])

            cv2.circle(image,(x,y),2,(255,0,0),-1)  ##Blue
            cv2.circle(image,(x_rp,y_rp),5,(0,0,255),-1) ### Red


        cv2.imwrite(f"Output_images/{i}.jpg",image)

        # break



    # pass









def doAll():


    img_files=glob.glob('../Calibration_Imgs/*.jpg')
    img_files.sort()

    print(img_files[0])

    img_list=[cv2.imread(file) for file in img_files]

    x,y=np.meshgrid(range(9),range(6))
    all_world_points = np.hstack((x.reshape(54, 1), y.reshape(
        54, 1))).astype(np.float32)
    all_world_points=all_world_points*21.5


    all_Homographies,all_corner_points=getHomographies(img_list,all_world_points)
    K=get_K(all_Homographies)

    print(f'The initial K is : \n {K}')

    all_extrensic_matrices=getRotationMatrices(K,all_Homographies)


    
    

    error,_=getProjectionError(K,all_Homographies,all_extrensic_matrices,all_corner_points,all_world_points,0,0)

    print(f'Initial Error is: {np.mean(error)}')

    init_parameters=[K[0, 0], K[0, 1], K[0, 2], K[1, 1], K[1, 2], 0, 0]

    final_parameters=optimize.least_squares(fun=optimizationFunction,x0=init_parameters,method="lm",args=[K,all_Homographies,all_extrensic_matrices,all_corner_points,all_world_points])

    # print(final_parameters)

    K_optimized=np.zeros((3,3))

    K_optimized[0, 0], K_optimized[0, 1], K_optimized[0, 2], K_optimized[1, 1], K_optimized[1, 2],K_optimized[2,2]=final_parameters.x[0],final_parameters.x[1],final_parameters.x[2],final_parameters.x[3],final_parameters.x[4],1

    k1_optimized=final_parameters.x[5]
    k2_optimized=final_parameters.x[6]

    print(f'The optimized K is : \n {K_optimized}')

    print(f'The optimized k1 and k2 are: {k1_optimized} and {k2_optimized}')

    error,projected_points=getProjectionError(K_optimized,all_Homographies,all_extrensic_matrices,all_corner_points,all_world_points,k1_optimized,k2_optimized)
    print(f'Error After optimization is: {np.mean(error)}')


    reprojectAllImages(img_list,projected_points,all_corner_points)


doAll()