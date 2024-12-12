import numpy as np
import copy
from matplotlib import pyplot as plt
from math_feeg6043 import Vector,Matrix,Identity,Transpose,Inverse,v2t,t2v,HomogeneousTransformation, interpolate, short_angle, inward_unit_norm, line_intersection, cartesian2polar, polar2cartesian
from plot_feeg6043 import plot_kalman, show_information
from scipy.linalg import cholesky

def rigid_body_kinematics(mu,u,dt=0.1,mu_gt=None,sigma_motion=Matrix(3,2),sigma_xy=Matrix(3,3)):
    #################################################################################
    # This function models the forwards kinematics of rigid body, b
    # It implements a model of form
    #  mu_k = f(mu_k-1,u_k)
    # and progresses the pose of the robot 'mu' (in the fixed frame e)) 
    # to the next timestep 'mu_' by applying combined linear and angular velocity (twist) 
    # as its control action 'u' for the period 'dt'. 
    #
    # The notation of 'mu' is used as it represents the mean estimate of the robot pose p
    #
    # It implements the homogeneous transformation below to achieve this.
    # 
    # Xk = Xk-1 Trans(tbc)Rot(wdt)Trans(tcb')
    #
    # as 
    #
    # H_eb' = H_eb @ H_bc.H_T @ H_cb'.H_R @ H_cb'.H_T
    #
    # Inputs
    #.  mu: The previous pose p_{k-1}, which is a [3x1] matrix of form
    #          [[x]     Northings in m (fixed frame, e )
    #           [y]     Eastings in m (fixed frame, e )
    #           [g]]    Heading in rads (fixed frame, e )
    #.  u:  The control is a [2 x 1] matrix of the form, 
    #         [[v]      Linear forwards velocity in m/s (body frame, b)
    #.         [w]]     Angular velocity in rads/s (body frame, b)
    #   mu_gt: if a noise model is provided, mu has noise and mu_gt is the noise free version of the motion output
    #   sigma_motion: Noise coefficients that apply to control inputs [3x2]
    #   sigma_xy: Pre-existing noise on states [3x3]
    #
    # Output
    #.  mu: The new pose p_k
    #
    # Dependancies 
    #.  standard python libraries (numpy)
    #.  class HomogeneousTransformation from course library `math_feeg6043.py`
    #################################################################################    
 
    # create an empty containor for the output
    mu_=Vector(3)

    # create an empty containor for the output homogeneous transformation matrix     
    H_eb_ = HomogeneousTransformation()
    
    # convert input pose p_k-1 into a homogeneous transformation matrix 
    t_eb=Vector(2)
    t_eb[0] = mu[0] # northings (p_k-1)
    t_eb[1] = mu[1] #eastings (p_k-1)
    g_eb = mu[2] #heading (p_k-1)
    H_eb = HomogeneousTransformation(t_eb,g_eb)
    H_eb_gt =  HomogeneousTransformation()            

    # convert input pose ground truth p_k-1 into a homogeneous transformation matrix 
    if np.all(mu_gt != None):
        t_ebgt=Vector(2)
        t_ebgt[0] = mu_gt[0] # northings (p_k-1)
        t_ebgt[1] = mu_gt[1] #eastings (p_k-1)
        g_ebgt = mu_gt[2] #heading (p_k-1)
        H_ebgt = HomogeneousTransformation(t_ebgt,g_ebgt)      
        H_eb_gt =  HomogeneousTransformation()           

    u_hat = Vector(2) # create vector to add noise to    

    # Only add noise if there is control and noise
    if np.all(sigma_motion == 0.0) == False and np.all(u == 0.0) == False:    
        sigma=(sigma_motion@u)        
        sigma_u=Matrix(3,3)

        sigma_u[0,0] = sigma[0]*dt
        sigma_u[1,1] = sigma[1]*dt
        sigma_u[2,2] = sigma[2]*dt
        
        u_hat[0]=u[0]+np.random.normal(0, (sigma[0]), 1)
        u_hat[1]=u[1]+np.random.normal(0, (sigma[1]), 1)
        gamma_noise = np.random.normal(0, (sigma[2]), 1)
        
        #calculate the noise component
        J = Matrix(3,3)

        dx_dx = 1
        dx_dy = 0
        dx_dg = (u[0]/u[1])*(-np.cos(g_eb)+np.cos(g_eb+u[1]*dt))
        dy_dx = 0
        dy_dy = 1
        dy_dg = (u[0]/u[1])*(-np.sin(g_eb)+np.sin(g_eb+u[1]*dt))
        dg_dx = 0
        dg_dy = 0       
        dg_dg = 1
                
        J[0,0] = dx_dx
        J[0,1] = dx_dy
        J[0,2] = dx_dg        
        J[1,0] = dy_dx
        J[1,1] = dy_dy
        J[1,2] = dy_dg        
        J[2,0] = dg_dx
        J[2,1] = dg_dy
        J[2,2] = dg_dg        

        sigma_xy = J@sigma_u@J.T+sigma_xy

    else:
        u_hat = u
        gamma_noise = 0


    if u_hat[0]==0 and u_hat[1]==0:
        # handles the stationary case where 
        H_bb_ = HomogeneousTransformation(Vector(2),0)
        H_eb_ = H_eb

        if np.all(mu_gt != None):
            H_eb_gt = H_ebgt

    else:
        if u_hat[1]==0:        
            # implement a simpler vesion of the model that doesn't need to compute twist 
            # H_eb_ = H_eb@H_bb' 

            v = u[0] #surge rate        
                
            # compute motion in the body frame due to the pure linear velocity
            t_bb_=Vector(2) # [2x1] matrix of 0
            t_bb_[0] = v*dt
            
            # create the homogeneous transformation from b to b', 
            H_bb_ = HomogeneousTransformation(t_bb_,0) # heading doesn't change as w=0 (u[1]=0]
            
            # left multiply be the homogeneous transformation from the fixed frame e to the body frame b
            H_eb_.H = H_eb.H@H_bb_.H  

            if np.all(mu_gt != None):  
                v_gt = u_hat[0]        

                t_bb_gt=Vector(2) # [2x1] matrix of 0
                t_bb_gt[0] = v_gt*dt
                
                # create the homogeneous transformation from b to b', 
                H_bb_gt = HomogeneousTransformation(t_bb_gt,0) # heading doesn't change as w=0 (u[1]=0]
                
                # left multiply be the homogeneous transformation from the fixed frame e to the body frame b
                H_eb_gt.H = H_ebgt.H@H_bb_gt.H  

                
        else: 
            # implements the model derived for twist
            
            v = u[0] #surge rate
            w = u[1] #yaw rate

            # calculate centre of rotation from the initial body position
            t_bc = Vector(2) # [2x1] matrix of 0                       
            t_bc[1]=v/w      # centre of rotation is v/w in the +ve y direction of the body Eq(A1.2.12)
            
            # the centre of rotation 'c' keeps the heading of the body, so is 0 as seen from the body
            H_bc = HomogeneousTransformation(t_bc,0)
                
            # to rotate the body about 'c' so need 'b' as seen from the centre of rotation
            H_cb = HomogeneousTransformation() 
            H_cb.H = Inverse(H_bc.H)  

            # to rotate the body b around the centre of rotation w*dt while maintain the same radius
            H_cb_ = HomogeneousTransformation(H_cb.t,w*dt)
            
            # we rotate first, and then translate
            H_eb_.H = H_eb.H@H_bc.H@H_cb_.H_R@H_cb_.H_T                      
            
            if np.all(mu_gt != None): 
                v_gt = u_hat[0] #surge rate
                w_gt = u_hat[1] #yaw rate

                # calculate centre of rotation from the initial body position
                t_bcgt = Vector(2) # [2x1] matrix of 0                       
                t_bcgt[1]=v_gt/w_gt      # centre of rotation is v/w in the +ve y direction of the body Eq(A1.2.12)
                
                # the centre of rotation 'c' keeps the heading of the body, so is 0 as seen from the body
                H_bcgt = HomogeneousTransformation(t_bcgt,0)
                    
                # to rotate the body about 'c' so need 'b' as seen from the centre of rotation
                H_cbgt = HomogeneousTransformation() 
                H_cbgt.H = Inverse(H_bcgt.H)  

                # to rotate the body b around the centre of rotation w*dt while maintain the same radius
                H_cb_gt = HomogeneousTransformation(H_cbgt.t,w_gt*dt)
                
                # we rotate first, and then translate
                H_eb_gt.H = H_ebgt.H@H_bcgt.H@H_cb_gt.H_R@H_cb_gt.H_T                      

        # store the odometry, which is the relative change in pose resulting from control motion
        H_bb_ = HomogeneousTransformation()
        H_bb_.H =  Inverse(H_eb.H)@H_eb_.H        

        
    # create the pose, handling angle wrap at 2pi (in rads)
    mu_[0] = H_eb_.t[0]
    mu_[1] = H_eb_.t[1]
    mu_[2] = H_eb_.gamma % (2 * np.pi ) #(H_eb_.gamma + np.pi) % (2 * np.pi ) - np.pi 

    mu_gt=Vector(3)
    dmu_ = Vector(3)
    dmu_[0] = H_bb_.t[0]
    dmu_[1] = H_bb_.t[1]
    dmu_[2] = (H_bb_.gamma + np.pi) % (2 * np.pi ) - np.pi        

    if np.all(mu_gt != None):     
        mu_gt[0] = H_eb_gt.t[0]
        mu_gt[1] = H_eb_gt.t[1]
        mu_gt[2] = (H_eb_gt.gamma + gamma_noise*dt + np.pi) % (2 * np.pi ) - np.pi

        # dmu_[0] = H_bb_.t[0]
        # dmu_[1] = H_bb_.t[1]
        # dmu_[2] = (H_bb_.gamma + np.pi) % (2 * np.pi ) - np.pi

    return mu_, sigma_xy, dmu_, mu_gt

# def rigid_body_kinematics(mu,u,dt=0.1):
#     #################################################################################
#     # This function models the forwards kinematics of rigid body, b
#     # It implements a model of form
#     #  mu_k = f(mu_k-1,u_k)
#     # and progresses the pose of the robot 'mu' (in the fixed frame e)) 
#     # to the next timestep 'mu_' by applying combined linear and angular velocity (twist) 
#     # as its control action 'u' for the period 'dt'. 
#     #
#     # The notation of 'mu' is used as it represents the mean estimate of the robot pose p
#     #
#     # It implements the homogeneous transformation below to achieve this.
#     # 
#     # Xk = Xk-1 Trans(tbc)Rot(wdt)Trans(tcb')
#     #
#     # as 
#     #
#     # H_eb' = H_eb @ H_bc.H_T @ H_cb'.H_R @ H_cb'.H_T
#     #
#     # Inputs
#     #.  mu: The previous pose p_{k-1}, which is a [3x1] matrix of form
#     #          [[x]     Northings in m (fixed frame, e )
#     #           [y]     Eastings in m (fixed frame, e )
#     #           [g]]    Heading in rads (fixed frame, e )
#     #.  u:  The control is a [2 x 1] matrix of the form, 
#     #         [[v]      Linear forwards velocity in m/s (body frame, b)
#     #.         [w]]     Angular velocity in rads/s (body frame, b)
#     # Output
#     #.  mu: The new pose p_k
#     #
#     # Dependancies 
#     #.  standard python libraries (numpy)
#     #.  class HomogeneousTransformation from course library `math_feeg6043.py`
#     #################################################################################    
 
#     # create an empty containor for the output
#     mu_=Vector(3)

#     # create an empty containor for the output homogeneous transformation matrix     
#     H_eb_ = HomogeneousTransformation()
    
#     # convert input pose p_k-1 into a homogeneous transformation matrix 
#     t_eb=Vector(2)
#     t_eb[0] = mu[0] # northings (p_k-1)
#     t_eb[1] = mu[1] #eastings (p_k-1)
#     g_eb = mu[2] #heading (p_k-1)
#     H_eb = HomogeneousTransformation(t_eb,g_eb)            

#     if u[0]==0 and u[1]==0:
#         # handles the stationary case where 
#         H_eb_ = H_eb

#     elif u[1]==0:        
#         # implement a simpler vesion of the model that doesn't need to compute twist 
#         # H_eb_ = H_eb@H_bb' 

#         v = u[0] #surge rate        
            
#         # compute motion in the body frame due to the pure linear velocity
#         t_bb_=Vector(2) # [2x1] matrix of 0
#         t_bb_[0] = v*dt
        
#         # create the homogeneous transformation from b to b', 
#         H_bb_ = HomogeneousTransformation(t_bb_,0) # heading doesn't change as w=0 (u[1]=0]
        
#         # left multiply be the homogeneous transformation from the fixed frame e to the body frame b
#         H_eb_.H = H_eb.H@H_bb_.H            
            
#     else: 
#         # implements the model derived for twist
        
#         v = u[0] #surge rate
#         w = u[1] #yaw rate

#         # calculate centre of rotation from the initial body position
#         t_bc = Vector(2) # [2x1] matrix of 0                       
#         t_bc[1]=v/w      # centre of rotation is v/w in the +ve y direction of the body Eq(A1.2.12)
        
#         # the centre of rotation 'c' keeps the heading of the body, so is 0 as seen from the body
#         H_bc = HomogeneousTransformation(t_bc,0)
            
#         # to rotate the body about 'c' so need 'b' as seen from the centre of rotation
#         H_cb = HomogeneousTransformation() 
#         H_cb.H = Inverse(H_bc.H)  

#         # to rotate the body b around the centre of rotation w*dt while maintain the same radius
#         H_cb_ = HomogeneousTransformation(H_cb.t,w*dt)
        
#         # we rotate first, and then translate
#         H_eb_.H = H_eb.H@H_bc.H@H_cb_.H_R@H_cb_.H_T                      
        
#     # create the pose, handling angle wrap at 2pi (in rads)
#     mu_[0] = H_eb_.t[0]
#     mu_[1] = H_eb_.t[1]
#     mu_[2] = H_eb_.gamma % (2 * np.pi ) #(H_eb_.gamma + np.pi) % (2 * np.pi ) - np.pi 
    
#     return mu_

class actuator_configuration_model:

    """
    Class to model kinematics of a 2 wheel differential drive
    Assumes symmetry about the vehicle centre line and the 
    body frame origin coincides with the centre of rotation
    
    Parameters:
    -----------
    W: float
        Perpendicular distance of each wheel to robot's x-axis centre line in metres
    d: float
        Diametre of each wheel in metres

    Attributes:
    -----------
    G: Matrix(2,2)
        Actuator configuration matrix [d/4 d/4; -d/2W d/W] 
    invG: Matrix(2,2)
        Inverse of actuator configuration matrix [2/d -W/d; 2/d W/d]     
        
    Functions:
    -----------
    inv_kinematics(self, u):
        Models the actuator inverse kinematics and computes wheel 
        commands to achieve desired rigid body twist 

        Input:
            u:  Body twist is a [2 x 1] matrix of the form
                  [[v]      Linear forwards velocity in m/s (body frame, b)
                   [w]]     Angular velocity in rads/s (body frame, b)
        Output:
            q: The wheel rates of each wheel in rad/s, of the form
               [[w_r]    wheel aligned with +ve y-axis of robot
                [w_l]]   wheel aligned with -ve y-axis of robot   

    fwd_kinematics(self, u): 
        Models the actuator kinematics and computes rigid body twist corresponding 
        to provided wheel commands      
        
        Input:
            q: The wheel rates of each wheel in rad/s, of the form
               [[w_r]    wheel aligned with +ve y-axis of robot
                [w_l]]   wheel aligned with -ve y-axis of robot   

        Output:
            u:  Body twist is a [2 x 1] matrix of the form 
                  [[v]      Linear forwards velocity in m/s (body frame, b)
                   [w]]     Angular velocity in rads/s (body frame, b)

    """    
    
    def __init__(self, W, d):
        
        self.W = W
        self.d = d
        self.G = Matrix(2,2)
        self.invG = Matrix(2,2)
        
        self.G[0,0] = d/4
        self.G[0,1] = d/4
        self.G[1,0] = -d/(2*W)
        self.G[1,1] = d/(2*W)
        
        self.invG[0,0] = 2/d
        self.invG[0,1] = -W/d                
        self.invG[1,0] = 2/d        
        self.invG[1,1] = W/d                        
    
    def inv_kinematics(self, u): q = self.invG @ u; return q
    def fwd_kinematics(self, q): u = self.G @ q ; return u

    
class RangeAngleKinematics():
    
    """ Class to model the geometry of range and angle sensor measurements
    
    Parameters:
    -----------
    x_bl: Float
        Location of the sensor in the body frame x-direction in m
    y_bl: Float
        Location of the sensor in the body frame y-direction in m        
    gamma_bs: Float
        Zero angle bearing of the sensor relative to the body frame x-direction in radians   
        Default is 0 (aligned with robot x-axis)
    distance_range: [2 x 1] Float list
        Minimum and maximum distance the sensor can make a measurement in m
        Default is min 0.1 m to max 1 m
    scan_fov: Float
        Maximum range of bearing angles where objects can be detected in rad. Assumed to be centred about gamma_bs with 0.5 scan_fov either side
        Default is np.deg2rad(120), which gives 60degrees either side of the robot

    Attributes:
    -----------
    t_bl: Matrix 2x1
        Location of the sensor $s$ relative to the moving body, defined as
            t_bl = [[x_bl]
                    [y_bl]]
    H_bl: Class HomogenousTranformation()
        Homogenous transformation class instance to map the 
        2D location and orientation of the range bearing sensor $l$ 
        relative to the moving body $b$, using x_bl, y_bl and gamma_bl
        
    Functions:
    -----------
    loc_to_rangeangle(self, p_eb, t_em):
        Models the sensor inverse kinematics to computes range bearing measurements 
        that correspond to the robot observing some map landmark

        Input:
            p_eb:  Pose of the robot in the fixed frame $e$ as a [3 x 1] matrix of the form
                  [[x_eb]      Northings distance in m (fixed frame, e)
                   [y_eb]      Eastings distance in m (fixed frame, e)
                   [gamma_eb]] Heading angle in rads (body frame, b)
                   
            t_em:  Coordinate of the map landmark in the fixed frame $e$ as a [2 x 1] matrix of the form
                  [[x_em]      Northings distance in m (fixed frame, e)
                   [y_em]      Eastings distance in m (fixed frame, e)
        Output:
            z_lm: Sensor Range and bearing observation
               [[r_lm]    range to the map landmark as seen from the sensor, l
                [theta_lm]]   bearing to the map landmark as seen from the sensor, l

    rangeangle_to_loc(self, p_eb, z_lm): 
        Models the sensor kinematics to compute the location of a map landmark (fixed frame, e)
        accoridng to the a sensor observation range and bearing.
        
        Input:
            z_lm: Sensor range and bearing observation
               [[r_lm]    range to the map landmark as seen from the sensor, l
                [theta_lm]]   bearing to the map landmark as seen from the sensor, l
            p_eb:  Pose of the robot in the fixed frame $e$ as a [3 x 1] matrix of the form
                  [[x_eb]      Northings distance in m (fixed frame, e)
                   [y_eb]      Eastings distance in m (fixed frame, e)
                   [gamma_eb]] Heading angle in rads (body frame, b)                
                
        Output:
            t_em:  Coordinate of the map landmark in the fixed frame $e$ as a [2 x 1] matrix of the form
                  [[x_em]      Northings distance in m (fixed frame, e)
                   [y_em]      Eastings distance in m (fixed frame, e)
                   
    check_range(self,z_lm):
        Checks that both the distance_range and the scan_fov are feasible. If not it returns np.nan values
        
        Inputs:
            z_lm: Sensor range and bearing observations as a [2x1] matrix (previously define)
        Output:
            z_lm: Sensor range and bearing observations as a [2x1] matrix (previously define)
                after checking it is within the sensor specification. If not returns np/nan values        
    """
    
    def __init__(self, x_bl, y_bl, gamma_bl = 0, distance_range = [0.1, 1], scan_fov = np.deg2rad(120)):
        
        self.t_bl = Vector(2)
        self.t_bl[0] = x_bl
        self.t_bl[1] = y_bl                
        self.H_bl = HomogeneousTransformation(self.t_bl,gamma_bl)
        self.distance_range = distance_range
        self.scan_fov = scan_fov   
        
        # print('*******************************************************************************************')
        # print('Sensor located at:',self.t_bl[0],self.t_bl[1],'m offset, at angle:',np.rad2deg(gamma_bl),'deg to the body')
        # print('Measurement range is between:',self.distance_range,'m')
        # print('Scanning field of view is:',np.rad2deg(self.scan_fov),'deg') 
        # print('*******************************************************************************************')
        
    def loc_to_rangeangle(self, p_eb, t_em, sigma_observe = Matrix(2,2)):

        # generate a homogeneous transformation of the robot pose
        H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])

        # get the location of the map feature in the body frame using Inverse(H_eb)=H_be, and then to from the body to the sensor frame using Inverse(H_bl)=H_lb
        t_lm = t2v(Inverse(self.H_bl.H)@Inverse(H_eb.H)@v2t(t_em))

        # Convert the cartesian vector t_lm in the sensor frame to the equivalent polar coordinates
        r,theta = cartesian2polar(t_lm[0],t_lm[1])

        z_lm = Vector(2)
        z_lm[0] = r
        z_lm[1] = theta   
        
        # check that the sensor readings provided are possible
        z_lm = self.check_range(z_lm)      

        #simple range based uncertainty model
        sigma_rtheta = Matrix(2,2) 

        # add a random offset to measurements
        if np.all(sigma_observe == 0.0) == False and (r == 0.0) == False:     
            # measurement uncertainty that is proportional to range
            sigma = (sigma_observe@z_lm)
            #Sample the noise
            r += np.random.normal(0, (sigma[0]), 1)
            theta += np.random.normal(0, (sigma[1]), 1) 

            sigma_rtheta[0,0] = sigma[0]
            sigma_rtheta[1,1] = sigma[1]   

        t_lm[0],t_lm[1] = polar2cartesian(r[0],theta[0])        

        # observation Jacobian
        J = Matrix(2,2)
        dx_dr = np.cos(theta)
        dx_dtheta = -r*np.sin(theta)    
        dy_dr = np.sin(theta)
        dy_dtheta = r*np.cos(theta)    
            
        
        J[0,0] = dx_dr
        J[0,1] = dx_dtheta    
        J[1,0] = dy_dr
        J[1,1] = dy_dtheta        
        
        sigma_xy = J@sigma_rtheta@J.T    
        
        return z_lm, sigma_rtheta, t_lm, sigma_xy

    def rangeangle_to_loc(self, p_eb, z_lm):
        
        # generate a homogeneous transformation of the robot pose
        H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])
        
        # check that the sensor readings provided are possible
        z_lm = self.check_range(z_lm)
        
        # convert the sensor observation in from polar to cartesian coordinates        
        r = z_lm[0]
        theta = z_lm[1]
        
        
        t_lm = polar2cartesian(r,theta)
        
        # Apply the homogeneous transformation to get from the sensor frame to the body with H_bl, and then to the fixed frame H_eb
        t_em = t2v(H_eb.H@self.H_bl.H@v2t(t_lm))
        
        return t_em
    
    def check_range(self,z_lm):
 
        # out of range
        if z_lm[0]>self.distance_range[1] or z_lm[0]<self.distance_range[0]: 
            z_lm[0] = np.nan
            z_lm[1] = np.nan           
        
        # check wrapped angle
        theta = z_lm[1] % (2*np.pi)
        if theta > np.pi: theta -= 2*np.pi
        if theta < -np.pi: theta += 2*np.pi

        # out of scan fov
        if theta > 0.5*self.scan_fov or theta < -0.5*self.scan_fov: 
            z_lm[1] = np.nan
            z_lm[0] = np.nan       
            
        return z_lm     

class TrajectoryGenerate():
    """Class to generate trajectories from a path of x,y, coordinates defined in the earth fixed frame
    This path can be used as input for mobile robot control. 
    
    It key functions are to 
    - Assign timestamps, headings and linear and angular velocities to waypoints by implementing v-a couple trapezoidal trajectories between points
    - Generate turning arcs of a defined radius 
    - Sample the trajectory and associated model commands for any given timestamp
    - Manage waypoint progress considering waypoint acceptance radius and timeouts between consecutive waypoints

    Parameters:
    -----------
    x_path: List of Floats[]
        A sequence of x-axis (Northings) coordinates in the earth fixed frame in metres
    y_path: List of Floats[]
        A sequence of y-axis (Eastings) coordinates in the earth fixed frame in metres

    Attributes:
    -----------
    turning_radius: Float
        Radius in metres of turning arcs
    wp_id: Integer
        Pointer to log the next waypoint the robot trajectory should head towards
    t_complete: Float
        Log the time when the mission was completed
    
    P: Matrix(nx3) 
        Trajectory of poses connected by straightlines, with corresponding timestamps Tp. 
        Each row consists of x,y,gamma defined in the fixed frame
    Tp: Vector(n)    
        Timestamps in seconds, measured as elapsed mission time associated with the poses in the trajectory P
    V: Vector(n)        
        Linear velocities in the body-frame x-direction corresponding to each timestamp Tp
    W: Vector(n)            
        Angular velocities about the body-frame z-direction corresponding to each timestamp Tp    
    D: Vector(n)                
        The relative distance travelled between consecutive poses in P

    P_arc: Matrix(nx3) 
        Trajectory of poses connected by straightlines and turning arcs, with corresponding timestamps Tp. 
        Each row consists of x,y,gamma defined in the fixed frame
    Tp_arc: Vector(n)    
        Timestamps in seconds, measured as elapsed mission time associated with the poses in the trajectory P_arc
    V_arc: Vector(n)        
        Linear velocities in the body-frame x-direction corresponding to each timestamp Tp_arc
    W_arc: Vector(n)            
        Angular velocities about the body-frame z-direction corresponding to each timestamp Tp_arc    
    D_arc: Vector(n)                
        The relative distance travelled between consecutive poses in P_arc
    Arc: Matrix(nx2) 
        A path of turning arc centre points associated with the trajectory P_arc
        
    Functions:
    -----------
    path_to_trajectory(v, a):
        Takes the x_path and y_path parameters and generates trajectories by implementing v-a coupled trapezoidal 
        trajectories between point pairs.
        
        It assumes the first path entry starts from stationary, and that the last path entry ends at stationary. 
        It populates the class attributes:
        
        P, Tp, V, W, D 
        
        and uses the class internal functions _point_to_point() and _stack_trajectory() to do this.        

        Input:
            v:  Float
                Linear forwards velocity in m/s (body frame, b) to reach during coasting phases once accelerated and before decceleration
            a:  Float                
                Linear forwards acceleration in m/s2 (body frame, b) to apply during acceleration and decceleration phases
                
    turning_arcs(radius):
        A function that generates turning arcs of specified radius between staightline sections with different heading angles. 
        It populates the class attibutes:
        
        P_arc, Tp_arc, V_arc, W_arc, D_arc, Arc, turning_radius
        
        and uses geometric operations implemented in math_feeg6043 inward_unit_norm() and line_intersection() to do this.
        
        Input:
            radius:  Float
                Radius in metres of turning arcs
                
    p_u_sample(t):
        A function that samples P_arc, Tp_arc, V_arc, W_arc, D_arc, Arc, turning_radius at time t, where the outputs can 
        be used as inputs for control. 
        It uses geometric operations implemented in math_feeg6043 short_angle() amd interpolate() to do this
        
        Input:
            t: Float
                Timestamp in seconds to sample the trajectory at
        
        Output:
            p: Vector(3)
                Pose is a [3 x 1] matrix of the form
                  [[x]      Northings in m in the earth fixed frame, e
                   [y]      Eastings in m in the earth fixed frame, e
                   [gamma]] Heading in radians about the downwards axis in the earth fixed frame, e
                   
            u: Vector(2)            
                Body twist is a [2 x 1] matrix of the form
                  [[v]      Linear forwards velocity in m/s (body frame, b)
                   [w]]     Angular velocity in rads/s (body frame, b)
    
    wp_progress(t_robot, p_robot, accept_radius, timeout_factor = 2, initial_timeout = 30):
        A function checks whether the trajectory sampled by p_u_sample can progress beyond the next waypoint in P_arc. 
        It implements two conditions to allow this, the first is proximity, which progresses to the next waypoint 
        if the distance between the robot and the waypoint is less than the acceptance radius
        The second is a segment timeout, where waypoint will be progressed if the time takes to reach it is double 
        the expected time as default. This can be changed. The function logs any delay to expected arrival at each waypoint 
        to update future trajectory timestamps
        For the initial waypoint as timeout of 10s is set in the code. The function modifies the class attributes
        
        Tp_arc, wp_id
        
        Input:
            t_robot: Float 
                Timestamp in seconds corresponding to the robot pose
            p_robot: Vector(3)
                Pose of the robot is a [3 x 1] matrix of the form
                  [[x]      Northings in m in the earth fixed frame, e
                   [y]      Eastings in m in the earth fixed frame, e
                   [gamma]] Heading in radians about the downwards axis in the earth fixed frame, e            
            accept_radius: Float
                Acceptance radius in m for the next waypoint to be judged as passed
            timeout_factor: Float
                Default of 2 factor set and applied to the expected time to complete each segment of the trajectory. 
                If exceeded the waypoint will be bypassed and move on to the next
            initial_timeout: Float
                Default of 30, number of seconds to set the first wapyoint timeout to. 
                   
    _point_to_point(self, x_points,y_points, params, start_stationary = True, end_stationary = True):
        An internal function that is used by path_to_trajectory to compute the v-a trapezoidal constraints between a pair of consecutive waypoints
        
    _stack_trajectory(self,P,x_prev,y_prev,gamma,Tp,time,V, velocity,W, angular_velocity, D, distance):
        An internal function that concatenates the outputs of _point_to_point to compute the v-a trapezoidal constraints between a pair of consecutive waypoints

    """


    def __init__(self, x_path, y_path):

        self.x_path = x_path
        self.y_path = y_path        
        self.turning_radius = np.nan
        self.t_complete = np.nan
        
        self.wp_id = 0        
        
        # straight line trajectories
        self.P = np.nan*Matrix(1,3)    
        self.Tp = np.nan*Vector(1)
        self.V = np.nan*Vector(1)        
        self.W = np.nan*Vector(1)                
        self.D = np.nan*Vector(1)
        
        # straight line trajectories
        self.P_arc = np.nan*Matrix(1,3)    
        self.Tp_arc = np.nan*Vector(1)
        self.V_arc = np.nan*Vector(1)        
        self.W_arc = np.nan*Vector(1)                
        self.D_arc = np.nan*Vector(1)                
        self.Arc = np.nan*Matrix(1,2)

    def path_to_trajectory(self, v, a):
    
        for i in range(len(self.x_path)-1):

            # if conditions so only the first waypoint starts as a stationary one, and only the last way point ends stationary
            if i==0: start_stationary = True
            else: start_stationary = False

            if i==len(self.x_path)-2: end_stationary = True
            else: end_stationary = False

            # calculate trajectory between points and store in temporary containors. Remember the python range convention means [i-1:i+1] gives i-1 and i
            P_p2p, Tp_p2p, V_p2p, W_p2p, D_p2p = self._point_to_point(self.x_path[i:i+2],self.y_path[i:i+2], [v, a], start_stationary, end_stationary)

            # stack the trajectories
            if i == 0:
                # For the first trajectory, replace the containors as they are initialised with 0 values            
                self.P = P_p2p
                self.Tp = Tp_p2p
                self.V = V_p2p
                self.W = W_p2p                
                self.D = D_p2p
            else:
                # Note that the first new trajectory pose information should overwrite the last of the previous so that at the intermediate waypoint, the heading is correct
                self.P = np.vstack((self.P[:-1],P_p2p))
                self.V = np.vstack((self.V[:-1],V_p2p))        
                self.W = np.vstack((self.W[:-1],W_p2p))                        

                # Conversly the last distance should be kept and underwrite the first of the new point to point trajectory, since the first entry of this is alway 0
                self.D = np.vstack((self.D,D_p2p[1:]))      

                # Timestamps need to be cumulative. Since point to point timestamps are all start at t=0, we need to add the time so far (last element of T). We follow the convention for poses when we stack time (should not matter either way)
                Tp_p2p = Tp_p2p + self.Tp[-1]
                self.Tp = np.vstack((self.Tp[:-1],Tp_p2p))
                
    def turning_arcs(self,radius):
        # This function takes a trajectory defined by the poses P, corresponding timestamps Tp, and associated velocities V and segment distances D, and replaces corners with turning arcs of defined radius.
        # The function assumes that the provided trajectory has straight line sections between the poses in P
        # It outputs a new trajectory with turning arcs implemented, where segment velocities, distances and waypoint timestamps are updated to be consisent with the new trajectory    

        self.turning_radius = radius
        # store accumulated time to modify waypoints
        t_accumulate = 0
        delta_distance = 0

        # set a threshold angle below which to judge line segment angles to be the same either side of a waypoint
        min_angle = np.deg2rad(1)

        # temporary containor to store waypoint positions
        wp = Vector(3)
        arc_centre = Vector(2)


        for i in range(len(self.Tp)):

            # Calculate the angle between consecutive line segments to decide if a turning arc is needed using arctan2(dy,dx)                
            if i == len(self.Tp)-1 or i==0: delta_angle = 0
            else:
                angle = np.arctan2(self.P[i+1][1] - self.P[i][1], self.P[i+1][0] - self.P[i][0])    
                angle_prev = np.arctan2(self.P[i][1] - self.P[i-1][1], self.P[i][0] - self.P[i-1][0])    

                # handle negative values
                if(angle<0): angle += 2 * np.pi
                if(angle_prev<0): angle_prev += 2 * np.pi           

                delta_angle = abs(angle - angle_prev)

            # store corrected timestamp accounting for cumulative increase due to lengthening/shortening of trajectory
            t = self.Tp[i] + t_accumulate  

            # if the line angle is straight these are used in the new trajectory    
            if delta_angle < min_angle:
                wp[0] = copy.copy(self.P[i,0])
                wp[1] = copy.copy(self.P[i,1])            
                wp[2] = copy.copy(self.P[i,2])                        

                arc_centre[0] = np.nan
                arc_centre[1] = np.nan            

                # angular velocity for straight line sections
                w = 0

                # account for increased distance up to the point
                d = self.D[i]+delta_distance
                # reset delta_distance
                delta_distance = 0


                # initial matrices are 0 values, need to be overwritten
                if i == 0:
                    self.P_arc = copy.copy(wp.T)
                    self.Arc = copy.copy(arc_centre.T)
                    self.Tp_arc = copy.copy(t)
                    self.V_arc = copy.copy(self.V[i])
                    self.D_arc = copy.copy(d)
                    self.W_arc = np.array([w])

                else:
                    self.P_arc = np.vstack((self.P_arc, wp.T))                                
                    self.Arc = np.vstack((self.Arc, arc_centre.T))                
                    self.Tp_arc = np.vstack((self.Tp_arc, t))                                
                    self.V_arc = np.vstack((self.V_arc, self.V[i]))                                
                    self.D_arc = np.vstack((self.D_arc, d))
                    self.W_arc = np.vstack((self.W_arc, np.array([w])))  

            # calculate the coordinates of the rounded corners
            else:
                # reset the distance change
                delta_distance = 0

                #determine direction to offset lines by to get the internal intersect (i.e., centre of the turning arc)            
                x_list = [self.P[i-1][0], self.P[i][0], self.P[i+1][0]]
                y_list = [self.P[i-1][1], self.P[i][1], self.P[i+1][1]]

                # vector0 for incoming line0, and vector1 for outgoing line1
                vector0 = np.array([x_list[1]-x_list[0], y_list[1]-y_list[0]])
                vector1 = np.array([x_list[1]-x_list[2], y_list[1]-y_list[2]])


                # calculate normal vectors in the small angle direction between the lines
                u_norm0, u_norm1 = inward_unit_norm(vector0, vector1)        

                #offset line0 endpoints inwards by the turning radius
                line0_offset_wp0 = np.array([x_list[0], y_list[0]]) + radius * u_norm0
                line0_offset_wp1 = np.array([x_list[1], y_list[1]]) + radius * u_norm0


                #offset line1 endpoints inwards by the turning radius
                line1_offset_wp1 = np.array([x_list[1], y_list[1]]) + radius * u_norm1
                line1_offset_wp2 = np.array([x_list[2], y_list[2]]) + radius * u_norm1

                #define lines between the offset points
                line0_offset = np.array([line0_offset_wp0, line0_offset_wp1])
                line1_offset = np.array([line1_offset_wp1, line1_offset_wp2])

                # calculate intersect of the offset lines as the centre of the turning arc                        
                intersect = line_intersection(line0_offset,line1_offset) 
                arc_centre[0] = intersect[0]
                arc_centre[1] = intersect[1]

                # calculate where the arc's tangent intersects as the projection of the intersect by a radius offset outwards along the normals from earlier
                line0_arc_intersect = arc_centre.T[0] - radius * u_norm0
                line1_arc_intersect = arc_centre.T[0] - radius * u_norm1

                # calculate the distance of the arc from the vector dot product
                cos_theta = np.dot(u_norm0,u_norm1) 

                # Calculate the angle in radians using cos(theta) = 2 vector dot product and determine distance along the arc
    #             angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                arc_angle = np.arccos(np.dot(u_norm0,u_norm1))                
                arc_distance = arc_angle*radius

                # calculate the straight line distances from the arc intersect
                straight_distance0 = np.sqrt((line0_arc_intersect[0]-x_list[1])**2+(line0_arc_intersect[1]-y_list[1])**2)
                straight_distance1 = np.sqrt((line1_arc_intersect[0]-x_list[1])**2+(line1_arc_intersect[1]-y_list[1])**2)    

                max_allowable0 = np.sqrt((x_list[0]-x_list[1])**2+(y_list[0]-y_list[1])**2)
                max_allowable1 = np.sqrt((x_list[2]-x_list[1])**2+(y_list[2]-y_list[1])**2)                

                if straight_distance0 < max_allowable0 and straight_distance1 < max_allowable1:
                    # add first turning arc waypoint
                    wp[0] = line0_arc_intersect[0]
                    wp[1] = line0_arc_intersect[1] 
                    wp[2] = self.P[i-1][2] # on line0 so should be same angle as incoming line
                    self.P_arc = np.vstack((self.P_arc, wp.T))
                    self.V_arc = np.vstack((self.V_arc, self.V[i])) #velocity also needs to be maintained

                    # store arc centre that we need to go around            
                    self.Arc = np.vstack((self.Arc, arc_centre.T))   

                    # need to determine if it is a right turn or a left turn
                    H_eb = HomogeneousTransformation(wp[0:2],wp[2])
                    t_ec = v2t(arc_centre)
                    t_bc = Inverse(H_eb.H)@t_ec
                    if t_bc[1]>0: w = self.V[i]/radius #if centre is +ve y, turn clockwise at w=v/r                
                    else: w = -self.V[i]/radius # if centre is -ve y, turn anticlockwise                  
                    self.W_arc = np.vstack((self.W_arc, np.array([w])))            

                    # timestamp is the corner waypoint minus time to travel the distance from turning arc intersect    
                    t = self.Tp[i] - straight_distance0/self.V[i] + t_accumulate            
                    self.Tp_arc = np.vstack((self.Tp_arc, t))

                    # distance is cornver distance minus distance from turning arc intersect
                    d = self.D[i]-straight_distance0
                    self.D_arc = np.vstack((self.D_arc, d))                                                                   

                    # add second turning arc waypoint
                    wp[0] = line1_arc_intersect[0]
                    wp[1] = line1_arc_intersect[1] 
                    wp[2] = self.P[i][2] # on line1 so should be same angle as outgoing line
                    self.P_arc = np.vstack((self.P_arc, wp.T))
                    self.V_arc = np.vstack((self.V_arc, self.V[i])) #velocity also needs to be maintained

                    # add second turning arc waypoint
                    arc_centre[0] = np.nan
                    arc_centre[1] = np.nan            
                    self.Arc = np.vstack((self.Arc, arc_centre.T))  
                    w = 0
                    self.W_arc = np.vstack((self.W_arc, np.array([w])))            

                    # timestamp added to first arc + time to traverse the arc
                    t += arc_distance/self.V[i]
                    self.Tp_arc = np.vstack((self.Tp_arc, t))

                    # distance is cornver distance minus distance from turning arc intersect
                    d = arc_distance
                    self.D_arc = np.vstack((self.D_arc, d))

                    t_accumulate += (arc_distance-(straight_distance0+straight_distance1))/self.V[i]
                    delta_distance = -straight_distance1
                else:
                    wp[0] = copy.copy(self.P[i,0])
                    wp[1] = copy.copy(self.P[i,1])            
                    wp[2] = copy.copy(self.P[i,2])                        

                    arc_centre[0] = np.nan
                    arc_centre[1] = np.nan            

                    # angular velocity for straight line sections
                    w = 0

                    # account for increased distance up to the point
                    d = self.D[i]+delta_distance
                    # reset delta_distance
                    delta_distance = 0

                    self.P_arc = np.vstack((self.P_arc, wp.T))                                
                    self.Arc = np.vstack((self.Arc, arc_centre.T))                
                    self.Tp_arc = np.vstack((self.Tp_arc, t))                                
                    self.V_arc = np.vstack((self.V_arc, self.V[i]))                                
                    self.D_arc = np.vstack((self.D_arc, d))
                    self.W_arc = np.vstack((self.W_arc, np.array([w])))  
                    
                    print('Turning arcs cannot fit along line section. Increase angle between line section or reduce turning arc radius.')
    
    def p_u_sample(self, t):
        # function that uses interpotation and extrapolation to sample the desired trajectory 
        # defined by poses P, timestamps Tp, velocities V, angular velocities W,
        # turning arc centre's Arc with radius 'radius' at time t

        if len(self.Tp_arc) == 1:
            # means turning_arcs has not run so sample straight line trajectories
            P = self.P
            Tp = self.Tp
            V = self.V
            W = self.W
            D = self.D
            Arc = Matrix(len(self.Tp),2)*np.nan  
            turning_arc_flag = False
        else:
            P = self.P_arc
            Tp = self.Tp_arc
            V = self.V_arc
            W = self.W_arc
            D = self.D_arc
            Arc = self.Arc
            turning_arc_flag = True

        # initialise the pose to return
        p = Vector(3)
        u = Vector(2)        
        init_wp_timeout=30        

        # find the closest timestamp for gamma
        idx = (np.argmin(abs(Tp-t)))

        # determine nearest timestamps to sample
        # if Tp[idx][0] it is smaller than t, then the next timestamp must be after t
        if t>=Tp[idx][0]:        
            id_T0 = idx #id_Tg0 is always the lower id to use
            id_T1 = idx+1 #id_Tg1 is always the upper id to use            

        else:
            # if it is larger, then the previous timestamp must be before t   
            id_T0 = idx-1
            id_T1 = idx


        # check if trajectory can pass the next waypoint
        if id_T1 > self.wp_id:

            #cap pose and control commands to waypoint        
            p[0] = P[self.wp_id][0]
            p[1] = P[self.wp_id][1]
            p[2] = P[self.wp_id][2]

            u[0] = V[self.wp_id][0]
            u[1] = W[self.wp_id][0]              

        # if we are at the end of the data, in which case we sample the last
        else: 
            if t > Tp[-1]:
                #means final waypoint
                p[0] = P[-1][0]
                p[1] = P[-1][1]
                p[2] = P[-1][2]

                u[0] = V[-1][0]
                u[1] = 0 #straightline                    
            # else if we are at the end of the data, in which case we sample the last
            elif t < Tp[0]:
                #means final waypoint
                p[0] = P[0][0]
                p[1] = P[0][1]
                p[2] = P[0][2]

                u[0] = V[0][0]
                u[1] = 0 #straightline                    
            # anything intermediate use extrapolation
            else:        
                # samples times and angles as scalar values expected by interpolate
                t0 = Tp[id_T0][0] 
                t1 = Tp[id_T1][0]         

                # if there is no corresponding turning arc, it is a straight line
                if np.isnan(Arc[id_T0]).any():
                    p[0] = interpolate(t, t0, t1, P[id_T0][0], P[id_T1][0])
                    p[1] = interpolate(t, t0, t1, P[id_T0][1] , P[id_T1][1]) 
                    if turning_arc_flag == False: p[2] = P[id_T0][2] #needs to maintain heading until it arrives at the next waypoint for linear 
                    else: p[2] = P[id_T1][2] #needs to adopt the next headingp[2] = interpolate(t, t0, t1, P[id_T0][2] , P[id_T1][2] , wrap = np.deg2rad(360)) 


                    u[0] = interpolate(t, t0, t1, V[id_T0][0] , V[id_T1][0])
                    u[1] = 0 #straightline

                else:
                    # calculate arc angles respect to arc centre
                    dx = P[id_T0][1]-Arc[id_T0][1]
                    dy = P[id_T0][0]-Arc[id_T0][0]
                    gamma_arc0 = np.arctan2(dx,dy)# measure from dx (north)

                    dx = P[id_T1][1]-Arc[id_T0][1]
                    dy = P[id_T1][0]-Arc[id_T0][0]
                    gamma_arc1 = np.arctan2(dx,dy)# measure from dx (north)            

                    # interpolate the current arc angle
                    arc_angle = interpolate(t, t0, t1, gamma_arc0, gamma_arc1, wrap = np.deg2rad(360))

                    # calculate position for the current arc angle
                    p[0] = Arc[id_T0][0] + self.turning_radius * np.cos(arc_angle)
                    p[1] = Arc[id_T0][1] + self.turning_radius * np.sin(arc_angle)                              
                    p[2] = interpolate(t, t0, t1, P[id_T0][2] , P[id_T1][2], wrap = np.deg2rad(360))            

                    # calculate linear and angular velocity command
                    u[0] = interpolate(t, t0, t1, V[id_T0][0] , V[id_T1][0])
                    u[1] = W[id_T0][0] # angular velocity is constant along the turning arc

        return p, u
    
    def wp_progress(self, t_robot, p_robot, accept_radius, timeout_factor = 2, initial_timeout = 30):

        # Containor for updated timestamps 
        if len(self.Tp_arc) == 1:
            # means turning_arcs has not run so sample straight line trajectories
            P = self.P
            Tp = self.Tp
            V = self.V
            W = self.W
            D = self.D
            Arc = Matrix(len(self.Tp),2)*np.nan        
        else:
            P = self.P_arc
            Tp = self.Tp_arc
            V = self.V_arc
            W = self.W_arc
            D = self.D_arc
            Arc = self.Arc

        #set an initial schedule time to get to the first waypoint
        scheduled_time = 0 #s
        
        wp_progress_flag = False # flag to progress to next waypoint

        # check distance to the next waypoint
        distance_to_wp = np.sqrt((P[self.wp_id,0]-p_robot[0])**2+(P[self.wp_id,1]-p_robot[1])**2)

        # calculate the time spent at on the current waypoint segment
        if self.wp_id ==0: segment_time = t_robot
        else: segment_time = t_robot - Tp[self.wp_id-1]

        # set a segment delay timeout as double the scheduled time
        scheduled_time = Tp[self.wp_id]
        if self.wp_id == 0: 
            timeout = initial_timeout
            latest_allowable = timeout
        else: 
            timeout = timeout_factor*(scheduled_time-Tp[self.wp_id-1])
            latest_allowable = timeout+Tp[self.wp_id-1]

        # compute delay
        if t_robot > scheduled_time: delay = t_robot - scheduled_time
        else: delay = 0  



        if self.wp_id == len(Tp)-1: 
            if np.isnan(self.t_complete): 
                self.t_complete = t_robot
                
                print('************************************************************')
                print('Trajectory completed at:',self.t_complete,'s')    
                print('************************************************************')                
        else:
            # if within acceptance radius, follow trajectory to next waypoint
            if distance_to_wp <= accept_radius:                
           
                print('************************************************************')
                print('Reached waypoint ',self.wp_id,' at t=',t_robot,'s') 
                print('Delay',delay,'s, Scheduled arrival', scheduled_time,'s, Latest allowable arrival', latest_allowable,'s')        
                print('************************************************************')
                wp_progress_flag = True

            elif t_robot > latest_allowable:            
                print('************************************************************')
                print('Failed to reach waypoint ',self.wp_id) 
                print('Delay',delay,'s, Scheduled arrival', scheduled_time,'s')
                print('Time ',t_robot,' s exceeds Latest allowable arrival', latest_allowable,'s')
                print('************************************************************')
                wp_progress_flag = True

            if wp_progress_flag == True:

                if self.wp_id <= len(Tp)-2: 
                    self.wp_id += 1             
                    print('Go to next waypoint ',self.wp_id,' at ',P[self.wp_id])

                    # add any delay to all future waypoints
                    if delay > 0:
                        for i in range(self.wp_id-1,len(Tp)): Tp[i]=Tp[i]+delay
                        if len(self.Tp_arc) == 1: self.Tp = copy.copy(Tp)
                        else: self.Tp_arc = copy.copy(Tp)                 
         
                
    def _point_to_point(self, x_points,y_points, params, start_stationary = True, end_stationary = True):

        # computes the trapezoidal trajectory between two points (x_points, y_points) subject to 
        # velocity and acceleration constraints (params = [v, a])
        # returns the trajectory of poses P with start-relative timestamps Tp, velocities V and distances of each segment

        #initialise
        t=0 #time 
        d=0 #distance

        # Trajectory matrices: pose, timestamps, velocities and distances to store
        P = Matrix(1,3)    
        Tp = Vector(1)
        V = Vector(1)
        W = Vector(1)
        D = Vector(1)
        

        # read parameters
        v = params[0]
        a = params[1]

        # check they are OK
        if v==0: print('Must have v > 0') 
        if a==0: print('Must have a > 0')                

        # determine straight line angle between points
        gamma=np.arctan2(y_points[1]-y_points[0],x_points[1]-x_points[0]) 
        if gamma<0: gamma += 2*np.pi # wraps angles to be between 0 and 2pi 

        # determine straight line distance between points
        total_distance = np.sqrt((y_points[1]-y_points[0])**2+(x_points[1]-x_points[0])**2)

        if start_stationary == True:
            # start of accelerate phase corresponds to first element of Trajectory matrices
            # store starting pose (note velocity, time and distance are initially zero)
            P[0]=np.array([x_points[0],y_points[0],gamma]).T

            # end of accelerate phase corresponds to next element of Trajectory matrices
            t += v/a
            d = 0.5*a*(v/a)**2 

            # store next trajectory row

            P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

            if end_stationary == True: 
                # end of coast phase corresponds to next element of Trajectory matrices                      
                t += (total_distance - 2*(0.5*a*(v/a)**2))/v #total time - time to accelerate and decelerate
                d += total_distance - 2*(0.5*a*(v/a)**2)

                # store next trajectory row            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

                # end of deccelerate phase corresponds to next element of Trajectory matrices                      
                t += v/a
                d += 0.5*a*(v/a)**2

                # store next trajectory row, note velocity needs to be 0            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,0,W,0,D,d)

            else:
                # coast to the end
                dd = total_distance - (0.5*a*(v/a)**2)
                d += dd
                t += dd/v            

                # store next trajectory row            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

        else: #start already coasting

            # store starting pose and velocity (time, distance are initially zero)
            P[0] = np.array([x_points[0],y_points[0],gamma]).T
            V[0] = v  
            W[0] = 0

            if end_stationary == True:
                # end of coast phase corresponds to next element of Trajectory matrices
                dd = total_distance - (0.5*a*(v/a)**2)
                d += dd
                t += dd/v

                # store next trajectory row            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

                # end of deccelerate phase corresponds to next element of Trajectory matrices                                                              
                t += v/a
                d += 0.5*a*(v/a)**2

                # store next trajectory row, note velocity needs to be 0            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,0,W,0,D,d)

            else:
                # coast to end
                d += total_distance
                t += d/v

                # store next trajectory row            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

        return P, Tp, V, W, D
    
    def _stack_trajectory(self,P,x_prev,y_prev,gamma,Tp,time,V, velocity,W, angular_velocity, D, distance):

        # calculate new position after traveling 'distance' along 'gamma'
        x = x_prev + distance*np.cos(gamma)
        y = y_prev +distance*np.sin(gamma)

        # calculate the distance increment
        distance_increase = distance - sum(D[:,0])
        Tp = np.vstack((Tp,time))         
        D = np.vstack((D,distance_increase)) #we store the distance increment, so subtract the previous distance 
        V = np.vstack((V,velocity)) 
        W = np.vstack((W,angular_velocity))         
        P=np.vstack((P,np.array([x,y,gamma]).T))        
        
        return P, Tp, V, W, D     

def kalman_filter_predict(mu, Sigma, u, A, B, R, view_flag=None, x=None, ylim=None):
    """
    Keyword arguments:    
    view_flag -- A boolean to show intermediate plots of the prediction and measurement update
    optional argument -- a range of states to plot probabilities over, using numpy arrays e.g. np.arange(-8,8,0.05)    
    """
    # (1) Project the state forward: x = Ax + Bu
    pred_mu = A @ mu+ B @ u

    # (2) Project the error forward: 
    pred_Sigma = A @ Sigma @ A.T + R
    
    # Return the predicted state and the covariance
    if view_flag is True:
        if x is None: x = np.arange(-8,8,0.05) 
        if ylim is None: ylim=[0,0.55] #default, otherwise use what is provided
    
        plot_kalman(mu, Sigma, pred_mu, pred_Sigma, x, ylim)

    return pred_mu, pred_Sigma

def kalman_filter_update(mu, Sigma, z, C, Q, view_flag=None, x=None, ylim=None):
    """
    Keyword arguments:    
    view_flag -- A boolean to show intermediate plots of the prediction and measurement update
    optional argument -- a range of states to plot probabilities over, using numpy arrays e.g. np.arange(-8,8,0.05)    
    """

    # (3) Compute the Kalman gain
    K = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + Q)        

    # (4) Compute the updated state estimate
    update_mu = mu + K @ (z - C @ mu)

    # (5) Compute the updated state covariance
    update_Sigma = (np.eye(mu.shape[0], dtype=float) - K @ C) @ Sigma
    
    if view_flag is True:
        if x is None: x = np.arange(-8,8,0.05) 
        if ylim is None: ylim=[0,0.55] #default, otherwise use what is provided

        print(C[0,0])
        plot_kalman(mu, Sigma, update_mu, update_Sigma, x, ylim,z, C[0,0], Q)

    # Return the correct state and the covariance        
    return update_mu, update_Sigma

def extended_kalman_filter_predict(mu, Sigma, u, f, R, dt, view_flag=None, x=None, ylim=None):
    """
    Keyword arguments:    
    view_flag -- A boolean to show intermediate plots of the prediction and measurement update
    optional argument -- a range of states to plot probabilities over, using numpy arrays e.g. np.arange(-8,8,0.05)    
    """    
    # (1) Project the state forward
    pred_mu, F = f(mu, u, dt)
      
    # (2) Project the error forward: 
    pred_Sigma = (F @ Sigma @ F.T) + R
    
    # Return the predicted state and the covariance

    if view_flag is True:
        if x is None: x = np.arange(-8,8,0.05) 
        if ylim is None: ylim=[0,0.55] #default, otherwise use what is provided
            
        plot_kalman(mu, Sigma, pred_mu, pred_Sigma, x, ylim)

    # Return the state and the covariance
    return pred_mu, pred_Sigma

# def extended_kalman_filter_update(mu, Sigma, z, h, Q, view_flag=None, x=None, ylim =None):
#     """Keyword arguments:    
#     view_flag -- A boolean to show intermediate plots of the prediction and measurement update
#     optional argument -- a range of states to plot probabilities over, using numpy arrays e.g. np.arange(-8,8,0.05)    
#     """
    
#     # Prepare the estimated measurement
#     pred_z, H = h(mu)
 
#     # (3) Compute the Kalman gain
#     K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Q)
    
#     # (4) Compute the updated state estimate
#     cor_mu = mu + K @ (z - pred_z)

#     # (5) Compute the updated state covariance
#     cor_Sigma = (np.eye(mu.shape[0], dtype=float) - K @ H) @ Sigma
    
#     if view_flag is True:
#         if x is None: x = np.arange(-8,8,0.05) 
#         if ylim is None: ylim=[0,0.55] #default, otherwise use what is provided
            
#         plot_kalman(mu, Sigma, cor_mu,cor_Sigma, x, ylim, z, H[0,0], Q)    

#     # Return the state and the covariance
#     return cor_mu, cor_Sigma

def extended_kalman_filter_update(mu, Sigma, z, h, Q, wrap_index = None):
    
    # Prepare the estimated measurement
    pred_z, H = h(mu)
 
    # (3) Compute the Kalman gain
    K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Q)
    
    # (4) Compute the updated state estimate
    delta_z = z - pred_z        
    if wrap_index != None: delta_z[wrap_index] = (delta_z[wrap_index] + np.pi) % (2 * np.pi) - np.pi    
    cor_mu = mu + K @ (delta_z)

    # (5) Compute the updated state covariance
    cor_Sigma = (np.eye(mu.shape[0], dtype=float) - K @ H) @ Sigma
    
    # Return the state and the covariance
    return cor_mu, cor_Sigma

def feedback_control(ds, ks = None, kn = None, kg = None):
    
    if ks == None: ks = 0.1
    if kn == None: kn = 0.1
    if kg == None: kg = 0.1        
    
    dv = ks*ds[0]                
    dw = (kn*ds[1]+kg*ds[2])
    
    du = Vector(2)
    du[0] = dv
    du[1] = dw
    
    return du



class RangeAngle:
    
    """ Class to plot and filter range angle measurements
    
    Parameters:
    -----------
    angles: Matrix [nx1] Matrix
        Bearing angles along which sensor measurements are made, in radians. Can be any length
    dr: Float
        Factor of range to assign as measuremnt range uncertainty
        Default is 0.1
    dangles: Float
        Angle uncertainty in the lidar meansrements in radians
        Default is np.deg2rad(3)
    r: Matrix [n x 1]
        Ranges along each angle, in m
        Striclty an attribute, but treates as a parameter as it must be defined to compute other attributes        

    Attributes:
    -----------
    x: Matrix [nx1]
        relative x distance in the sensor frame
    y: Matrix [nx1]
        relative y distance in the sensor frame        
    dx: Matrix [nx1]
        relative x distance uncertainty in the sensor frame
    dy: Matrix [nx1]
        relative y distance uncertainty in the sensor frame     
    dr: Matrix [nx1]
        Individual range uncertainties along each beam, in m
    dangles: Matrix [nx1]
        Individual angle uncertainty of each beam, in radians

    label: String
        Class label 
    landmark: [1 x 2 or 3] Float
        Reference coordinate, x, y location for a corner, x, y, radius for a circle
    r_f: Matrix [n x 1]
        Same as r but with a zero mean and all nan values replaced by 0 (i.e., the mean)

        
    Functions:
    -----------
    r(self):
        setter and getter that populates all attributes when self.r is externally set

    plot_polar: 
        Option polar = True (Default)
            Plot observations and any associated information in a polar plot
        Option polar = False
            Plots angles vs range in a standard x-y plot
                   
    plot_cartesian: 
        Plot observations and any associated information in a cartesian plot  

    _fill.nan:
        Populates self.r_f
    """
        
    def __init__(self, angles, dangles=np.deg2rad(3), dr=1/10):
        self.angles = angles
        self._r = np.nan * np.ones([self.angles.size,1])
        self.dr = np.nan * np.ones([self.angles.size,1])
        self.dr_factor = dr

        self.x = np.nan * np.ones([self.angles.size,1])
        self.y = np.nan * np.ones([self.angles.size,1])
        self.dx = np.nan * np.ones([self.angles.size,1])
        self.dy = np.nan * np.ones([self.angles.size,1])        

        self.dangles = dangles * np.ones([self.angles.size,1])

        self.label = np.nan
        self.landmark = [np.nan, np.nan]
        self.r_f = np.nan * np.ones([self.angles.size,1])           
        
    @property
    def r(self):
        return self._r
    
    # auto updates self.dr when self.r is set
    @r.setter
    def r(self, new_r):
        self._r = new_r
        self.dr = new_r * self.dr_factor
        
        self.x, self.y = polar2cartesian(self.r,self.angles)        
        dx_r, dy_r = polar2cartesian(self.dr,self.angles)
        dx_angle, dy_angle = polar2cartesian(self.dangles, self.angles+np.deg2rad(90))        
        self.dx = np.sqrt(dx_r**2+dx_angle**2)
        self.dy = np.sqrt(dy_r**2+dy_angle**2)
        self._fill_nan()                
        
        
    def plot_polar(self, polar = True):
        
        mask = ~np.isnan(self.r)
        
        theta = self.angles[mask]
        r = self.r[mask]
        dtheta = self.dangles[mask].flatten()
        dr = self.dr[mask].flatten()

        fig = plt.figure(figsize=(10, 10))
        if polar == True: ax = fig.add_subplot(projection='polar')
        else: ax = fig.add_subplot()            
       
        ax.errorbar(theta, r, xerr=dtheta, yerr=dr, fmt="o", c="k",label="observations")
        if polar == True: plt.polar(0, 0, marker='o',label='sensor origin')
        else: plt.plot(0, 0, marker='o',label='sensor origin')

        if np.all(~np.isnan(self.landmark)):
            r,theta = cartesian2polar(self.landmark[0],self.landmark[1])
            
            if polar == True: plt.polar(theta,r, marker='o',color="r",label='reference coordinate', zorder=10)#zorder on top                           
            else: plt.plot(theta,r, marker='o',color="r",label='reference coordinate', zorder=10)#zorder on top                

                
            if len(self.landmark) == 3:
                theta = np.linspace(-np.pi, np.pi, 360)
                x = self.landmark[1] + self.landmark[2] * np.cos(theta)
                y = self.landmark[0] + self.landmark[2] * np.sin(theta)
                r,theta = cartesian2polar(y,x)
                plt.polar(theta,r, marker='',color="r",label='circle', zorder=10)#zorder on top                
                
                        
                
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))        
        
        if polar != True: 
            plt.xlabel('Bearing, Radians')
            plt.ylabel('Range, m')

        else: plt.show()


    def plot_cartesian(self): 
        
        mask = ~np.isnan(self.r)
        
        x = (self.x[mask])
        y = self.y[mask]
        dx = self.dx[mask].flatten()
        dy = self.dy[mask].flatten()
        
        plt.errorbar(y, x, xerr= dy, yerr= dx, fmt="o", color="k",label='observations')
            
        plt.scatter(0, 0, marker='o',label='sensor origin')
        plt.xlabel('y-distance, m')
        plt.ylabel('x-distance, m')
        
        if np.all(~np.isnan(self.landmark)):
            plt.scatter(self.landmark[1],self.landmark[0], marker='o',color="r",label='reference coordinate', zorder=10)#zorder on top
            
            if len(self.landmark) == 3:
                circle = patches.Circle((self.landmark[1],self.landmark[0]), radius=self.landmark[2], edgecolor='r', facecolor='none')
                plt.gca().add_patch(circle)  # Add the circle patch to the current axes
                plt.axis('equal')
                    
            
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


    def _fill_nan(self):
        mean=np.nanmean(self.r)  
        for i in range(len(self.angles)):
            if np.isnan(self.r[i]):
                self.r_f[i]=0
            else: 
                self.r_f[i]=self.r[i]-mean

# Graph SLAM
import copy
from plot_feeg6043 import plot_graph
# from model_feeg6043 import graphslam_frontend

import mpmath
mpmath.mp.dps = 32

class graphslam_frontend:
    """
    This class implements the graph SLAM frontend
    
    Functions construct the graph information by:
    - Register elements of the extended state vector as poses or landmarks
    - Register motion constraints between pose-pose pairs based in external model generated covariances 
    - Register observation constraints between pose-landmark pairs based on external model generated covariances
    
    Attributes are:
        pose: [3n x 1] vector of floats
                n time-consecutive poses in the e-frame lined up one after another (x-northings, y-easting, g-heading), each with
                    [x y g]^T
        pose_covariance: [3n x 3] matrix of floats
                n time-consecutive pose covariance matrices lined up one after another, each with 
                    [s_xx s_xy s_xg; s_yx s_yy s_yg; s_gx s_gy s_gg] 
        landmark: [2m x 1] vector of floats
                l unique landmarks coordinates in order of observation lined up one after another, each with
                    [x y]^T
        landmark_covariance: [2m x 2] matrix of floats
                l observation-consecutive landmark covariance matrices lined up one after another, each with
                [s_xx s_xy; s_yx s_yy]  
        edge: ['string', int, int, float] mixed list
                Data association where the 'string' indicates the type of constraint, followed by the nodes connected by the edge and the float is the value of the constraint 
        pose_id: integer pointer to corresponding pose entry
        landmark_id_array: [] integer array logging order of landmark observations
        observation_id: integer pointer to corresponding observation entry
        b: [3n+2m x 1] vector of floats 
                information_vector
        H: 3n+2m x 3n+2m] matrix of floats 
                information matrix
        sigma_anchor: [3 x 3] matrix of floats
                Constraint on the initial pose
        residual: float used during optmisation            
        unique_landmark: [] integer list of unique landmarks
        sigma: Covariance matrix used during Jacobian calculation for determining b and H
        information_vector: [3n+2m x 1] vector of floats #is this needed (BLAIR)
        state_vector: [3n+2m x 1] vector of floats #is this needed (BLAIR)
        information_graph: [3n+2m x 3n+2m] matrix of floats #is this needed (BLAIR)
        n: int number of poses
        m: int number of landmarks
        e: int number of edges

    Functions:

        Front end functions:
        
        anchor(self,sigma): 
            Sets initial pose constraint, takes a 
        observation(self,m,sigma,landmark_id,z):  
            Register a pose (p) to landmark (m) observation (z) as a graph edge constraint with relative covariance
        motion(self, p, sigma, z, final = False):
            Register a consecutive poses (p, p_) as a motion graph edge constraint with relative covariance
        construct_graph(self, visualise_flag = False, update = graph):  
            Populates the information vector b and matrix H based on the observation and motion data associations
        _Jacobian(self,edge_type,x_i,x_j,z_ij = None, visualise_flag = False):  
            Uses the motion and landmark constraints to determine the Jacobian elements and error the 
            returns e,A,B            

    """
    def __init__(self, graph = None):
        # Note that the class implementation ONLY stores the translation vector and
        # rotation angle. The rotation matrix and homogeneous transformation matrix are
        # computed on the fly.
        if graph != None:
            
            self.pose = graph.pose            
            self.pose_covariance = graph.pose_covariance
            self.landmark = graph.landmark
            self.landmark_covariance = graph.landmark_covariance
            self.edge = graph.edge
            self.state_vector = graph.state_vector
            self.pose_id = graph.pose_id
            self.landmark_id_array = graph.landmark_id_array
            self.observation_id = graph.observation_id
            self.b = graph.b
            self.H = graph.H
            self.sigma_anchor = graph.sigma_anchor
            self.residual = graph.residual
            self.unique_landmark = graph.unique_landmark
            self.sigma = graph.sigma
            self.dx = graph.dx
            
            self.n = graph.n #number of poses
            self.m = graph.m #number of unique landmarks
            self.e = graph.e #number of edge
            
        else:
            self.pose = []
            self.pose_covariance = []
            self.landmark = []
            self.landmark_covariance = []
            self.edge = []        
            self.state_vector = None        
            self.pose_id = 0
            self.landmark_id_array = []
            self.observation_id = 0        
            self.b = None
            self.H = None
            self.sigma_anchor = None      
            self.residual = None
            self.unique_landmark = []
            self.sigma = []
            self.dx = None
            
            self.n = 0 #number of poses
            self.m = 0 #number of unique landmarks
            self.e = 0 #number of edge
    
    def anchor(self,sigma):
        self.sigma_anchor = sigma
        
    def observation(self,m,sigma,landmark_id,z):        
        self.landmark.append(copy.copy(m))          # this is need to avoid deep copy behaviour in python
        self.landmark_covariance.append(copy.copy(sigma))
        self.edge.append(['landmark',self.pose_id,self.observation_id,z])
        self.landmark_id_array.append(landmark_id)
        self.observation_id += 1        
        
    def motion(self, p,sigma, z, final = False):        
        if final == False:
            self.edge.append(['motion',self.pose_id,self.pose_id+1,z]) 
            self.pose_id += 1                            
        self.pose.append(copy.copy(p))   
        self.pose_covariance.append(copy.copy(sigma))          
        
    def construct_graph(self,visualise_flag = False):        
        print('Constructing graph with:')   
                    
        unique_landmark = []
        for n in range(len(self.landmark_id_array)):
            if self.landmark_id_array[n] not in unique_landmark:
                unique_landmark.append(self.landmark_id_array[n])
        self.unique_landmark=sorted(unique_landmark)

        self.n = len(self.pose)
        self.m = len(unique_landmark)
        print(self.n+self.m,'nodes')
        print('(',self.n,':poses, ',self.m,':landmark)')                
        self.e = len(self.edge)
        print(self.e,'edge')
       
        self.state_vector = Vector(3*self.n+2*self.m)        
        
        self.b = Vector(3*self.n+2*self.m)        
        self.H = Matrix(3*self.n+2*self.m,3*self.n+2*self.m)
                
        #constrain the initial location
        self.H[0:3,0:3] = Inverse(self.sigma_anchor) 

        # work through the edges to construct b and H
        for k in range(self.e):
                        
            if visualise_flag == True: print('Edge',self.edge[k])
            
            edge_type=self.edge[k][0]
            i=self.edge[k][1]
            j=self.edge[k][2]    
        
            if edge_type == 'motion':
                #point to correct location in the extended state vector
                self.state_vector[3*i:3*i+3]=self.pose[i]
                self.state_vector[3*j:3*j+3]=self.pose[j]
                # associate the constraint
                z_ij =self.edge[k][3]

                # construct the information vector and matrix using the motion Jacobian                
                e_ij,A_ij,self.bij = self._Jacobian(edge_type,self.state_vector[3*i:3*i+3],self.state_vector[3*j:3*j+3],z_ij,visualise_flag)
                
                if visualise_flag == True:
                    print('self.pose[i]',self.pose[i])
                    print('self.pose[j]',self.pose[j])                
                    print('z_ij',z_ij)         
                    print('e_ij',e_ij)
                    print('A_ij',A_ij)         
                    print('self.bij',self.bij)
                    
                sigma_ij = self.pose_covariance[j]-self.pose_covariance[i]#-

                # populate information vector and matrix
                self.b[3*i:3*i+3] += (e_ij.T@Inverse(sigma_ij)@A_ij).T
                self.b[3*j:3*j+3] += (e_ij.T@Inverse(sigma_ij)@self.bij).T              
               
                self.H[3*i:3*i+3,3*i:3*i+3] += A_ij.T@Inverse(sigma_ij)@A_ij
                self.H[3*i:3*i+3,3*j:3*j+3] += A_ij.T@Inverse(sigma_ij)@self.bij                
                self.H[3*j:3*j+3,3*i:3*i+3] += self.bij.T@Inverse(sigma_ij)@A_ij
                self.H[3*j:3*j+3,3*j:3*j+3] += self.bij.T@Inverse(sigma_ij)@self.bij     

            if edge_type == 'landmark':        
                #point to correct location in the extended state vector
                self.state_vector[3*i:3*i+3]=self.pose[i]                                
                l = 3*self.n+2*self.unique_landmark.index(self.landmark_id_array[j])  
                # associate the constraint
                z_il =self.edge[k][3]
                
                # construct the information vector and matrix using the observation Jacobian                      
                if np.all(self.state_vector[l:l+2]) == 0.0: 
                    self.state_vector[l:l+2]=copy.copy(self.landmark[j])
                    e_il,A_il,self.bil = self._Jacobian(edge_type,self.state_vector[3*i:3*i+3],self.state_vector[l:l+2],z_il)
                    
                else:                    
                    # deals with the loop closure by keeping first landmark position but using new landmark observation constraint to isolate inconsistency
                    e_il,A_il,self.bil = self._Jacobian(edge_type,self.state_vector[3*i:3*i+3],self.state_vector[l:l+2],z_il,visualise_flag)

                if visualise_flag == True:
                    print('self.pose[i]  ',self.pose[i]  )
                    print('self.landmark[j]  ',self.landmark[j]  )                
                    print('self.state_vector[l]',self.state_vector[l:l+2])
                    print('z_il',z_il)  
                    print('e_il',e_il)
                    print('A_il',A_il)                
                    print('self.bil',self.bil)
                
                sigma_il = self.landmark_covariance[j][0:2,0:2]
                
                self.b[3*i:3*i+3] += (e_il.T@Inverse(sigma_il)@A_il).T
                self.b[l:l+2] += (e_il.T@Inverse(sigma_il)@self.bil).T                                            

                self.H[3*i:3*i+3,3*i:3*i+3] += A_il.T@Inverse(sigma_il)@A_il
                self.H[3*i:3*i+3,l:l+2] += A_il.T@Inverse(sigma_il)@self.bil
                self.H[l:l+2,3*i:3*i+3] += self.bil.T@Inverse(sigma_il)@A_il                    
                self.H[l:l+2,l:l+2] += self.bil.T@Inverse(sigma_il)@self.bil  
                
            if visualise_flag == True:
                print('Information vector (b)')
                show_information(self.b,self.n,3,self.m,2)                
                print('Information matrix (H)')                        
                show_information(self.H,self.n,3,self.m,2)      
                
    def _Jacobian(self,edge_type,x_i,x_j,z_ij = None, visualise_flag = False):                             
        if edge_type == 'motion':
            
            X_i=HomogeneousTransformation(x_i[0:2],x_i[2])
            X_j=HomogeneousTransformation(x_j[0:2],x_j[2])                                    

            # this is for the first instance only and cannot be itterated over
            Z_ij=HomogeneousTransformation(z_ij[0:2],z_ij[2])            
            
            e_ij = Vector(3) 
            e_ij[0:2] = Z_ij.R.T@(X_i.R.T@(X_j.t-X_i.t)-Z_ij.t)
            e_ij[2] = (X_j.gamma -  X_i.gamma - Z_ij.gamma + np.pi) % (2 * np.pi ) - np.pi        
            
            if visualise_flag == True:
                print('e_ij[0:2]',e_ij[0:2])
                print('(X_i.R.T@(X_j.t-X_i.t)',X_i.R.T@(X_j.t-X_i.t))
                print('Z_ij.t',Z_ij.t)            
                print('Z_ij.R.T',Z_ij.R.T)
                print('e_ij[2]',e_ij[2])  
                print('X_j.gamma,X_i.gamma,Z_ij.gamma',X_j.gamma,X_i.gamma,Z_ij.gamma)
            
            dR_i_dg_i = Matrix(2,2)
            dR_i_dg_i[0,0] = -np.sin(X_i.gamma)
            dR_i_dg_i[1,1] = -np.sin(X_i.gamma)
            dR_i_dg_i[0,1] = -np.cos(X_i.gamma)
            dR_i_dg_i[1,0] = np.cos(X_i.gamma)            

            A_ij = Matrix(3,3)            
            A_ij[0:2,0:2] = -Z_ij.R.T@X_i.R.T
            A_ij[0:2,2:3] = Z_ij.R.T@dR_i_dg_i.T@(X_j.t-X_i.t)
            A_ij[2,2] = -1
            
            self.bij = Matrix(3,3)
            self.bij[0:2,0:2] = Z_ij.R.T@X_i.R.T
            self.bij[2,2] = 1
            
            e=e_ij            
            A=A_ij
            B=self.bij            
            
        if edge_type == 'landmark':            

            X_i=HomogeneousTransformation(x_i[0:2],x_i[2])
            
            # takes first observation at the actual position
            x_l = x_j
            X_l=HomogeneousTransformation(x_l,0)            

            # uses new observation to calculate the observation error
            z_il = z_ij
            Z_il=HomogeneousTransformation(z_il,0)            
             
            # z_il is based on the new observation whereas x_l is based on original position
            # if only 1 observation is made then z_il is consistent with x_i  x_l
            e_il = Vector(3)            
            e_il = X_i.R.T@(X_l.t-X_i.t) - Z_il.t

            if visualise_flag == True:
                print('e_il',e_il)
                print('(X_i.R.T@(X_l.t-X_i.t)',X_i.R.T@(X_l.t-X_i.t))
                print('Z_il.t',Z_il.t)            
            
            dR_i_d_g_i = Matrix(2,2)
            dR_i_d_g_i[0,0] = -np.sin(X_i.gamma)
            dR_i_d_g_i[1,1] = -np.sin(X_i.gamma)
            dR_i_d_g_i[0,1] = -np.cos(X_i.gamma)
            dR_i_d_g_i[1,0] = np.cos(X_i.gamma)            

            A_il = Matrix(2,3)            
            A_il[0:2,0:2] = -X_i.R.T
            A_il[0:2,2:3] = dR_i_d_g_i.T@(X_l.t-X_i.t)
            
            self.bil = Matrix(2,2)
            self.bil = X_i.R.T            
            
            e=e_il
            A=A_il
            B=self.bil            
            
        return e,A,B

class graphslam_backend:
    """
    This class implements the graph SLAM back end
    
    Functions optimise the graph by:
    - Linearising the graph to the information form
    - Reducing information by transferring loop-closures (i.e., pose-landmark-pose) to pose-pose constrains and eliminating landmarks
    - Optimisation of node positions based on contraint equilibrium
    - Reconstructing the map
    
    Parameters are:
        graph: Graph object constructed using graphslam_frontend

    Functions:            
        _update_nodes(self):
            updates location of nodes in the state vector
        _update_covariance(self):    
            updates the covariances 
        reduce2pose(self, visualise_flag = False):
            shifts information from pose-landmark-pose loop closures to pose-pose
            returns equivalent pose_graph with no landmarks
        solve(self, visualise_flag = False):
            optimises that graph using cholesky decomposition                    
    """
    def __init__(self,graph):
        # Note that the class implementation ONLY stores the translation vector and
        # rotation angle. The rotation matrix and homogeneous transformation matrix are
        # computed on the fly.
        
        self.pose = graph.pose
        self.pose_covariance = graph.pose_covariance
        self.landmark = graph.landmark
        self.landmark_covariance = graph.landmark_covariance
        self.edge = graph.edge          
        self.state_vector = graph.state_vector
        self.pose_id = graph.pose_id
        self.landmark_id_array = graph.landmark_id_array
        self.observation_id = graph.observation_id       
        self.b = graph.b
        self.H = graph.H
        self.sigma_anchor = graph.sigma_anchor
        self.residual = graph.residual
        self.unique_landmark = graph.unique_landmark
        self.sigma = graph.sigma
        self.dx = None
        
        
        self.n = graph.n #number of poses
        self.m = graph.m #number of unique landmarks
        self.e = graph.e #number of edge    
   
    def _update_nodes(self):        
        print('Update nodes')
        self.state_vector += self.dx                
        
        for i in range(self.n+self.m):
            if i<self.n:
                self.pose[i] = self.state_vector[3*i:3*i+3]
            elif i<self.n+self.m:        
                l=(i - self.n)

                landmark_id = self.unique_landmark[l]                
                indices = [index for index, value in enumerate(self.landmark_id_array) if value == landmark_id]

                for i in range(len(indices)):
                    self.landmark[indices[i]] = self.state_vector[l*2+3*self.n:l*2+3*self.n+2]                

    def _update_covariance(self):    
        print('Update covariance')

        for k in range(self.e):
            print('Edge',self.edge[k])

            edge_type=self.edge[k][0]
            i=self.edge[k][1]
            j=self.edge[k][2]    

            if edge_type == 'motion':    
                self.pose_covariance[i]=self.sigma[3*i:3*i+3,3*i:3*i+3]
                self.pose_covariance[j]=self.sigma[3*j:3*j+3,3*j:3*j+3]

            if edge_type == 'landmark':
                l=j
                landmark_id = self.landmark_id_array[l]
                indices = [index for index, value in enumerate(self.landmark_id_array) if value == landmark_id]
                for i in range(len(indices)):
                    self.landmark_covariance[indices[i]]=self.sigma[landmark_id*2+3*self.n:landmark_id*2+3*self.n+2,landmark_id*2+3*self.n:landmark_id*2+3*self.n+2]

    def reduce2pose(self, visualise_flag = False):
        print('Reducing information form to poses only')        
        l = (self.n)*3
        
        sigma_m = Inverse(self.H[l:,l:])
        
        H_ = copy.copy(self.H)
        b_ = copy.copy(self.b)        
        
        ij_mod_k =[]        
        for k in self.unique_landmark:                                    
            ij_mod = []

            for n in range(len(self.edge)):
                if self.edge[n][0] == 'landmark' and self.landmark_id_array[self.edge[n][2]]==k: 
                    ij_mod.append(self.edge[n][1])
                    
            ij_mod_k.append(ij_mod)

            i=ij_mod[0]                                                                          
            H_[i*3:i*3+3,i*3:i*3+3] -= H_[i*3:i*3+3,2*k+l:2*k+l+2]@sigma_m[k*2:k*2+2,k*2:k*2+2]@H_[2*k+l:2*k+l+2,i*3:i*3+3]                                        
            
            if len(ij_mod) > 1:
                j=ij_mod[1]                            
                
                H_[i*3:i*3+3,j*3:j*3+3] -= H_[i*3:i*3+3,2*k+l:2*k+l+2]@sigma_m[k*2:k*2+2,k*2:k*2+2]@H_[2*k+l:2*k+l+2,j*3:j*3+3]    
                H_[j*3:j*3+3,i*3:i*3+3] -= H_[j*3:j*3+3,2*k+l:2*k+l+2]@sigma_m[k*2:k*2+2,k*2:k*2+2]@H_[2*k+l:2*k+l+2,i*3:i*3+3]                        
                H_[j*3:j*3+3,j*3:j*3+3] -= H_[j*3:j*3+3,2*k+l:2*k+l+2]@sigma_m[k*2:k*2+2,k*2:k*2+2]@H_[2*k+l:2*k+l+2,j*3:j*3+3]
                            

        for k in range(len(ij_mod_k)):
            i=ij_mod_k[k][0]                                                                          
                
            b_[i*3:i*3+3] -= H_[i*3:i*3+3,2*k+l:2*k+l+2]@sigma_m[k*2:k*2+2,k*2:k*2+2]@b_[2*k+l:2*k+l+2]            
                        
            if len(ij_mod_k[k]) > 1:
                j=ij_mod_k[k][1]                            
                    
                b_[j*3:j*3+3] -= H_[j*3:j*3+3,2*k+l:2*k+l+2]@sigma_m[k*2:k*2+2,k*2:k*2+2]@b_[2*k+l:2*k+l+2]                            
            
        # remove information on landmarks
        pose_graph = copy.copy(self)  
        pose_graph.state_vector=copy.copy(self.state_vector[0:l])
        pose_graph.landmarks=None
        pose_graph.landmark_id_array=[]
        pose_graph.m=0 #number of landmarks
        pose_graph.e=pose_graph.n-1 #edges =nodes - 1
 
        # update information vector and matrix with those above
        pose_graph.H = copy.copy(H_[:l,:l])
        pose_graph.b = copy.copy(b_[:l])   
                
        
        new_list = []
        for i in range(len(pose_graph.edge)):
            if pose_graph.edge[i][0] == 'motion':
                new_list.append(pose_graph.edge[i])
        pose_graph.edge=new_list
        
        if visualise_flag == True:
            print('Original graph has:')
            print('Poses',self.pose)
            print('Landmarks',self.landmark)
            print('Edges',self.edge)

            print('Information vector (b)')
            show_information(b_,self.n,3,self.m,2)

            print('Information matrix (H)')
            show_information(H_,self.n,3,self.m,2)

            print('Reduced graph has:')
            print('Poses',pose_graph.pose)
            print('Landmarks',pose_graph.landmark)
            print('Edges',pose_graph.edge)

            #need to trim landmarks away also!!
            print('Pose only information vector (b_)')
            show_information(pose_graph.b,pose_graph.n,3,pose_graph.m,2, matrix_compare = self.b[0:self.n*3])        

            print('Pose only Information matrix (H_)')                     
            show_information(pose_graph.H,pose_graph.n,3,pose_graph.m,2, matrix_compare = self.H[0:self.n*3,0:self.n*3])
        
        return pose_graph
        
    def solve(self, visualise_flag = False):
        from datetime import datetime
        print('Solve linear system using Cholesky decomposition')

        cpu_start = datetime.now()
        
        # calculate the upper Cholesky triangle

        self.H = self.H + 0.00001*Identity(len(self.H))
        U = cholesky(self.H, lower=False)
        # Solve a linear system using Cholesky decomposition
        v = Inverse(U.T) @ self.b
        self.dx = - Inverse(U) @ v    
        self._update_nodes()  
        
        cpu_end = datetime.now()    
        
        self.sigma = Inverse(U) @ Inverse(U.T)

        self.residual = np.sum(abs(self.dx)/len(self.dx))
        delta =  cpu_end - cpu_start
        
        print('Solver took:',(delta.total_seconds() * 1000),'ms')   

        if visualise_flag == True:
            print('Information vector (b)')
            show_information(self.b,self.n,3,self.m,2)            
            print('Information matrix (H)')
            show_information(self.H,self.n,3,self.m,2)        
            print('Upper Cholesky triangle (U) of Omega.H')
            show_information(U,self.n,3,self.m,2)
            print('Covariance matrix (sigma)')
            show_information(self.sigma,self.n,3,self.m,2)

        print('Residual = ',self.residual) 