import copy
import json
from pathlib import Path
from G2O import GraphSLAM2D, plot_slam2d
from math_feeg6043 import HomogeneousTransformation, Identity, Inverse, Vector, l2m
import numpy as np
import argparse
from datetime import datetime

from uos_feeg6043_build.aruco_udp_driver import ArUcoUDPDriver

from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Pose, Vector3Stamped
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate

from matplotlib import pyplot as plt
from math_feeg6043 import Vector, Matrix, Identity, Transpose, Inverse, v2t, t2v, HomogeneousTransformation, l2m, polar2cartesian, cartesian2polar
from plot_feeg6043 import plot_zero_order, plot_trajectory, plot_2dframe
from model_feeg6043 import TrajectoryGenerate, actuator_configuration_model, graphslam_backend, graphslam_frontend, rigid_body_kinematics, feedback_control, extended_kalman_filter_predict, RangeAngleKinematics
from plot_live import update_mm_plot, update_mm_plot2
import sys
import matplotlib.patches as patches
import time
import csv
import math

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pickle
from math_feeg6043 import polar2cartesian, cartesian2polar
from model_feeg6043 import RangeAngle
from math_feeg6043 import Matrix, Vector,Inverse
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from math_feeg6043 import l2m
import g2o
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def find_sigmaxy(t_lm, sigma_observe):
    # Convert the cartesian vector t_lm in the sensor frame to the equivalent polar coordinates
    r,theta = cartesian2polar(t_lm[0],t_lm[1])
    z_lm = Vector(2)
    z_lm[0] = r
    z_lm[1] = theta       

    #simple range based uncertainty model
    sigma_rtheta = Matrix(2,2) 
    # add a random offset to measurements     
    # measurement uncertainty that is proportional to range
    sigma = (sigma_observe@z_lm)
    #Sample the noise
    r += np.random.normal(0, (sigma[0]), 1)
    theta += np.random.normal(0, (sigma[1]), 1) 

    sigma_rtheta[0,0] = sigma[0]
    sigma_rtheta[1,1] = sigma[1]    

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
    
    return sigma_xy

def calculate_distance(point1, point2):
    x1 = point1[0][0]
    y1 = point1[1][0]
    x2 = point2[0][0]
    y2 = point2[1][0]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_corner(corner):
    # identify the reference coordinate as the point with the largest slope of curvature    
    slope = np.gradient(corner.r[:,0])   
    inflection = np.nanargmax(abs(np.gradient(np.gradient(np.gradient((slope))))))
    corner.landmark = [corner.x[inflection], corner.y[inflection]]

def extended_kalman_filter_update(mu, Sigma, z, h, Q, wrap_index = None):
    
    # Prepare the estimated measurement
    pred_z, H = h(mu)
 
    # (3) Compute the Kalman gain
    K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Q)
    
    # (4) Compute the updated state estimate
    delta_z = z- pred_z        
    if wrap_index != None: delta_z[wrap_index] = (delta_z[wrap_index] + np.pi) % (2 * np.pi) - np.pi    
    cor_mu = mu + K @ (delta_z)

    # (5) Compute the updated state covariance
    cor_Sigma = (np.eye(mu.shape[0], dtype=float) - K @ H) @ Sigma
    
    # Return the state and the covariance
    return cor_mu, cor_Sigma

# Easy names for indexing
N = 0
E = 1
G = 2
DOTX = 3
DOTG = 4

class LaptopPilot:
    def __init__(self, simulation):
        # lower value =  tighter MM circle (further in)
        self.wheel_distance = 0.165  # default: 0.162 
        self.wheel_radius = 0.035 # default: 0.037 but i did a linear velocity test in the simulator and 0.035 results in position matching ground truth
        self.robot_ip = "192.168.90.1"
        if simulation:
            self.robot_ip = "127.0.0.1"
            self.isSimulation = True
        self.rate = 10.0

        print("Connecting to robot with IP", self.robot_ip)

        aruco_params = {
            "port": 50000,  # Port to listen to (DO NOT CHANGE)
            "marker_id": 0,  # Marker ID to listen to (CHANGE THIS to your marker ID)
            # "marker_id": 21,  # Marker ID to listen to (CHANGE THIS to your marker ID)
            # For simulation, it should be zero
        }
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)

        # Create logfile
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = Path("logs/log_" + current_time + ".csv")
        with self.filename.open("w") as f:
            f.write("timestamp,north,east,heading,body_speed,heading_speed,gt_north,gt_east,gt_heading,wl_speed_l,wl_speed_r,linear_vel,angular_vel,sensed_north,sensed_east,sensed_yaw\n")

        self.x = None
        self.y = None
        self.yaw = None

        self.wheel_speed_left = 0
        self.wheel_speed_right = 0
        self.u = Vector(2)
        self.u[0][0] = 0
        self.u[1][0] = 0

        # create an instance of ddrive class
        self.ddrive = actuator_configuration_model(self.wheel_distance, 2 * self.wheel_radius)

        self.states = []
        self.state = Vector(5)
        self.measurement = Vector(5)

        self.covariance = Identity(5) * 0.05**2
        self.covariance[G, G] = np.deg2rad(4)**2 
        self.covariance[DOTX, DOTX] = 0.0**2
        self.covariance[DOTG, DOTG] = np.deg2rad(0)**2

        # Process noise
        self.R = Identity(5) 
        self.R[N, N] = 0.0**2
        self.R[E, E] = 0.0**2
        self.R[G, G] = np.deg2rad(0.0)**2
        self.R[DOTX, DOTX] = 0.03**2
        self.R[DOTG, DOTG] = np.deg2rad(0.1)**2

        # Measurement noise
        self.Q = Identity(5) 
        self.Q[N, N] = 0.01**2
        self.Q[E, E] = 0.01**2
        self.Q[G, G] = np.deg2rad(4)**2

        
        self.first_loop = True
        self.previous_timestamp = None


        # -------------- TUNABLE PARAMETERS -------------------
        # Set Controller Parameters
        # desired time to remove any linear velocity error
        self.tau_s = 1.2  #s to remove along track error

        # allowable travel distance along track to remove heading and normal error
        self.L = 0.4 #m distance to remove normal and angular error
        self.accept_radius = 0.15
        self.v_max = 0.15 #fastest the robot can go
        self.w_max = np.deg2rad(30) #fastest the robot can turn

        # generating trajectories
        self.v = 0.08
        self.a = self.v/5
        self.arc_radius = 0.3

        # self.wp_x = [0, 1.2, 1.2, 0, 0]
        # self.wp_y = [0, 0, 1.2, 1.2, 0]
        self.wp_x = [0, 1.1, 1.1, 0, 0, 1.1, 1.1, 0, 0, 1.1, 1.1, 0, 0]
        self.wp_y = [0, 0, 1.1, 1.1, 0, 0, 1.1, 1.1, 0, 0, 1.1, 1.1, 0]

        # File name
        filename = Path("logs/params_" + current_time + ".csv")

        # List of specific attributes to save
        attributes = ["tau_s", "L", "accept_radius", "v_max", "w_max", "v", "a", "arc_radius", "wp_x", "wp_y"]

        # Write parameters to CSV
        with filename.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            for attr in attributes:
                writer.writerow([attr, getattr(self, attr)])


        self.start_time = None

        # ------------ FOR LIVE PLOTTING ------------
        self.lidar = RangeAngleKinematics(0.1, 0)
        # Motion model pose N
        self.yaw_mm_log = []
        self.N_mm_log = []  
        self.E_mm_log = []  
        self.t_mm_log = []
        
        # Ground truth pose N
        self.N_gt_log = []  
        self.E_gt_log = [] 

        # Sensed pose
        self.N_sensed_log = [] 
        self.E_sensed_log = []

        self.LD_envxcoords = []
        self.LD_envycoords = []

        self.LD_xcoords = []
        self.LD_ycoords = []

        # ---------------- GPC---------------------
        self.gpc_corner = pickle.load(open("gpc_corner_v9_180.dump","rb"))
        plt.ion()
        self.fig, self.ax = plt.subplots()

        # ---------------- SLAM ---------------------
        # Initial uncertainty / covariance
        # self.sigma = Identity(3) * 0.05**2
        self.sigma = Matrix(3,3) 
        self.sigma[0,0]=0.1
        self.sigma[0,1]=0.01
        self.sigma[1,0]=0.01
        self.sigma[1,1]=0.1
        self.sigma[0,2]=0.01
        self.sigma[1,2]=0.01
        self.sigma[2,0]=0.01
        self.sigma[2,1]=0.01
        self.sigma[2,2]=0.1
        # Motion noise added at every timestep
        self.sigma_motion=Matrix(3,2)
        self.sigma_motion[0,0]=0.02*2 # impact of v linear velocity on x           #Task
        self.sigma_motion[0,1]=np.deg2rad(0.01)**2 # impact of w angular velocity on x
        self.sigma_motion[1,0]=0.03**2 # impact of v linear velocity on y
        self.sigma_motion[1,1]=np.deg2rad(0.03)**2 # impact of w angular velocity on y
        self.sigma_motion[2,0]=0.01**2 # impact of v linear velocity on gamma
        self.sigma_motion[2,1]=np.deg2rad(0.03)**2 # impact of w angular velocity on gamma
        # LIDAR observation noise
        self.sigma_observe = Matrix(2,2)
        self.sigma_observe[0,0] = 0.01**2 #10% of range                             #Task
        self.sigma_observe[0,1] = 0                                              #Task
        self.sigma_observe[1,0] = np.deg2rad(0.05) **2 #5 degree per metre range                #Task
        self.sigma_observe[1,1] = 0     
        
        # Observation model
        # locate lidar on robot (keep it simple)
        t_bl = Vector(2)
        t_bl[0] = 0.1  # Blair said it's 10cm from the center
        t_bl[1] = 0
        self.H_bl = HomogeneousTransformation(t_bl[0:2],0)
        self.simple_observation_model = RangeAngleKinematics(t_bl[0], t_bl[1], distance_range = [0.1, 5], scan_fov = np.deg2rad(360))    # Task
        
        # self.graph = graphslam_frontend()
        # self.graph.anchor(self.sigma)
        self.graph = GraphSLAM2D(verbose=True)

        self.landmark_id = -1
        self.isGraphConstructed = False

        # Time since last node update
        self.lastNodeUpdateTime = 0
        # Time since last landmark (observation) update
        self.lastLandmarkUpdateTime = 0

        self.landmark_list = [] # id, north, east


        self.lastPoseState = [[0],[0],[0]]
        # -------------------------------------------

        self.datalog = DataLogger(log_dir="logs")
        # Wheels speeds are in rad/s, convention is that the first wheel is the left
        # and the second one is the right.
        # The message is encoded as a Vector3 with timestamp, so the x component is
        # the left wheel, the y component is the right wheel.
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )
        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",
            Vector3Stamped,
            self.true_wheel_speeds_cb,
            ip=self.robot_ip,
        )
        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip
        )
        self.laserscan_sub = Subscriber(
            "/lidar", LaserScan, self.laserscan_callback, ip=self.robot_ip
        )

    def run(self, time_to_run=-1):
        start_time = datetime.utcnow().timestamp()
        try:
            r = Rate(self.rate)
            while True:
                current_time = datetime.utcnow().timestamp()
                if time_to_run > 0 and current_time - start_time > time_to_run:
                    print("Time is up, stopping…")
                    break
                self.infinite_loop()
                r.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            self.laserscan_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()

    def true_wheel_speeds_cb(self, msg):
        # print("Received L/R true wheel speeds", msg.vector.x, msg.vector.y) #debug
        self.wheel_speed_left = msg.vector.x
        self.wheel_speed_right = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def groundtruth_callback(self, msg):
        """This callback receives the odometry ground truth from the simulator."""
        self.x = msg.position.x
        self.y = msg.position.y
        _, _, self.yaw = msg.orientation.to_euler() + np.deg2rad(90)
        # print("Pose:",self.x,self.y,self.yaw) #debug
        # self.yaw += np.pi #debug
        self.datalog.log(msg, topic_name="/groundtruth")

    def laserscan_callback(self, msg):
        """This is a callback function that is called whenever a message is received

        The message is of type LaserScan and these are the fields:
        - header: Header object that contains (stamp, seq, frame_id)
        - angle_min: float32 - start angle of the scan [rad]
        - angle_max: float32 - end angle of the scan [rad]
        - angle_increment: float32 - angular distance between measurements [rad]
        - time_increment: float32 - time between measurements [seconds]
        - scan_time: float32 - time between scans [seconds]
        - range_min: float32 - minimum range value [m]
        - range_max: float32 - maximum range value [m]
        - ranges: float32[] - range data [m]
        - intensities: float32[] - intensity data ## NOT USED ##
        """
        # print("Received lidar message", msg.header.seq)
        self.laserscan = msg
        self.datalog.log(msg, topic_name="/lidar")
        
        self.LD_angles = msg.angles
        self.LD_ranges = msg.ranges

        # Replace zero range values with NaN
        lidar_ranges = np.array(msg.ranges)
        lidar_angles = np.array(msg.angles)
        lidar_ranges[lidar_ranges == 0] = np.nan
        lidar_ranges.tolist()

        skip_count = 5
        lidar_ranges = lidar_ranges[::skip_count]
        lidar_angles = lidar_angles[::skip_count]


        # filtered_ranges = [r for r, a in zip(msg.ranges, msg.angles) if r != 0]
        # filtered_angles = [a for r, a in zip(msg.ranges, msg.angles) if r != 0]

        # filtered_angles, filtered_ranges = zip(*[(angle, range) for angle, range in zip(msg.angles, lidar_ranges) if np.deg2rad(-40) <= angle <= np.deg2rad(40)])
        filtered_angles, filtered_ranges = zip(*[(angle, range) for angle, range in zip(lidar_angles, lidar_ranges) if np.deg2rad(-90) <= angle <= np.deg2rad(90)])

        # print("FILTERED POINTS = ", len(filtered_angles))


        # print("LENGTH FILTERED = ", len(filtered_angles))

        test = RangeAngle(l2m(filtered_angles))

        test.r = l2m(filtered_ranges)


        threshold = 0.65
        # print(np.max(self.gpc_corner.predict_proba([test.r_f[:,0]])))

        if np.max(self.gpc_corner.predict_proba([test.r_f[:,0]]))>=threshold:
            test.label = (self.gpc_corner.classes_[np.argmax(self.gpc_corner.predict_proba([test.r_f[:,0]]))])
            # print('Landmark is a ',test.label)
        # else: 
        #     print('No landmark')

        if test.label == 'corner': 
            find_corner(test)
            print(np.max(self.gpc_corner.predict_proba([test.r_f[:,0]])))
            print('Coordinate', test.landmark)

            current_timestamp = datetime.utcnow().timestamp()
            # ---------------- SLAM ---------------------
            # Add landmark to graph if a certain time has passed since last observation
            if(current_timestamp - self.lastLandmarkUpdateTime > 5):
                t_lm = test.landmark
                p = Vector(3)
                p[0] = self.state[N]
                p[1] = self.state[E]
                p[2] = self.state[G]

                # Create node
                H_eb = HomogeneousTransformation(p[0:2],p[2])
                # H_eb_ = HomogeneousTransformation(self.lastPoseState[0:2], self.lastPoseState[2])
                # H_bb_ = HomogeneousTransformation()
                # H_bb_.H =  Inverse(H_eb.H)@H_eb_.H   
                # dp = Vector(3)
                # dp[0] = H_bb_.t[0]
                # dp[1] = H_bb_.t[1]
                # dp[2] = (H_bb_.gamma + np.pi) % (2 * np.pi ) - np.pi
                # self.graph.motion(p,self.sigma,dp,final=False)
                self.graph.add_odometry(p[0][0], p[1][0], p[2][0], self.sigma)
                self.lastNodeUpdateTime = current_timestamp
                self.lastPoseState = p

                # makes observation
                t_em = t2v(H_eb.H @ self.H_bl.H @ v2t(t_lm))
                sigma_xy = find_sigmaxy(t_lm, self.sigma_observe)
                # _, _, t_lm, sigma_xy = self.simple_observation_model.loc_to_rangeangle(p, test.landmark, self.sigma_observe) 

                if(len(self.landmark_list) > 0):
                    landmark_id = 0
                    min_id = 0
                    minDist = 9999
                    for i in range(len(self.landmark_list)):
                        dist = calculate_distance(t_em, self.landmark_list[i][1])
                        # print('distance to',self.landmark_list[i][1],':',dist)
                        if(dist < minDist):
                            minDist = dist
                            min_id = self.landmark_list[i][0]
                    if(minDist < np.sqrt(2)):
                        landmark_id = min_id
                    else:
                        landmark_id = None
                else:
                    landmark_id = None

                #adds to the graph as a landmark observation together with its ID
                self.graph.add_landmark(t_em[0][0], t_em[1][0], sigma_xy, pose_id=self.graph.vertex_count-1, landmark_id=landmark_id)
                if(landmark_id is None):
                    # print("Saved landmark",self.graph.vertex_count-1,"to list")
                    self.landmark_list.append([self.graph.vertex_count-1, t_em])
                # self.graph.observation(t_em,sigma_xy,landmark_id,t_lm) # Task
                # print('Observation of Landmark ID', landmark_id,'at',t_em)
                self.lastLandmarkUpdateTime = current_timestamp

            # -------------------------------------------

        self.LD_p_eb = Vector(3)
        self.LD_p_eb[0] = self.state[N][0]
        self.LD_p_eb[1] = self.state[E][0]
        self.LD_p_eb[2] = self.state[G][0]

        # self.LD_p_eb[0] = self.x
        # self.LD_p_eb[1] = self.y
        # self.LD_p_eb[2] = self.yaw

        self.lidar = RangeAngleKinematics(0.1, 0)
        # self.LD_xcoords = []s
        # self.LD_ycoords = []
        # self.LD_envxcoords = []
        # self.LD_envycoords = []

        for angle, dist in zip(self.LD_angles, self.LD_ranges):
            self.LD_x, self.LD_y = polar2cartesian(dist, angle)
            self.LD_xcoords.append(self.LD_x)
            self.LD_ycoords.append(self.LD_y)
            
            self.LD_z_lm = Vector(2)
            self.LD_z_lm[0] = dist
            self.LD_z_lm[1] = angle
            self.LD_t_em = self.lidar.rangeangle_to_loc(self.LD_p_eb, self.LD_z_lm)
            self.LD_envxcoords.append(self.LD_t_em[1][0])
            self.LD_envycoords.append(self.LD_t_em[0][0])


    def create_trajectory(self, wp_x, wp_y, v, a, radius):
        # Convert waypoints to trajectories
        C = l2m([wp_x, wp_y])
        trajectory = TrajectoryGenerate(C[:,0],C[:,1])

        # set velocity and acceleration constraints
        trajectory.path_to_trajectory(v, a)
        trajectory.turning_arcs(radius)

        print('The trajectory has waypoint expected timestamps: s.Tp:',trajectory.Tp)
        print('The trajectory has poses: s.P:',trajectory.P)
        print('The trajectory has velocities: s.V:',trajectory.V)
        print('The trajectory has angular velocities: s.W:',trajectory.W)
        print('The trajectory has segment distances: s.D:',trajectory.D)

        return trajectory

    def trajectory_planning(self, x, y, yaw):
        # Define wheel speed variables
        left_wheel_speed = 0
        right_wheel_speed = 0

        if(np.isnan(self.trajectory.t_complete)):

            ks = 1/self.tau_s
            # Set robot current pose
            p_robot = Vector(3)
            p_robot[0] = x
            p_robot[1] = y
            p_robot[2] = yaw

            # Sample reference pose and speed from trajectory on current time
            current_time = datetime.utcnow().timestamp()
            t = current_time - self.start_time

            accept_radius = 0.1
            timeout_factor = 2
            initial_timeout = 30
            self.trajectory.wp_progress(t, p_robot, accept_radius, timeout_factor, initial_timeout)
            p_ref, u_ref = self.trajectory.p_u_sample(t)

            # Feedback control
            dp = p_ref - p_robot
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi              
            H_eb = HomogeneousTransformation(p_robot[0:2],p_robot[2])
            ds = Inverse(H_eb.H_R) @ dp

            if(self.u[0][0] == 0 and self.u[1][0] == 0):
                kn = 2*u_ref[0]/(self.L**2)
                kg = u_ref[0]/self.L
            else:
                # update control gains for the next timestep
                kn = 2*self.u[0]/(self.L**2)
                kg = u_ref[0]/self.L

            du = feedback_control(ds,ks,kn,kg)
            self.u = u_ref + du

            # tuning wheel distance
            # self.u[0] = 0.0
            # self.u[1] = 0.0

            # impose actuator limits
            if self.u[1] > self.w_max: self.u[1] = self.w_max
            if self.u[1] < -self.w_max: self.u[1] =- self.w_max
            if self.u[0] > self.v_max: self.u[0] = self.v_max
            if self.u[0] < -self.v_max: self.u[0] = -self.v_max


            # inverse kinematics converts twists to wheel commands
            q = self.ddrive.inv_kinematics(self.u)
            left_wheel_speed = q[1][0]
            right_wheel_speed = q[0][0]
        
        # ---------------- SLAM ---------------------
        # Construct graph when trajectory is complete
        else:
            if(self.isGraphConstructed is False):
                p = Vector(3)
                p[0] = self.state[N]
                p[1] = self.state[E]
                p[2] = self.state[G]
                # self.graph.motion(p,Matrix(3,3),Vector(3),final=True)
                print('Finish graph data association')
                print('*************************************************')
                # self.graph.construct_graph(visualise_flag=False)
                self.isGraphConstructed = True

                fig = plot_slam2d(self.graph.optimizer, "Original")
                fig.show()

                self.graph.optimize(10, verbose=True)

                fig = plot_slam2d(self.graph.optimizer, "Optimized")
                fig.show()

                # Save the graph to a file
                self.graph.optimizer.save('graph.g2o')

                # raise Exception("Run complete")

                # Save graph to file
                # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                # filename = Path("logs/graph_" + current_time + ".pkl")
                # with open(filename, 'wb') as f:
                #     pickle.dump(self.graph, f)
        # -------------------------------------------

        # Return wheel speed
        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg.vector.x = left_wheel_speed # Left wheel 1 rev/s = 2*pi rad/s
        wheel_speed_msg.vector.y = right_wheel_speed  # Right wheel 1 rev/s = 2*pi rad/s

        return wheel_speed_msg


    def motion_model(self, state, u, dt):
        N_k_1 = state[N]
        E_k_1 = state[E]    
        G_k_1 = state[G]
        DOTX_k_1 = state[DOTX]
        DOTG_k_1 = state[DOTG]    

        p = Vector(3)
        p[0] = N_k_1
        p[1] = E_k_1
        p[2] = G_k_1
        
        # ---------------- SLAM ---------------------
        # note rigid_body_kinematics already handles the exception dynamics of w=0
        p, self.sigma, _, _  = rigid_body_kinematics(p,u,dt,mu_gt=None,sigma_motion=self.sigma_motion,sigma_xy=self.sigma)
        # -------------------------------------------
        p[2] = p[2] % (2*np.pi) 

        # print("YAW", np.rad2deg(p[2]))
        
        state = np.vstack((p, u))
        
        N_k = state[N]
        E_k = state[E]    
        G_k = state[G]              
        DOTX_k = state[DOTX]
        DOTG_k = state[DOTG]     
        
        # Compute its jacobian
        F = Identity(5)    
        
        if DOTG_k == 0:       
            F[N, G] = -DOTX_k[0] * dt * np.sin(G_k_1[0])
            F[N, DOTX] = dt * np.cos(G_k_1[0])
            F[E, G] = DOTX_k[0] * dt * np.cos(G_k_1[0])
            F[E, DOTX] = dt * np.sin(G_k_1[0])
            F[G, DOTG] = dt        
            
        else:
            F[N, G] = (DOTX_k[0] / DOTG_k[0]) * (np.cos(G_k[0]) - np.cos(G_k_1[0]))
            F[N, DOTX] = (DOTX_k[0] / DOTG_k[0]) * (np.sin(G_k[0]) - np.sin(G_k_1[0]))
            F[N, DOTG] = (DOTX_k[0]/(DOTG_k[0]**2))*(np.sin(G_k_1[0])-np.sin(G_k[0]))+(DOTX_k[0]*dt/DOTG_k[0])*np.cos(G_k[0])
            F[E, G] = (DOTX_k[0]/DOTG_k[0])*(np.sin(G_k[0])-np.sin(G_k_1[0]))
            F[E, DOTX] = (1/DOTG_k[0])*(np.cos(G_k_1[0])-np.cos(G_k[0]))
            F[E, DOTG] = (DOTX_k[0]/(DOTG_k[0]**2))*(np.cos(G_k[0])-np.cos(G_k_1[0]))+(DOTX_k[0]*dt/DOTG_k[0])*np.sin(G_k[0])
            F[G, DOTG] = dt

        return state, F
    
    def initialise_robot_mission(self):
        """
        waits for the first position update before starting the mission
        """

        sensed_pos = self.aruco_driver.read()

        while sensed_pos is None:
            sensed_pos = self.aruco_driver.read()

            time.sleep(0.01)
            print("Waiting for first pose reading")

            
            if sensed_pos is not None:
                self.sensed_pos_stamp_s = sensed_pos[0]
                self.sensed_pos_northings_m = sensed_pos[1]
                self.sensed_pos_eastings_m = sensed_pos[2]

                # in SIMULATION mode
                if "--simulation" in sys.argv:
                    if sensed_pos[6] >= 0:  # Right quadrant
                        self.sensed_pos_yaw_rad = (np.pi/2 - sensed_pos[6]) % (2 * np.pi)
                    else:  # Left quadrant
                        self.sensed_pos_yaw_rad = (-3*np.pi/2 - sensed_pos[6]) % (2 * np.pi)

                # in ARUCO mode
                else:
                    # if ARUCO HEADING is in DEGREES
                    self.sensed_pos_yaw_rad = np.deg2rad(sensed_pos[6])
                    #if ARUCO HEADING is in RADIANS
                    # self.sensed_pos_yaw_rad = sensed_pos[6]

    def convert_wps(self, wp_x, wp_y):
        wp_N = [self.state[N][0] - y for y in wp_y]
        wp_E = [self.state[E][0] + x for x in wp_x]

        return wp_N, wp_E
    

    def h_position_update(self, x):
        """
        This is to update H, which is the Jacobian
        """
        est_measurement = Vector(5)
        est_measurement[N] = x[N]
        est_measurement[E] = x[E]
        est_measurement[G] = x[G]
        H = Matrix(5,5)
        H[N, N] = 1
        H[E, E] = 1
        H[G, G] = 1
        return est_measurement, H # est measurement is z here, which is the entire matrix


    def infinite_loop(self):

        if self.first_loop:
            self.initialise_robot_mission()
            self.state[N][0] = self.sensed_pos_northings_m
            self.state[E][0] = self.sensed_pos_eastings_m
            self.state[G][0] = self.sensed_pos_yaw_rad 
           
            self.state[DOTX][0] = 0.0
            self.state[DOTG][0] = 0.0

            self.waypoint_N, self.waypoint_E = self.convert_wps(self.wp_x, self.wp_y)
            self.trajectory = self.create_trajectory(self.waypoint_N, self.waypoint_E, self.v, self.a, self.arc_radius)
            self.all_N_wps = [row[0] for row in self.trajectory.P_arc]
            self.all_E_wps = [row[1] for row in self.trajectory.P_arc]

            self.lastPoseState = [self.state[N], self.state[E], self.state[G]]
            self.graph.add_fixed_pose(g2o.SE2(self.state[N][0], self.state[E][0], self.state[G][0]))
            dt = 0


        current_timestamp = datetime.utcnow().timestamp()
        if(self.start_time is None):
            self.start_time = current_timestamp

        # if self.first_loop == False:
        #     dt = current_timestamp - self.t_mm_log[-1]
        dt = 1/self.rate

        # print("dt = ", dt)

        # Run motion model
        q = Vector(2)
        q[1][0] = self.wheel_speed_left
        q[0][0] = self.wheel_speed_right

        # print("L/R Wheel Speeds, ", self.wheel_speed_left, self.wheel_speed_right)

        u = self.ddrive.fwd_kinematics(q)

        self.state, self.covariance = extended_kalman_filter_predict(self.state, self.covariance, u, self.motion_model, self.R, dt)

        # ---------------- SLAM ---------------------
        # Add current pose as node in graph after a certain amount of time since last update
        if(current_timestamp - self.lastNodeUpdateTime >= 10):
            p = Vector(3)
            p[0] = self.state[N]
            p[1] = self.state[E]
            p[2] = self.state[G]

            # H_eb = HomogeneousTransformation(p[0:2],p[2])
            # H_eb_ = HomogeneousTransformation(self.lastPoseState[0:2], self.lastPoseState[2])
            # H_bb_ = HomogeneousTransformation()
            # H_bb_.H =  Inverse(H_eb.H)@H_eb_.H   
            # dp = Vector(3)
            # dp[0] = H_bb_.t[0]
            # dp[1] = H_bb_.t[1]
            # dp[2] = (H_bb_.gamma + np.pi) % (2 * np.pi ) - np.pi

            self.graph.add_odometry(p[0][0], p[1][0], p[2][0], self.sigma)
            # self.graph.motion(p,self.sigma,dp,final=False)
            self.lastNodeUpdateTime = current_timestamp
            self.lastPoseState = p
        # -------------------------------------------

        self.sensed_pos_northings_m = 0
        self.sensed_pos_eastings_m = 0
        self.sensed_pos_yaw_rad = 0

        sensed_pos = self.aruco_driver.read()


        if sensed_pos is not None:

            self.sensed_pos_stamp_s = sensed_pos[0]
            self.sensed_pos_northings_m = sensed_pos[1]
            self.sensed_pos_eastings_m = sensed_pos[2] 

            # in SIMULATION mode
            if "--simulation" in sys.argv:
                # Map angles from webots coordinate system to global coordinate system
                if sensed_pos[6] >= 0:  # Right quadrant
                    self.sensed_pos_yaw_rad = (np.pi/2 - sensed_pos[6]) % (2 * np.pi)
                else:  # Left quadrant
                    self.sensed_pos_yaw_rad = (-3*np.pi/2 - sensed_pos[6]) % (2 * np.pi)
            # in ARUCO mode
            else:
                # if ARUCO HEADING is in DEGREES
                self.sensed_pos_yaw_rad = np.deg2rad(sensed_pos[6])
                #if ARUCO HEADING is in RADIANS
                # self.sensed_pos_yaw_rad = sensed_pos[6]

            # only use the first new pose update
            if(self.sensed_pos_stamp_s != self.previous_timestamp or self.first_loop):

                self.measurement[N][0] = self.sensed_pos_northings_m
                self.measurement[E][0] = self.sensed_pos_eastings_m
                self.measurement[G][0] = self.sensed_pos_yaw_rad
                # print("----------------- POSITION UPDATE ----------------------")
                # print("RAW YAW = ", np.rad2deg(sensed_pos[6]), " deg")
                # print("FIXED YAW = ", np.rad2deg(self.sensed_pos_yaw_rad))

                self.previous_timestamp = self.sensed_pos_stamp_s
                # self.state, self.covariance = extended_kalman_filter_update(self.state, self.covariance, self.measurement, self.h_position_update, self.Q, wrap_index=G)

        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg = self.trajectory_planning(self.state[N], self.state[E], self.state[G])

        self.N_mm_log.append(self.state[N])
        self.E_mm_log.append(self.state[E])
        self.yaw_mm_log.append(self.state[G])
        self.t_mm_log.append(current_timestamp)

        self.N_gt_log.append(self.x)
        self.E_gt_log.append(self.y)

        self.N_sensed_log.append(self.measurement[N][0])
        self.E_sensed_log.append(self.measurement[E][0])


        update_mm_plot2(self.ax, self.fig, self.LD_envxcoords, self.LD_envycoords, self.N_mm_log, self.E_mm_log, self.N_gt_log, self.E_gt_log)

        # self.ax.plot(self.trajectory.P[:,1],self.trajectory.P[:,0],'o--k')
        # self.ax.plot(self.trajectory.P_arc[:,1],self.trajectory.P_arc[:,0],'ok')
        # self.ax.plot(self.trajectory.P[0,1],self.trajectory.P[0,0],'oc',label = 'Start')
        # self.ax.plot(self.trajectory.P[-1,1],self.trajectory.P[-1,0],'or',label = 'End')
        # self.ax.legend()

        self.ax.plot(self.E_sensed_log, self.N_sensed_log, "xr", label="Sensed")
        plt.pause(0.001)


        # Log Data to file
        with self.filename.open("a") as f:

            f.write(f"{current_timestamp}, "
                    f"{self.state[N][0]}, "
                    f"{self.state[E][0]}, "
                    f"{self.state[G][0]}, "
                    f"{self.state[DOTX][0]}, "
                    f"{self.state[DOTG][0]}, "
                    f"{self.x}, "
                    f"{self.y}, "
                    f"{self.yaw}, "
                    f"{q[1][0]}, "
                    f"{q[0][0]}, "
                    f"{self.u[1][0]}, "
                    f"{self.u[0][0]}, "
                    f"{self.sensed_pos_northings_m}, "
                    f"{self.sensed_pos_eastings_m}, "
                    f"{self.sensed_pos_yaw_rad}\n")
            
                
        # > Act < #
        # Send commands to the robot
        self.wheel_speed_pub.publish(wheel_speed_msg)
        self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")
        self.first_loop = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    args = parser.parse_args()

    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)
