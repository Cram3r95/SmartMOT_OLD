#!/usr/bin/env python2 
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:38:44 2020

@author: Carlos Gómez-Huélamo

SmartMOT: Code to track the detections given by a sensor fusion algorithm (converted into Bird's Eye View (Image frame, z-axis inwards 
 with the origin located at the top-left corner) using the SORT (Simple Online and Real-Time Tracking) algorithm as backbone

Communications are based on ROS (Robot Operating Sytem)

Inputs: 3D Object Detections topic
Outputs: Tracked obstacles topic and monitors information (collision prediction, )

Note that each obstacle shows an unique ID in addition to its semantic information (person, car, ...), 
in order to make easier the decision making processes.

Executed via Python2.7 (python2 map-filtered-mot.py --arguments ...)
"""

from __future__ import print_function

# General-use imports

import os
import sys
import time
import cv2
import numpy as np
import math
import matplotlib
matplotlib.use('Agg') # In order to avoid: RuntimeError: main thread is not in main loop exception

from argparse import ArgumentParser

# Sklearn imports

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Custom functions imports

from aux_functions import geometric_functions
from aux_functions import monitors_functions
from aux_functions import sort_functions
from aux_functions import tracking_functions

# ROS imports

sys.path.insert(0,'/opt/ros/melodic/lib/python2.7/dist-packages')
import rospy
import visualization_msgs.msg
import nav_msgs.msg
import std_msgs.msg
import tf 

from t4ac_msgs.msg import BEV_detection, BEV_detections_list, MonitorizedLanes, Node
from message_filters import TimeSynchronizer, Subscriber

# Auxiliar variables

detection_threshold = 0.3
max_age = 3
min_hits = 1 # 1
kitti = 0
n = 4 # Predict x bounding boxes ahead
seconds_ahead = 2 # Predict all trajectories at least x seconds ahead
    
filename = ''
path = os.path.curdir + '/results' + filename
header_synchro = 5

class Map_Filtered_MOT:
    def __init__(self,args):
        # Auxiliar variables
       
        self.init_scene = False
        self.use_gaussian_noise = True
        self.filter_hdmap = True
        self.colours = np.random.rand(32,3)

        # Display config

        self.image_width = 0 # This value will be updated
        self.image_height = 1000 # Pixels 
        self.shapes = []
        self.view_ahead = 0 

        # Ego-vehicle and Prediction variables

        self.n = n 
        self.ego_vehicle_x, self.ego_vehicle_y = 0.0,0.0
        self.ego_orientation_cumulative_diff = 0 # Cumulative orientation w.r.t the original 
                                                 # orientation of the ego-vehicle (radians)
        self.initial_angle = False
        self.previous_angle = float(0)
        self.previous_yaw = float(0)
        self.current_yaw = float(0) 
        self.ego_braking_distance = 0
        self.ego_dimensions = np.array([4.4,  # Length
                                        1.8]) # Width
        self.ego_trajectory_forecasted_marker_list = visualization_msgs.msg.MarkerArray()
        self.seconds_ahead = seconds_ahead

        # MOT-Prediction Callback 

        self.start = float(0)
        self.end = float(0)
        self.frame_no = 0
        self.avg_fps = float(0)
        self.write_video = False
        self.video_flag = False 

        # Emergency break

        self.cont = 0
        self.collision_flag = std_msgs.msg.Bool()
        self.collision_flag.data = False
        self.emergency_break_timer = float(0)
        self.nearest_object_in_route = 50000
        self.geometric_monitorized_area = []

        # Arguments from ROS params

        self.display = args[0]
        self.trajectory_prediction = args[1]
        self.ros = args[2]
        self.grid = args[3]
        self.rc_max = args[4]
        
        # ROS publishers

        self.pub_monitorized_area = rospy.Publisher("/t4ac/perception/detection/monitorized_area_marker", visualization_msgs.msg.Marker, queue_size = 20)
        self.pub_bev_sort_tracking_markers_list = rospy.Publisher('/t4ac/perception/tracking/obstacles_markers', visualization_msgs.msg.MarkerArray, queue_size = 20)
        self.pub_particular_monitorized_area_markers_list = rospy.Publisher('/t4ac/perception/monitors/individual_monitorized_area', visualization_msgs.msg.MarkerArray, queue_size = 20)
        self.pub_ego_vehicle_forecasted_trajectory_markers_list = rospy.Publisher('/t4ac/perception/prediction/ego_vehicle_forecasted_trajectory', visualization_msgs.msg.MarkerArray, queue_size = 20)
        self.pub_collision = rospy.Publisher('/t4ac/perception/monitors/predicted_collision', std_msgs.msg.Bool, queue_size = 20)
        self.pub_nearest_object_distance = rospy.Publisher('/t4ac/perception/monitors/nearest_object_distance', std_msgs.msg.Float64, queue_size = 20)

        # ROS subscribers

        if not self.filter_hdmap:
            self.sub_road_curvature = rospy.Subscriber("/control/rc", std_msgs.msg.Float64, self.road_curvature_callback)
        self.detections_topic = "/t4ac/perception/detection/merged_obstacles"
        self.odom_topic = "/t4ac/localization/pose"
        self.monitorized_lanes_topic = "/t4ac/mapping/monitor/lanes"

        self.detections_subscriber = Subscriber(self.detections_topic, BEV_detections_list)
        self.odom_subscriber = Subscriber(self.odom_topic, nav_msgs.msg.Odometry)
        self.monitorized_lanes_subscriber = Subscriber(self.monitorized_lanes_topic, MonitorizedLanes)

        ts = TimeSynchronizer([self.detections_subscriber, 
                               self.odom_subscriber, 
                               self.monitorized_lanes_subscriber], 
                               header_synchro)
        ts.registerCallback(self.mot_prediction_callback)
        
        # Listeners
        
        self.listener = tf.TransformListener()

    def road_curvature_callback(self, msg):
        """
        """

        rc = msg.data
        rc_ratio = rc/self.rc_max

        self.geometric_monitorized_area
        
        xmax = float(rc_ratio*self.ego_braking_distance*1.4) # We consider a 40 % safety factor 

        if xmax > 30:
            xmax = 30
        elif xmax < 12 and rc_ratio > 0.8: # rc_ratio < 0.8, we are in curve, so x_max must be reduced to 0
            xmax = 12
        
        lateral = 2.6
        xmin = 0
        ymin = rc_ratio * (-lateral)
        ymax = rc_ratio * lateral

        self.geometric_monitorized_area = [xmax,xmin,ymax,ymin]

        geometric_monitorized_area_marker = visualization_msgs.msg.Marker()

        geometric_monitorized_area_marker.header.frame_id = "/ego_vehicle/lidar/lidar1"
        geometric_monitorized_area_marker.ns = "geometric_monitorized_area"
        geometric_monitorized_area_marker.action = geometric_monitorized_area_marker.ADD
        geometric_monitorized_area_marker.lifetime = rospy.Duration.from_sec(1)
        geometric_monitorized_area_marker.type = geometric_monitorized_area_marker.CUBE

        geometric_monitorized_area_marker.color.r = 1.0
        geometric_monitorized_area_marker.color.g = 1.0
        geometric_monitorized_area_marker.color.b = 1.0
        geometric_monitorized_area_marker.color.a = 0.4

        geometric_monitorized_area_marker.pose.position.x = xmax/2   
        geometric_monitorized_area_marker.pose.position.y = (ymin+ymax)/2
        geometric_monitorized_area_marker.pose.position.z = -2.0
        
        geometric_monitorized_area_marker.scale.x = xmax
        geometric_monitorized_area_marker.scale.y = abs(ymin) + ymax
        geometric_monitorized_area_marker.scale.z = 0.2

        self.pub_monitorized_area.publish(geometric_monitorized_area_marker)
        
    def mot_prediction_callback(self, detections_rosmsg, odom_rosmsg, monitorized_lanes_rosmsg):
        """
        """

        try:                                                         # Target # Pose
            (translation,quaternion) = self.listener.lookupTransform('/map', '/ego_vehicle/lidar/lidar1', rospy.Time(0)) 
            # rospy.Time(0) get us the latest available transform
            rot_matrix = tf.transformations.quaternion_matrix(quaternion)
            
            self.tf_map2lidar = rot_matrix
            self.tf_map2lidar[:3,3] = self.tf_map2lidar[:3,3] + translation # This matrix transforms local to global coordinates

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("\033[1;33m"+"TF exception"+'\033[0;m')

        self.start = time.time()

        self.detections_subscriber.unregister()
        self.odom_subscriber.unregister()
        self.monitorized_lanes_subscriber.unregister()

        # Initialize the scene

        if not self.init_scene:
            self.real_height = detections_rosmsg.front - detections_rosmsg.back # Grid height (m)
            self.real_front = detections_rosmsg.front + self.view_ahead # To study obstacles (view_head) meters ahead
            self.real_width = -detections_rosmsg.left + detections_rosmsg.right # Grid width (m)
            self.real_left = -detections_rosmsg.left

            r = float(self.image_height)/self.real_height

            self.image_width = int(round(self.real_width*r)) # Grid width (pixels)
            self.image_front = int(round(self.real_front*r))
            self.image_left = int(round(self.real_left*r))

            print("Real width: ", self.real_width)
            print("Real height: ", self.real_height)
            print("Image width: ", self.image_width)
            print("Image height: ", self.image_height)

            self.shapes = (self.real_front,self.real_left,self.image_front,self.image_left)
            self.scale_factor = (self.image_height/self.real_height,self.image_width/self.real_width)

            # Our centroid is defined in BEV camera coordinates but centered in the ego-vehicle. 
            # We need to traslate it to the top-left corner (required by the SORT algorithm)

            self.tf_bevtl2bevcenter_m = np.array([[1.0, 0.0, 0.0, self.shapes[1]],
                                                  [0.0, 1.0, 0.0, self.shapes[0]],
                                                  [0.0, 0.0, 1.0, 0.0], 
                                                  [0.0, 0.0, 0.0, 1.0]])

            # From BEV centered to BEV top-left (in pixels)

            self.tf_bevtl2bevcenter_px = np.array([[1.0, 0.0, 0.0, -self.shapes[3]], 
                                                   [0.0, 1.0, 0.0, -self.shapes[2]],
                                                   [0.0, 0.0, 1.0, 0.0], 
                                                   [0.0, 0.0, 0.0, 1.0]])

            # Rotation matrix from LiDAR frame to BEV frame (to convert to global coordinates)
    
            self.tf_lidar2bev = np.array([[0.0,-1.0,0.0,0.0],
                                          [-1.0,0.0,0.0,0.0],
                                          [0.0,0.0,-1.0,0.0],
                                          [0.0,0.0,0.0, 1.0]])

            # Initialize the regression model to calculate the braking distance

            self.velocity_braking_distance_model = monitors_functions.fit_velocity_braking_distance_model()

            # Tracker
            
            self.mot_tracker = sort_functions.Sort(max_age,min_hits,n,self.shapes,self.trajectory_prediction,self.filter_hdmap)

            self.init_scene = True

        output_image = np.ones((self.image_height,self.image_width,3),dtype = "uint8") # Black image to represent the scene

        print("----------------")

        # Initialize ROS and monitors variables

        self.trackers_marker_list = visualization_msgs.msg.MarkerArray() # Initialize trackers list
        monitorized_area_colours = []

        trackers_in_route = 0
        nearest_distance = std_msgs.msg.Float64()
        nearest_distance.data = float(self.nearest_object_in_route)

        world_features = []
        trackers = []
        dynamic_trackers = []
        static_trackers = []
        ndt = 0 # Number of dynamic trackers
        
        timer_rosmsg = detections_rosmsg.header.stamp.to_sec()

        # Predict the ego-vehicle trajectory

        monitors_functions.ego_vehicle_prediction_real(self,odom_rosmsg)
        # monitors_functions.ego_vehicle_prediction(self,odom_rosmsg,output_image)
        print("Braking distance ego vehicle: ", float(self.ego_braking_distance))

        # Convert input data to bboxes to perform Multi-Object Tracking 
        """
        #print("\nNumer of total detections: ", len(detections_rosmsg.bev_detections_list))
        bboxes_features,types = sort_functions.bbox_to_xywh_cls_conf(self,detections_rosmsg,odom_rosmsg,detection_threshold,output_image)
        #print("Number of relevant detections: ", len(bboxes_features)) # score > detection_threshold

        ## Multi-Object Tracking

        # TODO: Publish on tracked_obstacle message instead of visualization marker
        # TODO: Evaluate the predicted position to predict its influence in a certain use case

        if (len(bboxes_features) > 0): # At least one object was detected
            trackers,object_types,object_scores,object_observation_angles,dynamic_trackers,static_trackers = self.mot_tracker.update(bboxes_features,types,
                                                                                                                                     self.ego_vel_px,
                                                                                                                                     self.tf_map2lidar,
                                                                                                                                     self.shapes,
                                                                                                                                     self.scale_factor,
                                                                                                                                     monitorized_lanes_rosmsg,
                                                                                                                                     timer_rosmsg,
                                                                                                                                     self.angle_bb,
                                                                                                                                     self.geometric_monitorized_area)




            
            
            #print("Number of trackers: ", len(trackers))
            if len(dynamic_trackers.shape) == 3:
                #print("Dynamic trackers", dynamic_trackers.shape[1])
                ndt = dynamic_trackers.shape[1]
            else:
                #print("Dynamic trackers: ", 0)
                ndt = 0
            #print("Static trackers: ", static_trackers.shape[0])

            id_nearest = -1

            if (len(trackers) > 0): # At least one object was tracked
                for i,tracker in enumerate(trackers): 
                    object_type  = object_types[i]
                    object_score = object_scores[i]
                    object_rotation = np.float64(object_observation_angles[i,0])
                    object_observation_angle = np.float64(object_observation_angles[i,1])

                    color = self.colours[tracker[5].astype(int)%32]
                    #print("hago append")
                    monitorized_area_colours.append(color)

                    if self.ros:
                        world_features = monitors_functions.tracker_to_topic(self,tracker,object_type,color) # world_features (w,l,h,x,y,z,id)
                        #print("WF: ", world_features)
                        if kitti:
                            num_image = detections_rosmsg.header.seq-1 # Number of image in the dataset, e.g. 0000.txt -> 0
                            object_properties = object_observation_angle,object_rotation,object_score
                            monitors_functions.store_kitti(num_image,path,object_type,world_features,object_properties)

                    if (self.display):
                        my_thickness = -1
                        geometric_functions.compute_and_draw(tracker,color,my_thickness,output_image)

                    label = 'ID %06d'%tracker[5].astype(int)
                    cv2.putText(output_image,label,(tracker[0].astype(int),tracker[1].astype(int)-20), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)
                    cv2.putText(output_image,object_type,(tracker[0].astype(int),tracker[1].astype(int)-40), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)

                    # Evaluate if there is some obstacle in lane and calculate nearest distance
     
                    if self.filter_hdmap:
                        if tracker[-1]: # In route, last element of the array 
                            trackers_in_route += 1
                            obstacle_local_position = np.zeros((1,9))

                            obstacle_local_position[0,7] = world_features[3]
                            obstacle_local_position[0,8] = world_features[4]

                            obstacle_global_position = sort_functions.store_global_coordinates(self.tf_map2lidar,obstacle_local_position)
    
                            #distance_to_object = monitors_functions.calculate_distance_to_nearest_object_inside_route(monitorized_lanes_rosmsg,obstacle_global_position)
                            
                            
                            detection = Node()

                            detection.x = obstacle_global_position[0,0]
                            detection.y = -obstacle_global_position[1,0]
                            
                            ego_x_global = odom_rosmsg.pose.pose.position.x
                            ego_y_global = -odom_rosmsg.pose.pose.position.y

                            distance_to_object = math.sqrt(pow(ego_x_global-detection.x,2)+pow(ego_y_global-detection.y,2))
                            distance_to_object -= 5 # QUITARLO, DEBERIA SER DISTANCIA CENTROIDE OBJETO A MORRO, EN VEZ DE LIDAR A LIDAR, POR ESO
                            # LE METO ESTE OFFSET
                            
                            
                            
                            
                            #print("Distance to object: ", distance_to_object)
                            if distance_to_object < self.nearest_object_in_route:
                                id_nearest = tracker[5]
                                self.nearest_object_in_route = distance_to_object
                    else:
                        # Evaluate in the geometric monitorized area

                        x = world_features[3]
                        y = world_features[4]


                        print("main")
                        print("goemetric area: ", self.geometric_monitorized_area)
                        print("x y: ", x,y)

                        if (x < self.geometric_monitorized_area[0] and x > self.geometric_monitorized_area[1]
                            and y < self.geometric_monitorized_area[2] and y > self.geometric_monitorized_area[3]):
                            trackers_in_route += 1
                            self.cont = 0
                            print("\n\n\nDentro")
                            distance_to_object = math.sqrt(pow(x,2)+pow(y,2))
                            print("Nearest: ", self.nearest_object_in_route)
                            print("distance: ", distance_to_object)
                            if distance_to_object < self.nearest_object_in_route:
                                self.nearest_object_in_route = distance_to_object

                print("Collision: ", self.collision_flag.data)
                print("trackers in route: ", trackers_in_route)
                print("Distance nearest: ", self.nearest_object_in_route)
                if self.collision_flag.data and (trackers_in_route == 0 or (self.nearest_object_in_route > 12 and self.abs_vel < 1)):
                    print("suma A")
                    self.cont += 1
                else:
                    self.cont == 0

                nearest_distance.data = self.nearest_object_in_route
 
                if(self.trajectory_prediction):

                    collision_id_list = [[],[]]
                    
                    # Evaluate collision with dynamic trackers
                    
                    for a in range(dynamic_trackers.shape[1]):
                        for j in range(self.n.shape[0]):
                            e = dynamic_trackers[j][a]
                            color = self.colours[e[5].astype(int)%32]
                            my_thickness = 2
                            geometric_functions.compute_and_draw(e,color,my_thickness,output_image)
                            
                            if (self.ros):
                                object_type = "trajectory_prediction"
                                monitors_functions.tracker_to_topic(self,e,object_type,color,j) 

                        # Predict possible collision (including the predicted bounding boxes)
                        
                        collision_id,index_bb = monitors_functions.predict_collision(self.ego_predicted,dynamic_trackers[:,a]) 
                        
                        if collision_id != -1:
                            collision_id_list[0].append(collision_id)
                            collision_id_list[1].append(index_bb)     

                    # Evaluate collision with static trackers
                    
                    for b in static_trackers:
                        if b[-1]: # In route, last element of the array                     
                            collision_id,index_bb = monitors_functions.predict_collision(self.ego_predicted,b,static=True) # Predict possible collision
                            if (collision_id != -1): 
                                collision_id_list[0].append(collision_id)
                                collision_id_list[1].append(index_bb)
                    

                    #if self.nearest_object_in_route < self.ego_braking_distance:
                    #    self.collision_flag.data = True

                    





                    # Collision

                    if not self.collision_flag.data:
                        if not collision_id_list[0]: # Empty
                        #if id_nearest == -1:
                            collision_id_list[0].append(-1) # The ego-vehicle will not collide with any object
                            collision_id_list[1].append(-1) 
                            self.collision_flag.data = False 
                        elif collision_id_list[0] and (self.nearest_object_in_route < self.ego_braking_distance or self.nearest_object_in_route < 12):
                        #elif id_nearest != -1 and (self.nearest_object_in_route < self.ego_braking_distance or self.nearest_object_in_route < 12):
                            self.collision_flag.data = True  
                            self.cont = 0 

                    #print("Collision id list: ", collision_id_list)
                    if (len(collision_id_list)>1):
                        message = 'Predicted collision with objects: ' + str(collision_id_list[0])
                    else:
                        message = 'Predicted collision with object: ' + str(collision_id_list[0])
    
                    cv2.putText(output_image,message,(30,140), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2) # Predicted collision message 
            else:
                print("\033[1;33m"+"No object to track"+'\033[0;m')
                monitors_functions.empty_trackers_list(self)  

                if self.collision_flag.data:
                    print("suma B")
                    self.cont += 1
        else: 
            print("\033[1;33m"+"No objects detected"+'\033[0;m')
            monitors_functions.empty_trackers_list(self)

            if self.collision_flag.data:
                print("suma C")
                self.cont += 1

        print("cont: ", self.cont)
        if self.cont >= 3:
            self.collision_flag.data = False
            self.nearest_object_in_route = 50000
            nearest_distance.data = float(self.nearest_object_in_route)

        """
        self.end = time.time()
                
        fps = 1/(self.end-self.start)

        self.avg_fps += fps 
        self.frame_no += 1
        
        print("SORT time: {}s, fps: {}, avg fps: {}".format(round(self.end-self.start,3), round(fps,3), round(self.avg_fps/self.frame_no,3)))

        message = 'Trackers: ' + str(len(trackers))
        cv2.putText(output_image,message,(30,20), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)
        message = 'Dynamic trackers: ' + str(ndt)
        cv2.putText(output_image,message,(30,50), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)
        try:
            message = 'Static trackers: ' + str(static_trackers.shape[0])
        except:
            message = 'Static trackers: ' + str(0)
        cv2.putText(output_image,message,(30,80), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)

        # Publish the list of tracked obstacles and predicted collision
        
        print("Data: ", nearest_distance.data)
        print("Collision: ", self.collision_flag.data)

        self.pub_bev_sort_tracking_markers_list.publish(self.trackers_marker_list) 
        self.pub_collision.publish(self.collision_flag)         
        self.pub_nearest_object_distance.publish(nearest_distance)

        print("moni: ", len(monitorized_area_colours))
        self.particular_monitorized_area_list = self.mot_tracker.get_particular_monitorized_area_list(detections_rosmsg.header.stamp,
                                                                                                      monitorized_area_colours)
        self.pub_particular_monitorized_area_markers_list.publish(self.particular_monitorized_area_list)


           
        if self.write_video:
            if not self.video_flag:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.output = cv2.VideoWriter("map-filtered-mot.avi", fourcc, 20, (self.image_width, self.image_height))
            self.output.write(output_image)
       
        # Add a grid to appreciate the obstacles coordinates
        
        if (self.grid):
            gap = int(round(self.view_ahead*(self.image_width/self.real_width)))
            geometric_functions.draw_basic_grid(gap,output_image,pxstep=50)

        if(self.display):
            cv2.imshow("SORT tracking", output_image)
            cv2.waitKey(1)
        
        self.detections_subscriber = Subscriber(self.detections_topic, BEV_detections_list)
        self.odom_subscriber = Subscriber(self.odom_topic, nav_msgs.msg.Odometry)
        self.monitorized_lanes_subscriber = Subscriber(self.monitorized_lanes_topic, MonitorizedLanes)

        ts = TimeSynchronizer([self.detections_subscriber, 
                               self.odom_subscriber, 
                               self.monitorized_lanes_subscriber], 
                               header_synchro)
        ts.registerCallback(self.mot_prediction_callback)   

def main():
    print("Init the node")

    rospy.init_node('map_filtered_mot_node', anonymous=True)
    
    args = []
    display = rospy.get_param('t4ac/map-filtered-mot/display')
    args.append(display)
    trajectory_forecasting = rospy.get_param('t4ac/map-filtered-mot/trajectory-forecasting')
    args.append(trajectory_forecasting)
    use_ros = rospy.get_param('t4ac/map-filtered-mot/use-ros')
    args.append(use_ros)
    use_grid = rospy.get_param('t4ac/map-filtered-mot/use-grid')
    args.append(use_grid)
    rc_max = rospy.get_param("/controller/rc_max")
    args.append(rc_max)

    print("Display: ", display)
    print("Trajectory forecasting: ", trajectory_forecasting)
    print("Publish real-world data: ", use_ros)
    print("Use grid: ", use_grid)
    print("Road curvature max: ", rc_max)
    
    Map_Filtered_MOT(args)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down ROS Tracking node")

if __name__ == '__main__':
    main()
