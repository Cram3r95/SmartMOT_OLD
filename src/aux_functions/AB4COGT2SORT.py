#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:07:29 2020

AB4COGT to SORT Data Generator

@author: Javier del Egido & Carlos Gómez-Huélamo

"""

# General purpose imports

import numpy as np
from scipy.spatial.distance import euclidean

# ROS imports

import rospy
from carla_msgs.msg import CarlaEgoVehicleInfo
from derived_object_msgs.msg import ObjectArray
from tf.transformations import euler_from_quaternion
from t4ac_msgs.msg import BEV_detection, BEV_detections_list

classification_list = ["unknown",
			"Unknown_Small",
			"Unknown_Medium",
			"Unknown_Big",
			"Pedestrian",
			"Cyclist",
			"Car",
			"Truck",
			"Motorcycle",
			"Other_Vehicle",
			"Barrier",
			"Sign"]

class AB4COGT2SORT():
    def rotz(self,t):
        """
        Rotation about the z-axis
        """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  -s,  0],
                         [s,   c,  0],
                         [0,   0,  1]])
    
    def listener(self):

        self.first_time = 0 #Flag
        self.first_seq_value = 0 #Stores first seq value to start on 0

        rospy.init_node('CarlaGroundtruth2SORT', anonymous=True)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_info", CarlaEgoVehicleInfo, self.ego_vehicle_callback)
        self.pub = rospy.Publisher("/t4ac/perception/detection/merged_obstacles", BEV_detections_list, queue_size=10)
        rospy.spin()
        
    def ego_vehicle_callback(self, data):
        print("Receiving ego_vehicle")
        self.ego_vehicle_id = data.id
        rospy.Subscriber("/carla/objects", ObjectArray, self.callback)
        
    def callback(self, data):
        
        if self.first_time ==0:
            self.first_time = 1
            self.first_seq_value = data.header.seq
            
        #Firstly, find ego_vehicle position
        for obj1 in range(len(data.objects)):
            identity 	= data.objects[obj1].id
            if identity == self.ego_vehicle_id:
                xyz_ego = data.objects[obj1].pose.position
                quat_ego = data.objects[obj1].pose.orientation
                break
        location_ego = [xyz_ego.x, xyz_ego.y, xyz_ego.z]

        #frame = data.header.seq - self.first_seq_value
        
        pp_list = BEV_detections_list()		##CREO EL MENSAJE
        pp_list.header.stamp = rospy.Time.now()
        pp_list.front = 30
        pp_list.back = -15
        pp_list.left = -15
        pp_list.right = 15
        published_obj = 0
        for obj in range(len(data.objects)):

            identity = data.objects[obj].id


            if identity != self.ego_vehicle_id:
							
                xyz = data.objects[obj].pose.position
                location = [xyz.x, xyz.y, xyz.z]

                #Only store groundtruth of objects in Lidar range
                #if euclidean(location, location_ego) < 25: #Compare to LiDAR range
                    #Get data from object topic
                
                quat_xyzw 	= data.objects[obj].pose.orientation
                lhw 		= data.objects[obj].shape.dimensions
                label 		= classification_list[data.objects[obj].classification]
                
                #Calculate heading and alpha (obs_angle)
                quaternion 	= np.array((quat_xyzw.x, quat_xyzw.y, quat_xyzw.z, quat_xyzw.w))
                heading 	= euler_from_quaternion(quaternion)[2] - np.pi/2
                beta = np.arctan2(xyz.x - xyz_ego.x, xyz_ego.y - xyz.y)
                obs_angle = ( (heading) + (beta) )

                wlh  = [lhw[0], lhw[1], lhw[2]]
                location_local = (np.asarray(location) - np.asarray(location_ego)).tolist()

                quaternion_ego = np.array((quat_ego.x, quat_ego.y, quat_ego.z, quat_ego.w))
                heading_ego    = euler_from_quaternion(quaternion_ego)[2]
                R = self.rotz(-heading_ego)
                location_local = np.dot(R, location_local)#[0:2]
               
                if (location_local[0] > pp_list.back) and (location_local[0] < pp_list.front) \
                    and (location_local[1] > pp_list.left) and (location_local[1] < pp_list.right):
                    published_obj += 1
                    box = [0,0,0,0]
                    #print("\nheading", heading_ego)
                    #print("location_local", location_local)
                    #out_data= [frame] + [identity] + [label] + [0, 0] + [obs_angle] + box + wlh + location_local + [heading]
    
    
                    #Publish in ROS topic
                    obj = BEV_detection()
                    #seq     = frame
                    obj_id  = int(identity)
                    #alpha   = obs_angle
                    wlh     = wlh
                    xyz     = location_local
                    heading = -heading

                    obj.type = label
                    obj.score = 0.99
                    #obj.object_id = obj_id
                    
                    w = wlh[0]
                    l = wlh[1]                   
                    h = wlh[2]

                    if w < 1.5 or l < 1.5:
                        w = 2.0
                        l = 2.0
                    #print("l, w, h: ", l,w,h)
                    if heading > np.pi:
                        heading = heading - np.pi
                    R = self.rotz(heading-np.pi/2)
                        
                    # 3d bounding box corners
                    #x_corners = [-l/2,-l/2,l/2, l/2]
                    #y_corners = [ w/2,-w/2,w/2,-w/2]
                    #z_corners = [0,0,0,0]

                    x_lidar = xyz[0]
                    y_lidar = xyz[1]
                    z_lidar = xyz[2]-3

                    obj.x = xyz[0]
                    obj.y = xyz[1]

                    x_corners = [-l/2,-l/2,l/2, l/2,-l/2,-l/2,l/2, l/2]
                    y_corners = [ w/2,-w/2,w/2,-w/2, w/2,-w/2,w/2,-w/2]
                    z_corners = [   0,   0,  0,   0,   h,   h,  h,   h]
                    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
                    corners_3d = corners_3d + np.vstack([x_lidar, y_lidar, z_lidar])
                    #obj.x_corners_3D = [corners_3d[0,0], corners_3d[0,1], corners_3d[0,2], corners_3d[0,3], corners_3d[0,4], corners_3d[0,5], corners_3d[0,6], corners_3d[0,7]]
                    #obj.y_corners_3D = [corners_3d[1,0], corners_3d[1,1], corners_3d[1,2], corners_3d[1,3], corners_3d[1,4], corners_3d[1,5], corners_3d[1,6], corners_3d[1,7]]
                    #obj.z_corners_3D = [corners_3d[2,0], corners_3d[2,1], corners_3d[2,2], corners_3d[2,3], corners_3d[2,4], corners_3d[2,5], corners_3d[2,6], corners_3d[2,7]]
                    
                    # rotate and translate 3d bounding box
                    #x_bev = -xyz[1]
                    #y_bev = -xyz[0]

                    x_bev = xyz[0]
                    y_bev = xyz[1]




                    #corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))[0:2]
                    #corners_3d = corners_3d + np.vstack([x_bev, y_bev])
                    
                    obj.x = x_bev   #in LiDAR BEV coordinates
                    obj.y = y_bev   #in LiDAR BEV coordinates
                    
                    obj.tl_br = [0,0,0,0] #2D bbox top-left, bottom-right  xy coordinates
                    obj.x_corners = [corners_3d[0,0], corners_3d[0,1], corners_3d[0,2], corners_3d[0,3]] #Array of x coordinates (upper left, upper right, lower left, lower right)
                    obj.y_corners = [corners_3d[1,0], corners_3d[1,1], corners_3d[1,2], corners_3d[1,3]]


                    obj.l = l  #in lidar_frame coordinates
                    obj.w = w  #in lidar_frame coordinates
                    #obj.h = h  #in lidar_frame coordinates
                    obj.o = heading      #in lidar_frame coordinates

                    pp_list.bev_detections_list.append(obj)


        if published_obj == 0:
            obj = BEV_detection()
            pp_list.bev_detections_list.append(obj)
                    
        self.pub.publish(pp_list) 		##PUBLICO
    

        

if __name__ == '__main__':

    program = AB4COGT2SORT()
    program.listener()
