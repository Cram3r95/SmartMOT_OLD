#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 02:15:37 2021

@author: Carlos Gomez-Huelamo

Code to 

Communications are based on ROS (Robot Operating Sytem)

Inputs: 

Outputs:  

Note that 

"""

import numpy as np
import math

import rospy
import geometry_msgs.msg
import visualization_msgs.msg

import geometric_functions
import monitors_functions
import tracking_functions

from t4ac_msgs.msg import Node

# SORT class #

class Sort(object):
    def __init__(self,max_age,min_hits,n,shapes,trajectory_prediction,filter_hdmap): # Ablation study
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.particular_monitorized_area_list = []
        self.trajectory_prediction = trajectory_prediction
        self.filter_hdmap = filter_hdmap

        # TFs
        
        self.tf_bevtl2bevcenter_px = np.array([[1.0, 0.0, 0.0, -shapes[3]], # From BEV Centered to BEV Top-Left 
                                               [0.0, 1.0, 0.0, -shapes[2]],
                                               [0.0, 0.0, 1.0, 0.0], 
                                               [0.0, 0.0, 0.0, 1.0]])
        
        self.tf_lidar2bev = np.array([[0.0, -1.0, 0.0,  0.0], # From LiDAR to BEV
                                      [-1.0, 0.0, 0.0,  0.0], 
                                      [0.0,  0.0, -1.0, 0.0], 
                                      [0.0,  0.0, 0.0,  1.0]])
        
        self.n = np.zeros((n,1))

    def update(self,dets,types,ego_vehicle_vel,tf_map2lidar,shapes,scale_factor,monitorized_lanes,timer_rosmsg,angle_bb,geometric_monitorized_area):
        """
        Params:
            
            dets - a numpy array of detections (x,y,w,l,theta,beta,score), where x,y are the centroid coordinates (BEV image plane), w and l the
            width and length of the obstacle (BEV image plane), theta (rotation angle), beta (angle between the ego-vehicle centroid and obstacle
            centroid) and the detection score
            
            types - a numpy array with the corresponding type of the detections
            
            ego_vehicle_vel - a float number that represents the absolute velocity of the car in the image
            
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        
        # Auxiliar variables 

        self.frame_count += 1
        self.particular_monitorized_area_list = []

        trks = np.zeros((len(self.trackers),6))
        to_del = []
        ret = []

        ret_static = []
        ret_dynamic = [] 
        in_route = False
        for i in range(self.n.shape[0]):
            ret_dynamic.append([])
        dyn = 0
            
        ret_type = []
        ret_score = []
        ret_angles = [] 

        # 1. Get predicted locations from existing trackers.
        
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]

            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], 0] 
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)
        #print("Detections: ", dets, len(dets))
        #print("Trackers: ", trks, len(trks))

        matched, unmatched_dets, unmatched_trks = tracking_functions.associate_detections_to_trackers(dets,trks) # Hungarian algorithm

        #print("Matched: ", matched)
        #print("Unmatched dets: ", unmatched_dets)
        #print("Unmatched trackers: ", unmatched_trks)

        # 2. Update matched trackers with assigned detections and evaluate its position inside the monitorized lanes
        num_trks = len(self.trackers)

        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0][0]
                ret_type.append(types[d])
                ret_score.append(dets[d,6])
            
                #print("Det theta: ", dets[d,4])
                #print("Tracker theta: ", trk.get_state()[0][4])
                if (abs(dets[d,4] - trk.get_state()[0][4]) > math.pi/2): # If the difference between the matched detection and its associated tracker
                                                                         # in two consecutives frames is greater than pi/2 radians, probably the detection is the opposite                                                     
                    dets[d,4] = dets[d,4] + math.pi
                    if (dets[d,4] > 1.8*math.pi): # If the tracker orientation is close to 0, and the detection is close to pi, if we sum pi, the result
                        # is 2*pi, but it does not make sense (huge angular velocity produced), so now substract 2*pi
                        dets[d,4] = dets[d,4] - 2*math.pi

                observation_angle = dets[d,5] + dets[d,4] # Beta + Theta (According to KITTI framework)
                angles = np.array([[dets[d,4],observation_angle]])    
                ret_angles.append(angles)  
                
                # Update the space state 
                                                                    
                trk.update(dets[d,:])
                
                # Get the current global position 
                """
                real_world_x, real_world_y, _, _ = geometric_functions.pixels2realworld(self,trk,shapes)

                aux_array = np.zeros((1,9))
                aux_array[0,4] = trk.kf.x[4]
                aux_array[0,7:] = real_world_x, real_world_y

                current_pos = store_global_coordinates(tf_map2lidar,aux_array)

                trk.current_pos[:2,0] = current_pos[:2,0] # x,y global centroid
                trk.current_pos[2,0] = current_pos[4,0] # global orientation
                """

                #print("kf: ", trk.kf.x, trk.kf.x.shape)
                trk.current_pos[:2,0] = trk.kf.x[:2,0]
                trk.current_pos[2,0] = trk.kf.x[4]









                if self.filter_hdmap:
                # Check if it is inside the monitorized lanes

                    detection = Node()
                    detection.x = trk.current_pos[0,0]
                    detection.y = -trk.current_pos[1,0]

                    trk_is_inside, trk_in_route, trk_particular_monitorized_area = evaluate_detection_in_monitorized_lanes(detection, monitorized_lanes, types[d])

                    if trk_particular_monitorized_area:
                        self.particular_monitorized_area_list.append(trk_particular_monitorized_area)

                    prediction_in_route = False

                    if trk_is_inside:
                        # Keep the tracker if it is inside the monitorized area (not only the road)

                        trk.off_road = 0

                        # Calculate the global velocities (to predict its position)
                        
                        trk.calculate_global_velocities_and_distance(ego_vehicle_vel,scale_factor,timer_rosmsg)
                        
                        trk.abs_vel = math.sqrt(pow(trk.global_velocities[0,0],2)+pow(trk.global_velocities[1,0],2))
                        #print("Tracker abs vel: ", trk.abs_vel)
                        if (trk.abs_vel > 15):
                            print("\033[1;36m"+"Huge velocity"+'\033[0;m')
                        else:
                            if (trk.abs_vel > 0.5):
                                trk.trajectory_prediction(angle_bb) # Get the predicted bounding box in n seconds 

                                if not trk_in_route:

                                    # Check if the prediction in the road at this moment

                                    # Get the predicted global position 

                                    predicted_bb = trk.trajectory_prediction_bb[:,-1]



                                    """
                                    real_world_x, real_world_y, _, _ = geometric_functions.pixels2realworld_prediction(self,predicted_bb,shapes)

                                    aux_array = np.zeros((1,9))
                                    aux_array[0,4] = trk.kf.x[4]
                                    aux_array[0,7:] = real_world_x, real_world_y

                                    current_pos = store_global_coordinates(tf_map2lidar,aux_array)

                                    trk.current_pos[:2,0] = current_pos[:2,0] # x,y global centroid
                                    trk.current_pos[2,0] = current_pos[4,0] # global orientation
                                    """











                                    # Check if it is inside the monitorized lanes

                                    detection = Node()
                                    detection.x = predicted_bb[0]
                                    detection.y = -predicted_bb[1]

                                    for lane in monitorized_lanes.lanes: 
                                        if len(lane.left.way) >= 2 and (lane.role == "current" or lane.role == "front"):
                                            is_inside, in_road, aux_monitorized_area, dist2centroid = monitors_functions.inside_lane(lane,detection,types[d])

                                            if in_road:
                                                prediction_in_route = True
                                                break
                            else:
                                trk.trajectory_prediction_bb = np.zeros((5,self.n.shape[0])) # If it is static, the prediction buffer must be zero

                        print("IN ROUTE 1: ", trk_in_route)
                        print("IN ROUTE: ", prediction_in_route)
                        if trk_in_route or prediction_in_route:      
                                   
                            trk.in_route = 1 # In the road, current lane or next lane             
                    else:
                        trk.off_route += 1
                else:
                    trk.calculate_global_velocities_and_distance(ego_vehicle_vel,scale_factor,timer_rosmsg)

                    x = real_world_x
                    y = real_world_y

                    print("goemetric area: ", geometric_monitorized_area)
                    print("x y: ", x,y)
                    if (x < geometric_monitorized_area[0] and x > geometric_monitorized_area[1]
                        and y < geometric_monitorized_area[2] and y > geometric_monitorized_area[3]):
                        print("in route geometric")
                        trk.in_route = 1
                    else:
                        trk.in_route = 0
                        
                    trk.abs_vel = math.sqrt(pow(trk.global_velocities[0,0],2)+pow(trk.global_velocities[1,0],2))
                    #print("Tracker abs vel: ", trk.abs_vel)
                    if (trk.abs_vel > 15):
                        print("\033[1;36m"+"Huge velocity"+'\033[0;m')
                    else:
                        if (trk.abs_vel > 1.0):
                            trk.trajectory_prediction(angle_bb) # Get the predicted bounding box in n seconds 
                        else:
                            trk.trajectory_prediction_bb = np.zeros((5,self.n.shape[0])) # If it is static, the prediction buffer must be zero

        # 3. Create and initialise new trackers for unmatched detections (if these detections are inside a monitorized_lane)

        for i in unmatched_dets:
            print("total dets: ", dets)
            print("dets: ", dets[i,:])
            aux = store_global_coordinates(tf_map2lidar,dets[i,:].reshape(1,9))
            type_object = types[i]

            if self.filter_hdmap:
                detection = Node()
                detection.x = aux[0,0]
                detection.y = -aux[1,0] # N.B. In CARLA this coordinate is the opposite

                is_inside, in_route, particular_monitorized_area = evaluate_detection_in_monitorized_lanes(detection,monitorized_lanes, type_object)

                if is_inside: # Create new tracker 
                    trk = tracking_functions.KalmanBoxTracker(dets[i,:],timer_rosmsg)
                    trk.current_pos[:2,0] = aux[:2,0] # x,y global centroid
                    trk.current_pos[2,0] = aux[4,0] # global orientation
                    trk.particular_monitorized_area = particular_monitorized_area

                    if in_route:
                        trk.in_route = 1

                    self.trackers.append(trk)

                if particular_monitorized_area:
                    self.particular_monitorized_area_list.append(particular_monitorized_area)

            else:
                #print("Create new tracker")
                trk = tracking_functions.KalmanBoxTracker(dets[i,:],timer_rosmsg)
                trk.current_pos[:2,0] = aux[:2,0] # x,y global centroid
                trk.current_pos[2,0] = aux[4,0] # global orientation 

                x = trk.current_pos[0,0]
                y = trk.current_pos[1,0]

                if (x < geometric_monitorized_area[0] and x > geometric_monitorized_area[1]
                    and y < geometric_monitorized_area[2] and y > geometric_monitorized_area[3]):
                    trk.in_route = 1
                else:
                    trk.in_route = 0

                self.trackers.append(trk)  

        i = len(self.trackers)
        print("Number of trackers: ", i)
        # 4. Store relevant trackers in lists

        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):  
                d = trk.get_state()[0] # Predicted state in next frame
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                # id+1 as MOT benchmark requires positive 
                ret.append(np.concatenate((d,[trk.id+1],[trk.in_route])).reshape(1,-1)) 

                # Distinguish between dynamic and static obstacles

                if self.trajectory_prediction: 
                    for k in range(self.n.shape[0]):
                        e = trk.get_trajectory_prediction_bb(k)[0] # Predicted the bounding box position
                        all_zeros = not np.any(e) # True if all elements of the array are equal to 0

                        if not all_zeros: # Only append if there are predicted bbs for the current tracker
                            ret_dynamic[k].append(np.concatenate((e,[trk.id+1])).reshape(1,-1))
                        else: 
                            ret_static.append(np.concatenate((d,[trk.id+1],[trk.in_route])).reshape(1,-1))
                            break
            i -= 1
            
            # Remove dead tracklet
            
            if(trk.time_since_update > self.max_age or trk.off_route > self.max_age):
                self.trackers.pop(i)
   
        dyn = len(ret_dynamic[0])

        if not ret_static: # The list is empty
            ret_static.append([]) # We need something into the list to be converted to a np.array
   
        if(len(ret)>0 and self.frame_count > 1):       
            a = np.concatenate(ret)   
            b = np.array(ret_type)   
            c = np.array(ret_score)  
            d = np.concatenate(ret_angles)

            try:
                e = np.concatenate(ret_dynamic).reshape(self.n.shape[0],dyn,6)
            except:
                e = np.empty((0,0))
            try:
                f = np.concatenate(ret_static)
            except:
                f = np.empty((0,0))
       
            return a,b,c,d,e,f     
        return np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),np.empty((0,6))

    def get_particular_monitorized_area_list(self, timestamp, monitorized_area_colours):
        """
        """

        particular_monitorized_area_marker_list = visualization_msgs.msg.MarkerArray()

        if len(monitorized_area_colours) == len(self.particular_monitorized_area_list):
            for i, area in enumerate(self.particular_monitorized_area_list):
                particular_monitorized_area_marker = visualization_msgs.msg.Marker()

                particular_monitorized_area_marker.header.frame_id = "/map"
                particular_monitorized_area_marker.header.stamp = timestamp
                particular_monitorized_area_marker.ns = "particular_monitorized_areas"
                particular_monitorized_area_marker.action = particular_monitorized_area_marker.ADD
                particular_monitorized_area_marker.lifetime = rospy.Duration.from_sec(1)
                particular_monitorized_area_marker.id = i
                particular_monitorized_area_marker.type = visualization_msgs.msg.Marker.LINE_STRIP

                color = monitorized_area_colours[i]

                particular_monitorized_area_marker.color.r = color[2]
                particular_monitorized_area_marker.color.g = color[1]
                particular_monitorized_area_marker.color.b = color[0]
                particular_monitorized_area_marker.color.a = 1.0

                # particular_monitorized_area_marker.color.r = 1.0
                # particular_monitorized_area_marker.color.g = 0.0
                # particular_monitorized_area_marker.color.b = 1.0
                # particular_monitorized_area_marker.color.a = 1.0

                particular_monitorized_area_marker.scale.x = 0.25
                particular_monitorized_area_marker.pose.orientation.w = 1.0

                for p in area:
                    point = geometry_msgs.msg.Point()

                    point.x = p.x
                    point.y = -p.y
                    point.z = 0.2

                    particular_monitorized_area_marker.points.append(point)

                point = geometry_msgs.msg.Point()
                #print("Area: ", area)
                point.x = area[0].x
                point.y = -area[0].y

                particular_monitorized_area_marker.points.append(point) # To close the polygon

                particular_monitorized_area_marker_list.markers.append(particular_monitorized_area_marker)

        return particular_monitorized_area_marker_list

# End SORT class #

def evaluate_detection_in_monitorized_lanes(detection,monitorized_lanes,type_object):
    """
    This function evaluates the trackers (its global coordinates centroid, according to "/map" frame) in the monitorized_lanes
    """  

    is_inside = False
    in_route = False
    monitorized_area = []
    dist2centroid = 99999

    for lane in monitorized_lanes.lanes: 
        if len(lane.left.way) >= 2 and not is_inside:
            aux_is_inside, aux_in_road, aux_monitorized_area, aux_dist2centroid = monitors_functions.inside_lane(lane,detection,type_object)
            #is_inside, particular_monitorized_area, dist2centroid, particular_road_area = monitors_functions.inside_lane(lane,detection,type_object)

            if aux_is_inside:
                if aux_dist2centroid < dist2centroid:
                    dist2centroid = aux_dist2centroid
                    monitorized_area = aux_monitorized_area

                is_inside = True

                if (aux_in_road and (lane.role == "current" or lane.role == "front")):
                    in_route = True
                    break

    return is_inside, in_route, monitorized_area        

def store_global_coordinates(tf_map2lidar,det):  
  """
  This function returns the global ("/map" frame) x,y coordinates of an object (m) 
  """
  lidar_centroid = np.array([0.0,0.0,0.0,1.0]).reshape(4,1)
  lidar_centroid[0,0] = det[0,7] 
  lidar_centroid[1,0] = det[0,8]

  aux = np.dot(tf_map2lidar,lidar_centroid) # == TF @ lidar_centroid 
  o = np.array(det[0,4]).reshape(1,1)
  aux = np.vstack((aux,o)) # Concatenate the global orientation

  return aux

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x,y,w,l,theta,score] and returns z in the form
  [x,y,s,r,theta] where x,y is the centre of the box, s is the scale/area, r is
  the aspect ratio and theta is the bounding box angle
  """

  x = bbox[0]
  y = bbox[1]
  w = bbox[2]
  h = bbox[3]
  
  s = w*h         # Area of the rectangle
  r = w/float(h)  # Aspect ratio of the rectangle
  theta = bbox[4]

  return np.array([x,y,s,r,theta]).reshape((5,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r,theta] and returns it in the form
    [x,y,w,l,theta] where x, y are the centroid, w and l the bounding box dimensions (in pixels)
    and theta is the bounding box angle
    """

    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    theta = x[4]

    if not score:
        return np.array([x[0],x[1],w,h,theta]).reshape((1,5))
    else:
        return np.array([x[0],x[1],w,h,theta,score]).reshape((1,6))

def bbox_to_xywh_cls_conf(self,detections_rosmsg):
    """
    """

    bboxes = []
    types = []
    k = 0

    # Evaluate detections

    for bbox_object in detections_rosmsg.bev_detections_list:
        if (bbox_object.score >= self.detection_threshold):

            bbox_object.x_corners = np.array(bbox_object.x_corners) # Tuple to np.ndarray
            bbox_object.y_corners = np.array(bbox_object.y_corners) 

            # Gaussian noise (If working with the groundtruth)

            if self.use_gaussian_noise:
                mu = 0
                sigma = 0.05 
                
                x_offset, y_offset = np.random.normal(mu,sigma), np.random.normal(mu,sigma)

                bbox_object.x_corners += x_offset
                bbox_object.y_corners += y_offset
                
                bbox_object.x += x_offset
                bbox_object.y += y_offset

                theta = bbox_object.o # self.ego_orientation_cumulative_diff # Orientation angle (KITTI)
                # + math.pi/2 if using AB4COGT2SORT
                # + self.ego_orientation_cumulative_diff if using PointPillars
                beta = np.arctan2(bbox_object.x-self.ego_vehicle_x,self.ego_vehicle_y-bbox_object.y) # Observation angle (KITTI)

            # Calculate bounding box dimensions

            w = math.sqrt(pow(bbox_object.x_corners[3]-bbox_object.x_corners[1],2)+pow(bbox_object.y_corners[3]-bbox_object.y_corners[1],2))
            l = math.sqrt(pow(bbox_object.x_corners[0]-bbox_object.x_corners[1],2)+pow(bbox_object.y_corners[0]-bbox_object.y_corners[1],2))

            # Translate local to global coordinates

            aux_array = np.zeros((1,9))
            aux_array[0,4] = bbox_object.o
            aux_array[0,7:] = bbox_object.x, bbox_object.y

            current_pos = store_global_coordinates(self.tf_map2lidar,aux_array)
     
            if k == 0:
                bboxes = np.array([[current_pos[0,0],current_pos[1,0], 
                                    w, l,
                                    theta, beta,
                                    bbox_object.score,
                                    bbox_object.x,bbox_object.y]])
                
                types = np.array([bbox_object.type])
            else:
                bbox = np.array([[current_pos[0,0],current_pos[1,0], 
                                w, l,
                                theta, beta,
                                bbox_object.score,
                                bbox_object.x,bbox_object.y]])

                type_object = np.array([bbox_object.type])
                bboxes = np.concatenate([bboxes,bbox])
                types = np.concatenate([types,type_object])
            k += 1  
        bboxes = np.array(bboxes)

    return bboxes, types