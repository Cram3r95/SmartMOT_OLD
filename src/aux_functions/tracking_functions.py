#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 02:19:04 2021

@author: Carlos Gomez-Huelamo

Code to conduct tracking-by-detection, using a standard Kalman
Filter as State Estimation and Hungarian Algorithm as Data Association. Trajectory prediction
is carried out assuming a CTRV model (Constant Turn Rate and Velocity magnitude model)

Communications are based on ROS (Robot Operating Sytem)

Inputs: 

Outputs:  

Note that 

"""

import numpy as np
import math
import time

import geometric_functions
import monitors_functions
import sort_functions
from filterpy.kalman import KalmanFilter # Bayesian filters imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.linear_assignment_ import linear_assignment # Hungarian Algorithm 

# Kalman

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  We assume constant velocity model
  """
  count = 0
  def __init__(self,bbox,timer_rosmsg):
    """
    Initialises a tracker using initial bounding box.
    """
    n_states = 9
    n_measurements = 5
    
    self.kf = KalmanFilter(dim_x=n_states, dim_z=n_measurements) # 9 variable vector, 5 measures
    
    self.previous_kf_measurements = np.zeros((n_measurements,1))
    self.global_velocities = np.zeros((3,1)) # x,y,theta
    
    n = 4
    self.n = np.zeros((n,1))
    self.trajectory_prediction_bb = np.zeros((5,self.n.shape[0])) # (x,y,s,r,theta) n times
    self.in_route = 0
    self.off_route = 0
    self.particular_monitorized_area = []

    # Velocity-Braking distance model
    
    self.velocity_braking_distance_model = monitors_functions.fit_velocity_braking_distance_model()
    self.pf = PolynomialFeatures(degree = 2)
    
    self.current_pos = np.zeros((3,1)) # Global coordinates ("/map" frame) (x,y,orientation)
    self.previous_pos = np.zeros((3,1)) 
    self.distance = 0 # Euclidean distance between the previous and current global position
    
    # Transition matrix: x(k+1) = F*x(k)
    
    self.kf.F = np.array([[1,0,0,0,0,1,0,0,0],  # x
                          [0,1,0,0,0,0,1,0,0],  # y
                          [0,0,1,0,0,0,0,1,0],  # s
                          [0,0,0,1,0,0,0,0,0],  # r
                          [0,0,0,0,1,0,0,0,1],  # theta
                          [0,0,0,0,0,1,0,0,0],  # x'
                          [0,0,0,0,0,0,1,0,0],  # y'
                          [0,0,0,0,0,0,0,1,0],  # s'
                          [0,0,0,0,0,0,0,0,1]]) # theta'

    # Measurement matrix: z(k) = H*x(k)
    
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0],  # x
                          [0,1,0,0,0,0,0,0,0],  # y
                          [0,0,1,0,0,0,0,0,0],  # s
                          [0,0,0,1,0,0,0,0,0],  # r
                          [0,0,0,0,1,0,0,0,0]]) # theta

    # Measurement uncertainty/noise matrix
    
    self.kf.R[2:,2:] *= 10. # So s, r and theta are affected
    
    # Covariance matrix 
    
    self.kf.P[4:,4:] *= 10000. # Give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    
    # Process uncertainty/noise matrix 
    
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[-2,-2] *= 0.01
    self.kf.Q[5:,5:] *= 0.01

    # Filter state estimate matrix (Initial state)

    self.kf.x[:5] = sort_functions.convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    
    self.start = timer_rosmsg
    self.hits_time = float(0)
    self.average_diff = float(0) # Average sample time

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    
    self.hits_time += 1

    self.kf.update(sort_functions.convert_bbox_to_z(bbox)) 

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """ 
    if((self.kf.x[7]+self.kf.x[2])<=0):
      self.kf.x[7] *= 0.0
    
    self.kf.predict()
                      
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(sort_functions.convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def trajectory_prediction(self,angle_bb):
    """
    This function returns the bounding box n seconds ahead
    """
    abs_vel = math.sqrt(self.global_velocities[0,0]**2+self.global_velocities[1,0]**2) # m/s
    vel_km_h = abs_vel*3.6 # m/s to km/h
    
    braking_distance = self.velocity_braking_distance_model.predict(self.pf.fit_transform([[vel_km_h]]))
    seconds_required = braking_distance/abs_vel # m / m/s

    if seconds_required < 3:
      seconds_required = 3

    k = 1
    for i in range(self.n.shape[0]):
        #self.n[i] = (seconds_required/self.n.shape[0])*(i+1)
        self.n[i] = k
        k += 1

        if i>0:
            previous_angle = self.trajectory_prediction_bb[4,i-1] + angle_bb 
        else:
            previous_angle = self.kf.x[4] + angle_bb  # Current angle
        #print("Previous angle: ", previous_angle)
        self.trajectory_prediction_bb[0,i] = self.kf.x[0] + abs_vel*math.cos(previous_angle)*self.n[i] # x centroid
        self.trajectory_prediction_bb[1,i] = self.kf.x[1] + abs_vel*math.sin(previous_angle)*self.n[i] # y centroid
        self.trajectory_prediction_bb[2,i] = self.kf.x[2] # s (scale) # TODO: Consider this in global coordinates?
        self.trajectory_prediction_bb[3,i] = self.kf.x[3] # r (aspect ratio) is assumed to be constant
        self.trajectory_prediction_bb[4,i] = self.kf.x[4] + self.global_velocities[2,0]*self.n[i]  # Theta (orientation) TODO ????? math.pi/2
    
  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return sort_functions.convert_x_to_bbox(self.kf.x)

  def get_trajectory_prediction_bb(self,i):
    """
    Returns the trajectory prediction of the bounding box.
    """
    all_zeros = not np.any(self.trajectory_prediction_bb[:,i]) # True if all elements of the matrix are 0
            
    if not all_zeros: # Only append if there are predicted bbs for the current tracker
        return sort_functions.convert_x_to_bbox(self.trajectory_prediction_bb[:,i])
    else:
        return np.zeros((4,1))

  def calculate_global_velocities_and_distance(self,ego_vehicle_vel_px,scale_factor,timer_rosmsg):
    # TODO: Improve this calculation: 
    #       RADAR (relative velocity) + ego_vehicle_velocity
    #       From kalman_relative_velocity (krv) (pixels) to real_world_relative + ego_vehicle_velocity
    
    krv = False

    # TODO: Test with the vehicle stopped and an obstacle moving

    if self.hit_streak > 1:# and not krv:
      self.end = timer_rosmsg
      self.diff = self.end - self.start
      self.start = self.end

      self.current_pos[0,0] = (self.current_pos[0,0] + self.previous_pos[0,0]) / 2

      self.global_velocities[0,0] = (self.current_pos[0,0] - self.previous_pos[0,0])/self.diff # x'
      self.global_velocities[1,0] = (self.current_pos[1,0] - self.previous_pos[1,0])/self.diff # y'
      self.global_velocities[2,0] = (self.current_pos[2,0] - self.previous_pos[2,0])/self.diff # theta'

      self.distance = np.linalg.norm(self.current_pos[:2,0] - self.previous_pos[:2,0])
    """
    else:
      print("Inferred x: ", self.kf.x[5])
      print("Inferred y: ", self.kf.x[6])
      print("ego vehicle: ", ego_vehicle_vel_px)
      self.global_velocities[0,0] = self.kf.x[5]/scale_factor[1] # px/s to m/s
      self.global_velocities[1,0] = (self.kf.x[6] - ego_vehicle_vel_px)/scale_factor[0] # px/s to m/s
      self.global_velocities[2,0] = -self.kf.x[8]
    """
    self.previous_pos[:3] = self.current_pos[:3]

# Data association

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.001):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """

  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,6),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  # Matched detections

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = geometric_functions.iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix) # Hungarian Algorithm

  # Unmatched detections
  
  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
      
  # Unmatched trackers
    
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  # Filter out matched with low IOU
  
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)