#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 17:36:17 2020

@author: Carlos Gómez Huélamo

Code to 

Communications are based on ROS (Robot Operating Sytem)

Inputs: 

Outputs:  

Note that 

"""

import numpy as np
import math
import cv2

from shapely.geometry import Polygon

def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    try:
        x,y,z = b.x, b.y, 0.0
        X,Y,Z = e.x, e.y, 0.0
    except:
        x,y,z = b
        X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v

    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    try:   
        X,Y,Z = w.x, w.y, 0.0
    except:
        X,Y,Z = w
    return (x+X, y+Y, z+Z)

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def rotz(t):
    """ 
    Rotation about the z-axis
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                     [s,   c,  0],
                     [0,   0,  1]])

def compute_corners_real(bbox,shapes=None,aux_centroid=None):
    """
    Compute the corners of the rectangle given its x,y centroid, width and length and
    its rotation (clockwise if you see the screen, according to OpenCV)
    
    If real_world is not None, compute the length and witdth of the real world tracker based 
    on the tracker state (Bird's Eye View camera frame)
    """
  
    rotation = bbox[4]
    
    if rotation > math.pi:
        rotation = rotation - math.pi
    
    R = rotz(rotation)

    # 3D bounding box corners

    x, y, w, l = bbox[0], bbox[1], bbox[2], bbox[3]

    x_corners = [-l/2,-l/2,l/2,l/2]
    y_corners = [w/2,-w/2,w/2,-w/2]
    z_corners = [0,0,0,0]

    corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d = np.dot(R, corners_3d)[0:2]
    corners_3d = corners_3d + np.vstack([x,y])
    
    return corners_3d

def compute_corners(bbox,shapes=None,aux_centroid=None):
    """
    Compute the corners of the rectangle given its x,y centroid, width and length and
    its rotation (clockwise if you see the screen, according to OpenCV)
    
    If real_world is not None, compute the length and witdth of the real world tracker based 
    on the tracker state (Bird's Eye View camera frame)
    """
    
    rotation = bbox[4]
    
    if rotation > math.pi:
        rotation = rotation - math.pi
    
    R = rotz(rotation)

    # 3D bounding box corners

    x, y, w, l = bbox[0], bbox[1], bbox[2], bbox[3]

    x_corners = [-l/2,-l/2,l/2,l/2]
    y_corners = [w/2,-w/2,w/2,-w/2]
    z_corners = [0,0,0,0]

    corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d = np.dot(R, corners_3d)[0:2]
    corners_3d = corners_3d + np.vstack([x,y])
    
    if not shapes: # Obtain the coordinates in BEV camera frame
        corners = []
        for i in range(4):
            c = int(round(corners_3d[0,i].item())), int(round(corners_3d[1,i].item()))
            corners.append(c)
        corners = tuple(corners)
        return corners
    else: # Obtain the real-world dimensions
        corners = np.array([])
        for i in range(4):
            a = corners_3d[0,i]*(shapes[1] / shapes[3])
            b = corners_3d[1,i]*(shapes[0] / shapes[2])
            if i == 0:
                corners = np.array([[a],[b]])
            else:
                c = np.array([[a],[b]])
                corners = np.vstack((corners,c))

        real_world_x = aux_centroid[0]*(shapes[1]/shapes[3])
        real_world_y = aux_centroid[1]*(shapes[0]/shapes[2])
        real_world_w = math.sqrt((corners[2]-corners[0])**2 + (corners[3]-corners[1])**2)
        real_world_l = math.sqrt((corners[6]-corners[2])**2 + (corners[7]-corners[3])**2)

        return real_world_x, real_world_y, real_world_w, real_world_l
        
def draw_rotated(contour,centroid,img,my_thickness,color=None):
    """ 
    Draw a rotated rectangle 
    """
    # Contour of the rectangle

    rotrect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotrect)
    box = np.rint(box).astype(int)

    # Color of the contour and its centroid

    if not color: 
        color = (255,255,255)
    cv2.drawContours(img, [box], 0, color, thickness = my_thickness)

    # Corners

    paint_corners = True

    if (paint_corners == True):
        radius = 5
        colours = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
        thickness = 1

        for i in range(len(contour)):
            reference = tuple(contour[i])
            colour = colours[i]
            cv2.circle(img, reference, radius, colour, thickness)

    # Centroid
    
    radius = 4
    thickness = 3
    cv2.circle(img, centroid, radius, color, thickness)

def compute_and_draw(d,color,my_thickness,output_image):
    """
    """
    d1c, d2c, d3c, d4c = compute_corners(d) # Tracker corners
    contour = np.array([[d1c[0],d1c[1]],[d2c[0],d2c[1]],[d3c[0],d3c[1]],[d4c[0],d4c[1]]])     
    centroid = (d[0].astype(int),d[1].astype(int)) # Tracker centroid
    color = 255*color[0], 255*color[1], 255*color[2] # Numpy float64 array to tuple   
    draw_rotated(contour,centroid,output_image,my_thickness,color)

def iou(bb_1,bb_2): 
  """
  Computes IOU between two (possibly) rotated bounding boxes in the form [x,y,w,l,theta]
  """

  corners_1 = compute_corners(bb_1)
  corners_2 = compute_corners(bb_2)
  #print("corners 1: ", corners_1)
  #print("corners 2: ", corners_2)
  # To build the polygon -> Left-bottom corner, Right-bottom, Top-right corner, Top-left corner

  b1 = Polygon([corners_1[2],corners_1[3],corners_1[1],corners_1[0]]) 
  b2 = Polygon([corners_2[2],corners_2[3],corners_2[1],corners_2[0]])

  o = b1.intersection(b2).area / b1.union(b2).area
  #print("o: ", o)
  return(o)

def draw_basic_grid(gap,output_image,pxstep):
    """
    This function draws a simple grid over the image based on the passed step
    The pxstep controls the size of the grid
    """
    x = 0
    y = 0

    while x < output_image.shape[1]: # Vertical lines
        cv2.line(output_image, (x, gap), (x, output_image.shape[0]+gap), color=(255, 255, 255), thickness=1)
        x += pxstep
    
    while y < output_image.shape[0]: # Horizontal lines
        cv2.line(output_image, (0, y+gap), (output_image.shape[1], y+gap), color=(255, 255, 255),thickness=1)
        y += pxstep

def pixels2realworld(self,trk,shapes):
    """
    """
    tracker = trk.kf.x.reshape(9)
    aux_points = np.array([[tracker[0]], [tracker[1]], [0.0], [1.0]]) # Tracker centroid in pixels
    aux_points = np.dot(self.tf_bevtl2bevcenter_px,aux_points)
    aux_centroid = np.dot(self.tf_lidar2bev,aux_points)
    real_world_x, real_world_y, real_world_w, real_world_l = compute_corners(tracker,shapes,aux_centroid)

    return real_world_x, real_world_y,real_world_w, real_world_l

def pixels2realworld_prediction(self,predicted_bb,shapes):
    """
    """
    aux_predicted_bb = np.zeros((9))
    aux_predicted_bb[:2] = predicted_bb[:2] # x y centroid
    aux_predicted_bb[2] = np.sqrt(predicted_bb[2]+predicted_bb[3]) # w
    aux_predicted_bb[3] = predicted_bb[2]/aux_predicted_bb[2]
    aux_predicted_bb[4] = predicted_bb[4]

    aux_points = np.array([[predicted_bb[0]], [predicted_bb[1]], [0.0], [1.0]]) # Tracker centroid in pixels
    aux_points = np.dot(self.tf_bevtl2bevcenter_px,aux_points)
    aux_centroid = np.dot(self.tf_lidar2bev,aux_points)
    real_world_x, real_world_y, real_world_w, real_world_l = compute_corners(predicted_bb,shapes,aux_centroid)

    return real_world_x, real_world_y,real_world_w, real_world_l


