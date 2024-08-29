#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
import os
import time
import threading
import subprocess
import rospy
import numpy as np
import copy
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from std_msgs.msg import UInt8, Float32, Bool, Empty, String
from sensor_msgs.msg import Image, PointCloud2
from nepi_ros_interfaces.msg import BoundingBox, BoundingBoxes

from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_img

NEPI_BASE_NAMESPACE = nepi_ros.get_base_namespace()

class ROSAiNodeIF:
    CHECK_RATE = 10
    img_acquire = False
    img_msg = None
    img_last_stamp = None
    img_lock = threading.Lock()
    depth_map_acquire = False
    depth_map_msg = None
    depth_map_last_stamp = None
    depth_map_lock = threading.Lock() 
    pc_acquire = False   
    pc_msg = None
    pc_last_stamp = None
    pc_lock = threading.Lock()

    def __init__( getBoundingBoxesDictFunction, classes_list, self,source_image_topic
                 ):
        node_name = rospy.get_name()
        rospy.loginfo(node_name + ": Starting AI Node IF setup")


        # getBoundingBoxesListFunction
        # Returns list of bounding box dictionaries for each detection, 
        #    returns empty list if no detections made, 
        #    ** then returns None after being called until next detection ***
        #    If new detection is made before last detection is pulled, the detection is just overwritten
        # Also returns a ros header pulled from image that was used in detection process that produced the bounding_boxes returned

        #bounding_box_dict = {
        #                    'Class': class_string_name
        #                    'id': int_index_in_classes_list
        #                    'uid': object_string_name default to class_id
        #                    'probability': float
        #                    'box': [xmin,ymin,xmax,ymax]
        #                    }

        # return bounding_box_list, image_header
        self.getBoundingBoxesDictFunction = getNewDetectionBoundingBoxesFunction 

        # Get classes_list is a string list of all classes provided by current AI model.  Then create colored bounding boxes
        self.classes_list = classes_list
        rgb_list = []
        if len(classes_list) > 0:
            cmap = plt.get_cmap('viridis')
            color_list = cmap(np.linspace(0, 1, len(self.current_classifier_classes))).tolist()
            for color in color_list:
                for i in range(3):
                    rgb.append(int(color[i]*255))
                rgb_list.append(rgb)
        self.class_color_list = rgb_list

        # Create AI Node Publishers and Subscribers
        self.source_image_topic = source_image_topic
        SOURCE_IMAGE_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/source_image"
        self.source_image_pub = rospy.Publisher(SOURCE_IMAGE_TOPIC, Image, self.depthMapCb, queue_size = 1)
        rospy.Subscriber(source_image_topic, Image, self.sourceImageCb, queue_size = 1)
        
       # Look for Depth Map
        source_depth_map_topic = source_image_topic.rsplit('/',1)[0] + "/depth_map"
        source_depth_map_topic = nepi_ros.find_topic(depth_map_topic)
        if source_depth_map_topic != '':  # create pubs and subs if topic found
            SOURCE_DEPTH_MAP_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/source_depth_map"
            self.source_depth_map_pub = rospy.Publisher(SOURCE_DEPTH_MAP_TOPIC, Image, self.depthMapCb, queue_size = 1)
            rospy.Subscriber(source_depth_map_topic, Image, self.sourceDepthMapCb, queue_size = 1)

        # Check for point_cloud
        source_pointcloud_topic = source_image_topic.rsplit('/',1)[0] + "/pointcloud"
        source_pointcloud_topic = nepi_ros.find_topic(source_pointcloud_topic)
        if source_pointcloud_topic != '':  # create pubs and subs if topic found
            SOURCE_POINTCLOUD_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/source_pointcloud"
            self.source_pointcloud_pub = rospy.Publisher(SOURCE_POINTCLOUD_TOPIC,PointCloud2,queue_size=1)
            rospy.Subscriber(source_pointcloud_topic, Pointcloud2, self.sourcePointcloudCb, queue_size = 1)

        FOUND_OBJECT_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/found_object"
        self.found_object_pub = rospy.Publisher(FOUND_OBJECT_TOPIC, ObjectCount,  queue_size = 1)

        BOUNDING_BOXES_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/bounding_boxes"
        self.bounding_boxes_pub = rospy.Publisher(BOUNDING_BOXES_TOPIC, BoundingBoxes, queue_size = 1)

        #DETECTION_IMAGE_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/detection_image"
        #self.detection_image_pub = rospy.Publisher(DETECTION_IMAGE_TOPIC, Image,  queue_size = 1)

        # Create AI Node Processes       
        check_interval_sec = float(1)/self.CHECK_RATE
        rospy.Timer(rospy.Duration(check_interval_sec), self.detectionCheckCb)

        rospy.loginfo(node_name + ":Initialization Complete")

    ###############################################################################################
  
      # callback to get color 2d image data
    def sourceImageCb(self, image_msg):
      self.img_lock.acquire()
      self.img_msg = image_msg
      self.img_lock.release()

    # callback to get depthmap
    def sourceDepthMapCb(self, image_msg):
      image_msg.header.stamp = rospy.Time.now()
      self.depth_map_lock.acquire()
      self.depth_map_msg = image_msg
      self.depth_map_lock.release()

    # callback to get and republish point_cloud
    def sourcePointcloudCb(self, pointcloud_msg):
      self.pc_lock.acquire()
      self.pc_msg = pointcloud_msg
      self.pc_lock.release()


    def detectionCheckCb(self,timer):
        bounding_boxes_list, image_header = self.getNewDetectionBoundingBoxesFunction()
        process_time = ros.Time.now()
        if bounding_boxes_list is not None:
            num_bounding_box = len(bounding_boxes_list)
           
            if self.source_image_pub.getNumSubscribers() > 0:
                self.img_lock.acquire()
                img_msg = None
                if self.img_msg != None:
                if self.img_msg.header.stamp != self.img_last_stamp:
                    img_msg = copy.deepcopy(self.img_msg)         
                    self.img_msg = None
                self.img_lock.release()
                if img_msg is not None and not rospy.is_shutdown():
                    self.source_image_pub.publish(img_msg)

            if self.source_depth_map_pub.getNumSubscribers() > 0:
                self.depth_map_lock.acquire()
                img_msg = None
                if self.depth_map_msg != None:
                if self.depth_map_msg.header.stamp != self.depth_map_last_stamp:
                    img_msg = copy.deepcopy(self.depth_map_msg)
                    self.depth_map_msg = None
                self.depth_map_lock.release()
                if depth_map_msg is not None and not rospy.is_shutdown():
                    self.source_depth_map_pub.publish(depth_map_msg)

            if self.source_pointcloud_pub.getNumSubscribers() > 0:
                self.pc_lock.acquire()
                pc_msg = None
                if self.pc_msg != None:
                if self.pc_msg.header.stamp != self.pc_last_stamp:
                pc_msg = copy.deepcopy(self.pc_msg)
                self.pc_msg = None
                self.pc_lock.release()
                if pointcloud_msg is not None and not rospy.is_shutdown():
                    self.source_pointcloud_pub.publish(pc_msg)

            if self.found_object_pub.getNumSubscribers() > 0:
                found_object_msg = ObjectCount()
                found_object_msg.header.stamp = process_time
                found_object_msg.count = num_bounding_boxes
                if not rospy.is_shutdown():
                    self.found_object_pub.publish(found_object_msg)

            if self.bounding_boxes_pub.getNumSubscribers() > 0:
                if num_bounding_boxes > 0:
                    bounding_box_msg_list = []
                    for bb in bounding_box_list:
                        bounding_box_msg = BoundingBox()
                        bounding_box_msg.Class = bb['Class']
                        bounding_box_msg.id = bb['id']
                        bounding_box_msg.uid = bb['uid']
                        bounding_box_msg.probability = bb['probability']
                        bounding_box_msg.xmin = bb['box'][0]
                        bounding_box_msg.ymin = bb['box'][1]
                        bounding_box_msg.xmax = bb['box'][2]
                        bounding_box_msg.ymax = bb['box'][3]
                        bounding_box_msg_list.append(bounding_box_msg)
                    bounding_boxes_msg = BoundingBoxes()
                    bounding_boxes_msg.header.stamp = process_time
                    bounding_boxes_msg.image_header = image_header
                    bounding_boxes_msg.image_topic = self.source_image_topic
                    bounding_boxes_msg.bounding_boxes = bounding_box_msg_list
                

