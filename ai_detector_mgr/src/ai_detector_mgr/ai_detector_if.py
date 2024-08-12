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
# ROS namespace setup
NEPI_BASE_NAMESPACE = '/nepi/s2x/'
os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1]
import rospy



import time
import sys
import numpy as np
import cv2

from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_pc 
from nepi_edge_sdk_base import nepi_img 

from std_msgs.msg import UInt8, Int32, Float32, Empty, String, Bool
from sensor_msgs.msg import Image, PointCloud2
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge
from nepi_ros_interfaces.msg import AiAppStatus, BoundingBox, BoundingBoxes, BoundingBox3D, BoundingBoxes3D, \
                                    ObjectCount, ClassifierSelection, \
                                    AddAiDetectorClass, StringArray, TargetLocalization, TargetLocalizationScene
from nepi_ros_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryRequest

from nepi_edge_sdk_base.save_data_if import SaveDataIF
from nepi_edge_sdk_base.save_cfg_if import SaveCfgIF




#########################################

#########################################
# Node Class
#########################################

class NepiAiDetectorApp(object):

  #Set Initial Values
  FACTORY_FOV_VERT_DEG=70 # Camera Vertical Field of View (FOV)
  FACTORY_FOV_HORZ_DEG=110 # Camera Horizontal Field of View (FOV)
  FACTORY_TARGET_BOX_REDUCTION_PERCENT=50 # Sets the percent of target box around center to use for range calc
  FACTORY_TARGET_DEPTH_METERS=0.3 # Sets the depth filter around mean depth to use for range calc
  FACTORY_TARGET_MIN_POINTS=10 # Sets the minimum number of valid points to consider for a valid range
  FACTORY_TARGET_MAX_AGE_SEC=10 # Remove lost targets from dictionary if older than this age

  NONE_CLASSES_DICT = dict()
  NONE_CLASSES_DICT["None"] = {'depth': FACTORY_TARGET_DEPTH_METERS}

  EMPTY_TARGET_DICT = dict()
  EMPTY_TARGET_DICT["None"] = {'class': 'None', 
                                    'depth': FACTORY_TARGET_DEPTH_METERS,
                                    'last_center_px': [0,0],
                                    'last_velocity_pxps': [0,0],
                                    'last_center_m': [0,0,0],
                                    'last_velocity_mps': [0,0,0],
                                    'last_detection_timestamp': 0                              
                                    }

  data_products = ["image"]
  
  current_classifier = "None"
  current_classifier_state = "None"
  current_classifier_classes = "['None']"

  current_image_topic = "None"
  img_width = 0
  img_height = 0
  image_sub = None
  depth_map_topic = ""
  depth_map_sub = None
  np_depth_array_m = None
  pointcloud_topic = ""
  pointcloud_sub = None

  detect_boxes=None
  bounding_box_msg = None
  bounding_box3d_msg = None
  bounding_box3d_pub = None
  targeting_data_pub = None

  last_image_topic = None
  
  selected_tracking_list = []
  current_targets_dict = EMPTY_TARGET_DICT
  lost_targets_dict = EMPTY_TARGET_DICT

  targeting_class_list = []
  targeting_target_list = []

###################
## App Callbacks

  def resetAppCb(self,msg):
    self.resetApp()

  def resetApp(self):
    rospy.set_param('~last_classifier', "")
    rospy.set_param('~selected_classes_dict', self.NONE_CLASSES_DICT)
    rospy.set_param('~image_fov_vert',  self.FACTORY_FOV_VERT_DEG)
    rospy.set_param('~image_fov_horz', self.FACTORY_FOV_HORZ_DEG)
    rospy.set_param('~target_box_reduction',  self.FACTORY_TARGET_BOX_REDUCTION_PERCENT)
    rospy.set_param('~default_target_depth',  self.FACTORY_TARGET_DEPTH_METERS)
    rospy.set_param('~target_min_points', self.FACTORY_TARGET_MIN_POINTS)
    rospy.set_param('~target_age_filter', self.FACTORY_TARGET_MAX_AGE_SEC)
    self.current_targets_dict = EMPTY_TARGET_DICT
    self.lost_targets_dict = EMPTY_TARGET_DICT
    self.publish_status()

  ###################
  ## Selection Callbacks

  def addClassCb(self,msg):
    ##rospy.loginfo(msg)
    class_name = msg.class_name
    class_depth_m = msg.class_depth_m
    if class_name == msg.class_name:
      selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
      selected_classes_dict[class_name] = {'depth': class_depth_m}
      rospy.set_param('~selected_classes_dict', selected_classes_dict)
    self.publish_status()

  def removeClassCb(self,msg):
    ##rospy.loginfo(msg)
    class_name = msg.data
    selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
    if class_name in selected_classes_dict.keys():
      del selected_classes_dict[class_name]
      rospy.set_param('~selected_classes_dict', selected_classes_dict)
    self.publish_status()

  def setVertFovCb(self,msg):
    ##rospy.loginfo(msg)
    fov = msg.data
    if fov > 0:
      rospy.set_param('~image_fov_vert',  fov)
    self.publish_status()


  def setHorzFovCb(self,msg):
    ##rospy.loginfo(msg)
    fov = msg.data
    if fov > 0:
      rospy.set_param('~image_fov_horz',  fov)
    self.publish_status()
    
  def setBoxReductionCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      rospy.set_param('~target_box_reduction',val)
    self.publish_status()   
      
  def setDefaultTargetDepthCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0:
      rospy.set_param('~default_target_depth',val)
    self.publish_status()   

  def setTargetMinPointsCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0:
      rospy.set_param('~target_min_points',val)
    self.publish_status() 

  def setAgeFilterCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0:
      rospy.set_param('~target_age_filter',val)
    self.publish_status()


  #######################
  ### Node Initialization
  def __init__(self):
   
    rospy.loginfo("AI_APP: Starting Initialization Processes")
    self.initParamServerValues(do_updates = False)
    self.resetParamServer(do_updates = False)
   

    # Set up save data and save config services ########################################################
    self.save_data_if = SaveDataIF(data_product_names = self.data_products)
    # Temp Fix until added as NEPI ROS Node
    self.save_cfg_if = SaveCfgIF(updateParamsCallback=self.initParamServerValues, 
                                 paramsModifiedCallback=self.updateFromParamServer)


    ## App Setup ########################################################
    app_reset_app_sub = rospy.Subscriber('~reset_app', Empty, self.resetAppCb, queue_size = 10)
    self.initParamServerValues(do_updates=False)

    add_class_sub = rospy.Subscriber('~add_class', AddAiDetectorClass, self.addClassCb, queue_size = 10)
    remove_class_sub = rospy.Subscriber('~remove_class', String, self.removeClassCb, queue_size = 10)
    vert_fov_sub = rospy.Subscriber("~set_image_fov_vert", Float32, self.setVertFovCb, queue_size = 10)
    horz_fov_sub = rospy.Subscriber("~set_image_fov_horz", Float32, self.setHorzFovCb, queue_size = 10)
    box_reduction_sub = rospy.Subscriber("~set_box_reduction_percent", Float32, self.setBoxReductionCb, queue_size = 10)
    default_target_depth_sub = rospy.Subscriber("~set_default_target_detpth", Float32, self.setDefaultTargetDepthCb, queue_size = 10)
    target_min_points_sub = rospy.Subscriber("~set_target_min_points", Int32, self.setTargetMinPointsCb, queue_size = 10)
    age_filter_sub = rospy.Subscriber("~set_age_filter", Float32, self.setAgeFilterCb, queue_size = 10)

    # Start an AI manager status monitoring thread
    AI_MGR_STATUS_SERVICE_NAME = NEPI_BASE_NAMESPACE + "ai_detector_mgr/img_classifier_status_query"
    self.get_ai_mgr_status_service = rospy.ServiceProxy(AI_MGR_STATUS_SERVICE_NAME, ImageClassifierStatusQuery)
    time.sleep(1)
    rospy.Timer(rospy.Duration(1), self.getAiMgrStatusCb)

    # Start AI Manager Subscribers
    FOUND_OBJECT_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/found_object"
    rospy.Subscriber(FOUND_OBJECT_TOPIC, ObjectCount, self.found_object_callback, queue_size = 1)
    BOUNDING_BOXES_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/bounding_boxes"
    rospy.Subscriber(BOUNDING_BOXES_TOPIC, BoundingBoxes, self.object_detected_callback, queue_size = 1)

    # Setup Node Publishers
    self.status_pub = rospy.Publisher("~status", AiAppStatus, queue_size=1, latch=True)
    self.image_pub = rospy.Publisher("~image",Image,queue_size=1)
    self.found_object_pub = rospy.Publisher("~found_object", ObjectCount, queue_size=1)
    self.bounding_boxes_pub = rospy.Publisher("~bounding_boxes", BoundingBoxes, queue_size=1)
    self.bounding_boxes_3d_pub = rospy.Publisher("~bounding_boxes_3D", BoundingBoxes3D, queue_size=1)
    self.target_localizations_pub = rospy.Publisher("~target_data", TargetLocalizationScene, queue_size=1)
    time.sleep(1)
    ## Initiation Complete
    rospy.loginfo("AI_APP: Initialization Complete")
    self.publish_status()


  #######################
  ### AI Magnager Callbacks

  def getAiMgrStatusCb(self,timer):
    try:
      ai_mgr_status_response = self.get_ai_mgr_status_service()
    except Exception as e:
      rospy.loginfo("AI_APP: Failed to call AI MGR STATUS service " + str(e))
      return
    self.current_classifier = ai_mgr_status_response.selected_classifier
    self.current_classifier_state = ai_mgr_status_response.classifier_state
    classes_string = ai_mgr_status_response.selected_classifier_classes
    self.current_classifier_classes = nepi_ros.parse_string_list_msg_data(classes_string)
    self.current_image_topic = ai_mgr_status_response.selected_img_topic

    last_classifier = rospy.get_param('~last_classiier', self.init_last_classifier)
    if last_classifier != self.current_classifier and self.current_classifier != "None":
      classes_dict = dict()  # Reset classes to all on new classifier
      selected_classes_dict = dict()
      for target_class in self.current_classifier_classes:
        selected_classes_dict[target_class] = {'depth': self.FACTORY_TARGET_DEPTH_METERS}
      #rospy.logwarn(selected_classes_dict)
      rospy.set_param('~selected_classes_dict', selected_classes_dict)
      rospy.set_param('~last_classiier', self.current_classifier)



    if self.last_image_topic != self.current_image_topic and self.current_image_topic != "None":
      if self.image_sub != None:
        self.image_sub.Unregister()
        time.sleep(1)
      self.image_sub = rospy.Subscriber(self.current_image_topic, Image, self.imageCb, queue_size = 10)
      self.last_image_topic = self.current_image_topic

      depth_map_topic = self.current_image_topic.rsplit('/',1)[0] + "/depth_map"
      depth_map_topic = nepi_ros.find_topic(depth_map_topic)
      self.depth_map_topic = depth_map_topic
      #rospy.logwarn(depth_map_topic)
      if depth_map_topic != "":
        if self.depth_map_sub != None:
          self.depth_map_sub.Unregister()
          time.sleep(1)
        self.depth_map_sub = rospy.Subscriber(depth_map_topic, Image, self.depthMapCb, queue_size = 10)

      pointcloud_topic = self.current_image_topic.rsplit('/',1)[0] + "/pointcloud"
      pointcloud_topic = nepi_ros.find_topic(pointcloud_topic)
      self.pointcloud_topic = pointcloud_topic
      if pointcloud_topic != "":
        if self.pointcloud_sub != None:
          self.pointcloud_sub.Unregister()
          time.sleep(1)
        self.pointcloud_sub = rospy.Subscriber(pointcloud_topic, PointCloud2, self.pointcloudCb, queue_size = 10)


  ### Monitor Output of AI model to clear detection status
  def found_object_callback(self,found_obj_msg):
    self.found_object_pub.publish(found_obj_msg)
    # Must reset target lists if no targets are detected
    if found_obj_msg.count == 0:
      #print("No objects detected")
      self.detect_boxes=None


  ### If object(s) detected, save bounding box info to global
  def object_detected_callback(self,bounding_box_msg):
    self.bounding_boxes_pub.publish(bounding_box_msg)
    #print("Objects detected, passsing to targeting process")
    self.detect_boxes=bounding_box_msg

  def imageCb(self,img_msg):
    ros_timestamp = img_msg.header.stamp
    # Convert ROS image to OpenCV for editing
    self.img_height = img_msg.height
    self.img_width = img_msg.width
    cv2_bridge = CvBridge()
    cv2_img = cv2_bridge.imgmsg_to_cv2(img_msg, "bgr8")
    tls = []
    bbs3d=[]
    selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
    # Iterate over all of the objects and calculate range and bearing data
    image_fov_vert = rospy.get_param('~image_fov_vert',  self.init_image_fov_vert)
    image_fov_horz = rospy.get_param('~image_fov_horz', self.init_image_fov_horz)
    target_reduction_percent = rospy.get_param('~target_box_reduction',  self.init_target_box_reduction)
    target_min_points = rospy.get_param('~target_min_points',  self.init_target_min_points)
    if self.detect_boxes is not None:
      for box in self.detect_boxes.bounding_boxes:
        if box.Class in selected_classes_dict.keys():
          target_depth_m = selected_classes_dict[box.Class]
          # Get target label
          target_label=box.Class
          # reduce target box based on user settings
          box_reduction_y_pix=int(float((box.ymax - box.ymin))*float(target_reduction_percent )/100/2)
          box_reduction_x_pix=int(float((box.xmax - box.xmin))*float(target_reduction_percent )/100/2)
          ymin_adj=int(box.ymin + box_reduction_y_pix)
          ymax_adj=int(box.ymax - box_reduction_y_pix)
          xmin_adj=box.xmin + box_reduction_x_pix
          xmax_adj=box.xmax - box_reduction_x_pix
          # Calculate target range
          if self.np_depth_array_m is not None:
            # Get target range from cropped and filtered depth data
            depth_box_adj= self.np_depth_array_m[ymin_adj:ymax_adj,xmin_adj:xmax_adj]
            depth_mean_val=np.mean(depth_box_adj)
            depth_array=depth_box_adj.flatten()
            min_filter=depth_mean_val-target_depth_m/2
            max_filter=depth_mean_val+target_depth_m/2
            depth_array=depth_array[depth_array > min_filter]
            depth_array=depth_array[depth_array < max_filter]
            depth_len=len(depth_array)
            if depth_len > target_min_points:
              target_range_m=np.mean(depth_box_adj)
            else:
              target_range_m=float(-999) # NEPI standard unset value
          else:
            target_range_m=float(-999)  # NEPI standard unset value
          # Calculate target bearings
          object_loc_y_pix = float(box.ymin + ((box.ymax - box.ymin))  / 2) 
          object_loc_x_pix = float(box.xmin + ((box.xmax - box.xmin))  / 2)
          object_loc_y_ratio_from_center = float(object_loc_y_pix - self.img_height/2) / float(self.img_height/2)
          object_loc_x_ratio_from_center = float(object_loc_x_pix - self.img_width/2) / float(self.img_width/2)
          target_vert_angle_deg = -(object_loc_y_ratio_from_center * float(image_fov_vert/2))
          target_horz_angle_deg = (object_loc_x_ratio_from_center * float(image_fov_horz/2))
          ### Print the range and bearings for each detected object
    ##      print(target_label)
    ##      print(str(depth_box_adj.shape) + " detection box size")
    ##      print(str(depth_len) + " valid depth readings")
    ##      print("%.2f" % target_range_m + "m : " + "%.2f" % target_horz_angle_deg + "d : " + "%.2f" % target_vert_angle_deg + "d : ")
    ##      print("")
          # Add Targeting_Data
          target_data_msg=TargetLocalization()
          tracking_id = -999 # Need to add unque id tracking
          target_data_msg.name=target_label
          target_data_msg.range_m=target_range_m
          target_data_msg.azimuth_deg=target_horz_angle_deg
          target_data_msg.elevation_deg=target_vert_angle_deg
          tls.append(target_data_msg)
    # To Do Add Bounding Box 3D Data
    # bounding_box_3D = BoundingBox3D()
        
          ###### Apply Image Overlays and Publish Targeting_Image ROS Message
          # Overlay adjusted detection boxes on image 
          start_point = (xmin_adj, ymin_adj)
          end_point = (xmax_adj, ymax_adj)
          cv2.rectangle(cv2_img, start_point, end_point, color=(255,0,0), thickness=2)
          # Overlay text data on OpenCV image
          font                   = cv2.FONT_HERSHEY_SIMPLEX
          fontScale              = 0.5
          fontColor              = (0, 255, 0)
          thickness              = 1
          lineType               = 1
        # Overlay Label
          text2overlay=box.Class
          bottomLeftCornerOfText = (int(object_loc_x_pix),int(object_loc_y_pix))
          cv2.putText(cv2_img,text2overlay, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
          # Overlay Data
          if target_range_m == -999:
            t_range_m = np.nan
          else:
            t_range_m = target_range_m
          text2overlay="%.1f" % t_range_m + "m," + "%.f" % target_horz_angle_deg + "d," + "%.f" % target_vert_angle_deg + "d"
          bottomLeftCornerOfText = (int(object_loc_x_pix),int(object_loc_y_pix)+15)
          cv2.putText(cv2_img,text2overlay, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    target_loc_scene_msg = TargetLocalizationScene()
    target_loc_scene_msg.targets = tls
    if not rospy.is_shutdown():
      self.target_localizations_pub.publish(target_loc_scene_msg)
    #Convert OpenCV image to ROS image
    img_out_msg = cv2_bridge.cv2_to_imgmsg(cv2_img,"bgr8")#desired_encoding='passthrough')
    # Publish new image to ros
    if self.image_pub is not None and not rospy.is_shutdown():
      self.image_pub.publish(img_out_msg)
    img_saving_is_enabled = self.save_data_if.data_product_saving_enabled('image')
    img_snapshot_enabled = self.save_data_if.data_product_snapshot_enabled('image')
    if img_saving_is_enabled is True or img_snapshot_enabled is True:
      self.save_img2file('image',cv2_img,ros_timestamp)


  def depthMapCb(self,depth_map_msg):
    # Zed depth data is floats in m, but passed as 4 bytes each that must be converted to floats
    # Use cv2_bridge() to convert the ROS image to OpenCV format
    #Convert the depth 4xbyte data to global float meter array
    cv2_bridge = CvBridge()
    cv2_depth_image = cv2_bridge.imgmsg_to_cv2(depth_map_msg, desired_encoding="passthrough")
    self.np_depth_array_m = (np.array(cv2_depth_image, dtype=np.float32)) # replace nan values
    self.np_depth_array_m[np.isnan(self.np_depth_array_m)] = 0 # zero pixels with no value
    self.np_depth_array_m[np.isinf(self.np_depth_array_m)] = 0 # zero pixels with inf value

  def pointcloudCb(self,pointcloud2_msg):
    pass # Need to implement point cloud bounding box filter and publishers for each unique target

  #######################
  ### Config Functions

  def saveConfigCb(self, msg):  # Just update class init values. Saving done by Config IF system
    pass # Left empty for sim, Should update from param server

  def setCurrentAsDefault(self):
    self.initParamServerValues(do_updates = False)

  def updateFromParamServer(self):
    #rospy.logwarn("Debugging: param_dict = " + str(param_dict))
    #Run any functions that need updating on value change
    # Don't need to run any additional functions
    pass

  def initParamServerValues(self,do_updates = True):
      rospy.loginfo("AI_APP: Setting init values to param values")
      self.init_last_classifier = rospy.get_param("~last_classifier", "")
      self.init_selected_classes_dict = rospy.get_param('~selected_classes_dict', self.NONE_CLASSES_DICT)
      self.init_image_fov_vert = rospy.get_param('~image_fov_vert',  self.FACTORY_FOV_VERT_DEG)
      self.init_image_fov_horz = rospy.get_param('~image_fov_horz', self.FACTORY_FOV_HORZ_DEG)
      self.init_target_box_reduction = rospy.get_param('~target_box_reduction',  self.FACTORY_TARGET_BOX_REDUCTION_PERCENT)
      self.init_default_target_depth = rospy.get_param('~default_target_depth',  self.FACTORY_TARGET_DEPTH_METERS)
      self.init_target_min_points = rospy.get_param('~target_min_points', self.FACTORY_TARGET_MIN_POINTS)
      self.init_target_age_filter = rospy.get_param('~target_age_filter', self.FACTORY_TARGET_MAX_AGE_SEC)
      self.resetParamServer(do_updates)

  def resetParamServer(self,do_updates = True):
      rospy.set_param('~last_classiier', self.init_last_classifier)
      rospy.set_param('~selected_classes_dict', self.init_selected_classes_dict)
      rospy.set_param('~image_fov_vert',  self.init_image_fov_vert)
      rospy.set_param('~image_fov_horz', self.init_image_fov_horz)
      rospy.set_param('~target_box_reduction',  self.init_target_box_reduction)
      rospy.set_param('~default_target_depth',  self.init_default_target_depth)
      rospy.set_param('~target_min_points', self.init_target_min_points)
      rospy.set_param('~target_age_filter', self.init_target_age_filter)
      if do_updates:
          self.updateFromParamServer()
          self.publish_status()



  ###################
  ## Status Publisher
  def publish_status(self):
    status_msg = AiAppStatus()

    status_msg.classifier_name = self.current_classifier
    status_msg.classifier_state = self.current_classifier_state
    status_msg.available_classes_list = str(self.current_classifier_classes)

    status_msg.image_topic = self.current_image_topic
    status_msg.depth_map_topic = self.depth_map_topic
    status_msg.pointcloud_topic = self.pointcloud_topic

    selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
    classes_list = []
    depth_list = []
    for key in selected_classes_dict.keys():
      classes_list.append(key)
      depth_list.append(selected_classes_dict[key]['depth'])
    status_msg.selected_classes_list = str(classes_list)
    status_msg.selected_classes_depth_list = str(depth_list)

    status_msg.image_fov_vert_degs = rospy.get_param('~image_fov_vert',  self.init_image_fov_vert)
    status_msg.image_fov_horz_degs = rospy.get_param('~image_fov_horz', self.init_image_fov_horz)

    status_msg.target_box_reduction_percent = rospy.get_param('~target_box_reduction',  self.init_target_box_reduction)
    status_msg.default_target_depth_m = rospy.get_param('~default_target_depth',  self.init_default_target_depth)
    status_msg.target_min_points = rospy.get_param('~target_min_points', self.init_target_min_points)
    status_msg.target_age_filter = rospy.get_param('~target_age_filter', self.init_target_age_filter)

    self.status_pub.publish(status_msg)


    # ToDo: Move into dedicated publishers
    '''
    current_targets_list = []
    for key in self.current_targets_dict:
      if key != 'None':
        center_px = self.current_targets_dict[key]['last_center_px']
        center_m = self.current_targets_dict[key]['last_center_m']
        target_entry = [key,str(center_px),str(center_m)]
        current_targets_list.append(target_entry)
    status_msg.current_targets_list = str(current_targets_list)

    lost_targets_list = []
    for key in self.current_targets_dict:
      if key != 'None':
        center_px = self.current_targets_dict[key]['last_center_px']
        center_m = self.current_targets_dict[key]['last_center_m']
        target_entry = [key,str(center_px),str(center_m)]
        lost_targets_list.append(target_entry)
    status_msg.lost_targets_list = str(lost_targets_list)
    '''

 
      
    
  #######################
  # Data Saving Funcitons
 

  def save_img2file(self,data_product,cv2_img,ros_timestamp):
      if self.save_data_if is not None:
          saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
          snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
          # Save data if enabled
          if saving_is_enabled or snapshot_enabled:
              if cv2_img is not None:
                  if (self.save_data_if.data_product_should_save(data_product) or snapshot_enabled):
                      full_path_filename = self.save_data_if.get_full_path_filename(nepi_ros.get_datetime_str_from_stamp(ros_timestamp), 
                                                                                              "pointcloud_app-" + data_product, 'png')
                      if os.path.isfile(full_path_filename) is False:
                          cv2.imwrite(full_path_filename, cv2_img)
                          self.save_data_if.data_product_snapshot_reset(data_product)

  def save_pc2file(self,data_product,o3d_pc,ros_timestamp):
      if self.save_data_if is not None:
          saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
          snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
          # Save data if enabled
          if saving_is_enabled or snapshot_enabled:
              if o3d_pc is not None:
                  if (self.save_data_if.data_product_should_save(data_product) or snapshot_enabled):
                      full_path_filename = self.save_data_if.get_full_path_filename(nepi_ros.get_datetime_str_from_stamp(ros_timestamp), 
                                                                                              "pointcloud_app-" + data_product, 'pcd')
                      if os.path.isfile(full_path_filename) is False:
                          nepi_pc.save_pointcloud(o3d_pc,full_path_filename)
                          self.save_data_if.data_product_snapshot_reset(data_product)

                
    
  #######################
  # Node Cleanup Function
  
  def cleanup_actions(self):
    rospy.loginfo("AI_APP: Shutting down: Executing script cleanup actions")


#########################################
# Main
#########################################
if __name__ == '__main__':
  node_name = "app_ai_detector"
  rospy.init_node(name=node_name)
  #Launch the node
  rospy.loginfo("AI_APP: Launching node named: " + node_name)
  node = NepiAiDetectorApp()
  #Set up node shutdown
  rospy.on_shutdown(node.cleanup_actions)
  # Spin forever (until object is detected)
  rospy.spin()





