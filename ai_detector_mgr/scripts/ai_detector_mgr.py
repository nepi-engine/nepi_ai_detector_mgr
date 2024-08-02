#!/usr/bin/env python

import os
import os.path
import errno
import glob
import subprocess
import yaml

import rospy

from std_msgs.msg import Empty, Float32
from nepi_ros_interfaces.srv import ImageClassifierListQuery, ImageClassifierListQueryResponse
from nepi_ros_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryResponse, SystemStorageFolderQuery
from nepi_ros_interfaces.msg import ClassifierSelection
from darknet_ros_msgs.msg import ObjectCount
from darknet_ros_msgs.msg import BoundingBoxes

from nepi_edge_sdk_base.save_cfg_if import SaveCfgIF
from nepi_edge_sdk_base.save_data_if import SaveDataIF



class AIDetectorManager:
    NODE_NAME = "ai_detector_mgr"
    DARKNET_CFG_PATH = '/mnt/nepi_storage/ai_models/darknet_ros/'
    MIN_THRESHOLD = 0.001
    MAX_THRESHOLD = 1.0
    FIXED_LOADING_START_UP_TIME_S = 5.0 # Total guess
    ESTIMATED_WEIGHT_LOAD_RATE_BYTES_PER_SECOND = 16000000.0 # Very roughly empirical based on YOLOv3

    darknet_cfg_files = []
    classifier_list = []
    classifier_weights_sizes = {}

    current_classifier = "None"
    current_img_topic = "None"
    current_threshold = 0.3
    classifier_state = ImageClassifierStatusQueryResponse.CLASSIFIER_STATE_STOPPED
    classifier_load_start_time = None

    darknet_ros_process = None

    found_object_sub = None

    def provide_classifier_list(self, req):
        return ImageClassifierListQueryResponse(self.classifier_list)

    def provide_classifier_status(self, req):
        # Update the loading progress if necessary
        loading_progress = 0.0
        if (self.classifier_state == ImageClassifierStatusQueryResponse.CLASSIFIER_STATE_RUNNING):
            loading_progress = 1.0
        elif (self.classifier_state == ImageClassifierStatusQueryResponse.CLASSIFIER_STATE_LOADING):
            loading_elapsed_s = (rospy.Time.now() - self.classifier_load_start_time).to_sec()
            estimated_load_time_s = self.FIXED_LOADING_START_UP_TIME_S + (self.classifier_weights_sizes[self.current_classifier] / self.ESTIMATED_WEIGHT_LOAD_RATE_BYTES_PER_SECOND)
            if loading_elapsed_s > estimated_load_time_s:
                loading_progress = 0.95 # Stall at 95%
            else:
                loading_progress = loading_elapsed_s / estimated_load_time_s
    
        return [self.current_img_topic, self.current_classifier, self.classifier_state, loading_progress, self.current_threshold]

    def stop_classifier(self):
        rospy.loginfo("Stopping classifier")
        if not (None == self.found_object_sub):
            self.found_object_sub.unregister()
        if not (None == self.darknet_ros_process):
            self.darknet_ros_process.terminate()
            self.darknet_ros_process = None
        self.classifier_state = ImageClassifierStatusQueryResponse.CLASSIFIER_STATE_STOPPED
        self.current_classifier = "None"
        self.current_img_topic = "None"
        #self.current_threshold = 0.0

    def stop_classifier_cb(self, msg):
        self.stop_classifier()

    def start_classifier_cb(self, classifier_selection_msg):
        self.start_classifier(classifier=classifier_selection_msg.classifier, input_img=classifier_selection_msg.img_topic, threshold=classifier_selection_msg.detection_threshold)

    def set_threshold_cb(self, msg):
        # All we do here is update the current_threshold so that it is up-to-date in status responses
        # and will be saved properly in the config file (on request).
        if (msg.data >= self.MIN_THRESHOLD and msg.data <= self.MAX_THRESHOLD):
            self.current_threshold = msg.data

    def start_classifier(self, classifier, input_img, threshold):
        # First, validate the inputs
        # Check that the requested topic exists and has the expected type
        all_topics = rospy.get_published_topics()
        found_topic = False
        for t in all_topics:
            if (t[0] == input_img) and (t[1] == 'sensor_msgs/Image'):
                found_topic = True
                break

        if (False == found_topic):
            rospy.logerr("Topic %s is not a valid image topic -- not starting classifier", input_img)
            return

        # Check that the requested classifier exists
        if not (classifier in self.classifier_list):
            rospy.logerr("Unknown classifier requested: %s", classifier)
            return

        # Validate the requested_detection threshold
        if (threshold < self.MIN_THRESHOLD or threshold > self.MAX_THRESHOLD):
            rospy.logerr("Requested detection threshold (%f) out of range (0.001 - 1.0)", threshold)
            return

        # Stop the current classifier if it is running
        self.stop_classifier()

        # Build up the arg string for the new classifier launch command
        launch_cmd_line = [
            "roslaunch", "nepi_darknet_ros", "darknet_ros.launch",
            "namespace:=" + rospy.get_namespace(), 
            "yolo_weights_path:=" + os.path.join(self.DARKNET_CFG_PATH, "yolo_network_config/weights"),
            "yolo_config_path:=" + os.path.join(self.DARKNET_CFG_PATH, "yolo_network_config/cfg"),
            "ros_param_file:=" + os.path.join(self.DARKNET_CFG_PATH, "config/ros.yaml"),
            "network_param_file:=" + os.path.join(self.DARKNET_CFG_PATH, "config", classifier + ".yaml"),
            "input_img:=" + input_img,
            "detection_threshold:=" + str(threshold)
        ]

        rospy.loginfo("Launching Darknet ROS Process: " + str(launch_cmd_line))
        self.darknet_ros_process = subprocess.Popen(launch_cmd_line)

        # Update our local status
        self.current_classifier = classifier
        self.current_img_topic = input_img
        self.current_threshold = threshold
        self.classifier_state = ImageClassifierStatusQueryResponse.CLASSIFIER_STATE_LOADING
        self.classifier_load_start_time = rospy.Time.now()

        # Resubscribe to found_object so that we know when the classifier is up and running again
        self.found_object_sub = rospy.Subscriber('classifier/found_object', ObjectCount, self.got_darknet_update)

    def got_darknet_update(self, msg):
        # Means that darknet is up and running
        self.classifier_state = ImageClassifierStatusQueryResponse.CLASSIFIER_STATE_RUNNING
        self.found_object_sub.unregister()

    def setCurrentSettingsAsDefault(self):
        rospy.set_param('~default_classifier', self.current_classifier)

        rospy.set_param('~default_image', self.current_img_topic)

        rospy.set_param('~default_threshold', self.current_threshold)

    def boundingBoxesCallback(self, msg):
        # Nothing to do if it isn't time to save yet.
        if not self.save_data_if.data_product_should_save('detection_bounding_boxes'):
            return

        # TODO: Should use one of the timestamps from the message (from "header" or "image_header") rather than constructing a new one here for the filename
        full_path_filename = self.save_data_if.get_full_path_filename(self.save_data_if.get_timestamp_string(), 'classifier_bounding_boxes', 'txt')

        with open(full_path_filename, 'w') as f:
            f.write('input_image: ' + self.current_img_topic + '\n')
            f.write(str(msg))

    def updateFromParamServer(self):
        try:
            default_classifier = rospy.get_param('~default_classifier')
            default_img_topic = rospy.get_param('~default_image')
            default_threshold = rospy.get_param('~default_threshold')
            if default_classifier != "None" and default_img_topic != "None":
                rospy.loginfo("Classifier sleeping for 5 seconds to allow cameras to start")
                rospy.sleep(5) # Let the cameras start up properly
                rospy.loginfo('Starting classifier with parameters [' + default_classifier + ', ' + default_img_topic + ', ' + str(default_threshold) + ']')
                self.start_classifier(default_classifier, default_img_topic, default_threshold)
        except KeyError:
            rospy.loginfo("Classifier unable to find default parameters... starting up with no classifier running")

    def __init__(self):

        # Try to obtain the path to Darknet models from the system_mgr
        try:
            rospy.wait_for_service('system_storage_folder_query', 10.0)
            system_storage_folder_query = rospy.ServiceProxy('system_storage_folder_query', SystemStorageFolderQuery)
            self.DARKNET_CFG_PATH = os.path.join(system_storage_folder_query('ai_models').folder_path, 'darknet_ros')
        except Exception as e:
            rospy.logwarn("Failed to obtain system ai_models/darknet_ros folder... falling back to " + self.DARKNET_CFG_PATH)
                
        darknet_cfg_path_config_folder = os.path.join(self.DARKNET_CFG_PATH, 'config')
        # Grab the list of all existing darknet cfg files
        self.darknet_cfg_files = glob.glob(os.path.join(darknet_cfg_path_config_folder,'*.yaml'))
        # Remove the ros.yaml file -- that one doesn't represent a selectable trained neural net
        try:
            self.darknet_cfg_files.remove(os.path.join(darknet_cfg_path_config_folder,'ros.yaml'))
        except:
            rospy.logwarn("Unexpected: ros.yaml is missing from the darknet config path %s", darknet_cfg_path_config_folder)

        for f in self.darknet_cfg_files:
            yaml_stream = open(f, 'r')
            # Validate that it is a proper config file and gather weights file size info for load-time estimates
            cfg_dict = yaml.load(yaml_stream)
            yaml_stream.close()
            if ("yolo_model" not in cfg_dict) or ("weight_file" not in cfg_dict["yolo_model"]) or ("name" not in cfg_dict["yolo_model"]["weight_file"]):
                rospy.logerr("Debug: " + str(cfg_dict))
                rospy.logwarn("File does not appear to be a valid A/I model config file: " + f + "... not adding this classifier")
                continue

            classifier_name = os.path.splitext(os.path.basename(f))[0]
            weight_file = os.path.join(self.DARKNET_CFG_PATH, "yolo_network_config", "weights",cfg_dict["yolo_model"]["weight_file"]["name"])
            if not os.path.exists(weight_file):
                rospy.logwarn("Classifier " + classifier_name + " specifies non-existent weights file " + weight_file + "... not adding this classifier")
                continue

            self.classifier_list.append(classifier_name)
            self.classifier_weights_sizes[classifier_name] = os.path.getsize(weight_file)
        
        if (len(self.classifier_list) < 1):
            rospy.logwarn("No classiers identified for this system at %s", darknet_cfg_path_config_folder)

        rospy.Service('img_classifier_list_query', ImageClassifierListQuery, self.provide_classifier_list)
        rospy.Service('img_classifier_status_query', ImageClassifierStatusQuery, self.provide_classifier_status)

        rospy.Subscriber('start_classifier', ClassifierSelection, self.start_classifier_cb)
        rospy.Subscriber('stop_classifier', Empty, self.stop_classifier_cb)
        rospy.Subscriber('nepi_darknet_ros/set_threshold', Float32, self.set_threshold_cb) # Must match the node that gets launched by start_classifier()

        self.save_cfg_if = SaveCfgIF(updateParamsCallback=self.setCurrentSettingsAsDefault, paramsModifiedCallback=self.updateFromParamServer)

        # Load default params
        #namespace = rospy.get_namespace()
        self.updateFromParamServer()

        # Setup for data saving -- just bounding boxes for now
        self.save_data_if = SaveDataIF(['detection_bounding_boxes'])
        rospy.Subscriber('classifier/bounding_boxes', BoundingBoxes, self.boundingBoxesCallback)


def main():
    rospy.init_node("ai_detector_manager")
    manager = AIDetectorManager()
    rospy.spin()

if __name__ == '__main__':
    main()
