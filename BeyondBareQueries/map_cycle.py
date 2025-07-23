#!/usr/bin/env python
import os
import yaml
import json
import threading

import cv2
import gzip
import torch
import rospy
import pickle
import numpy as np
from PIL import Image
from std_msgs.msg import String  # или другой тип сообщений для ваших топиков
from husky_demo_transport.msg import Bundle
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import CompressedImage

from bbq_core.objects_map.nodes_constructor import NodesConstructor


SAVE_PATH = "outputs"
os.makedirs(SAVE_PATH, exist_ok=True)
INTRINSICS = np.eye(4, 4)
INTRINSICS[:3, :3] = np.array([262.49603271484375, 0.0, 322.422119140625, 0.0, 
                               262.49603271484375, 182.1229248046875, 
                               0.0, 0.0, 1.0]).reshape(3, 3)

with open("/home/docker_user/BeyondBareQueries/BeyondBareQueries/config.yaml") as file:
    config = yaml.full_load(file)
    print("!!!!!!!!! DOUBLE CHECK CAMERA PARAMETERS IN THE CONFIG !!!!!!!!!!")
nodes_constructor = NodesConstructor(config["nodes_constructor"])

def save_depth_with_colormap(depth_array, save_path, max_depth=5.0):
    # Remove singleton channel dimension (H, W, 1) → (H, W)
    depth = np.squeeze(depth_array)

    # Normalize to 0–255 for visualization
    depth_clipped = np.clip(depth, 0, max_depth)
    depth_normalized = (depth_clipped / max_depth * 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)

    # Save image
    cv2.imwrite(save_path, depth_colored)

def compressed_image_to_numpy(msg: CompressedImage, is_depth=False) -> np.ndarray:
    """
    Преобразует ROS CompressedImage в numpy array.
    Для color: вернет RGB изображение (H, W, 3)
    Для depth: вернет grayscale или raw depth, зависит от формата сжатия
    """
    if not msg.data:
        raise ValueError("CompressedImage data is empty")

    if is_depth:
        # depth может быть сохранён в PNG 16UC1
        np_arr = np.frombuffer(msg.data[12:], np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    else:
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    if cv_image is None:
        raise ValueError("Не удалось декодировать CompressedImage")

    return cv_image

def pose_to_matrix(pose: Pose) -> np.ndarray:
    """
    Преобразует geometry_msgs/Pose в numpy 4x4 трансформационную матрицу.
    """
    translation = [pose.position.x, pose.position.y, pose.position.z]
    rotation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    matrix = tf_trans.quaternion_matrix(rotation)
    matrix[:3, 3] = translation
    return matrix

class SensorProcessor:
    def __init__(self):
        self.enable_sensor_callback = False
        self.sensor_subscriber = None
        self.lock = threading.Lock()

        self.STEP_IDX = 0
        self.COLORS_BANK = []
        self.POSES_BANK = []

        self.start_sub = rospy.Subscriber('/start_topic', String, self.start_callback, queue_size=1)
        self.stop_sub = rospy.Subscriber('/stop_topic', String, self.stop_callback, queue_size=1)

        rospy.loginfo("Node initialized. Waiting for start signal...")

    def start_callback(self, msg):
        if not self.enable_sensor_callback:
            rospy.loginfo("Start signal received. Subscribing to /sensor_topic.")
            self.enable_sensor_callback = True
            self.sensor_subscriber = rospy.Subscriber('/bundled_data_throttle', Bundle, self.sensor_callback, queue_size=1)
        else:
            rospy.loginfo("Already processing sensor data.")

    def sensor_callback(self, msg):
        if not self.enable_sensor_callback:
            rospy.logwarn("Received sensor data while. Ignoring.")
            return

        with self.lock:
            rospy.loginfo("Processing sensor data: №{}".format(self.STEP_IDX))
            # Здесь может быть любая обработка данных
            color = compressed_image_to_numpy(msg.color, is_depth=False)
            depth = compressed_image_to_numpy(msg.depth, is_depth=True) / 1000 # depth scale
            pose = pose_to_matrix(msg.pose)
            frame = color, depth, INTRINSICS, pose
            Image.fromarray(color).save(f"{SAVE_PATH}/latest_scan.png")
            save_depth_with_colormap(depth, f"{SAVE_PATH}/latest_depth.png")
            self.COLORS_BANK.append(color)
            self.POSES_BANK.append(pose)
            nodes_constructor.integrate(self.STEP_IDX, frame) # frame = color, depth, intrinsics, pose
            rospy.loginfo("Trigger sensor callback.")
            torch.cuda.empty_cache()
            self.STEP_IDX += 1

    def stop_callback(self, msg):
        if self.enable_sensor_callback:
            rospy.loginfo("Stop signal received. Unsubscribing from /sensor_topic and shutting down.")
            if self.sensor_subscriber is not None:
                self.sensor_subscriber.unregister()
            self.enable_sensor_callback = False
            with self.lock:
                rospy.loginfo("Start preprocessing.")
                nodes_constructor.postprocessing()
                torch.cuda.empty_cache()
                results = {'objects': nodes_constructor.objects.to_serializable()}
                with gzip.open(os.path.join(SAVE_PATH,
                    f"objects.pkl.gz"), "wb") as f:
                        pickle.dump(results, f)
                rospy.loginfo("Finish preprocessing.")
                
                rospy.loginfo("Start projecting.")
                nodes_constructor.project(
                    poses=self.POSES_BANK,
                    intrinsics=INTRINSICS
                )
                torch.cuda.empty_cache()
                rospy.loginfo("Finish projecting.")

                rospy.loginfo("Start describing.")
                nodes = nodes_constructor.describe(
                    colors=self.COLORS_BANK
                )
                torch.cuda.empty_cache()
                rospy.loginfo("Finish describing.")

                rospy.loginfo('Saving graph nodes in json file.')
                os.makedirs(SAVE_PATH, exist_ok=True)
                with open(os.path.join(SAVE_PATH, "objects.json"), 'w') as f:
                    json.dump(nodes, f)

            rospy.signal_shutdown("Stop signal received.")            
        else:
            rospy.loginfo("Stop signal received, but processing was not active.")

if __name__ == "__main__":
    rospy.init_node('sensor_processor_node')
    processor = SensorProcessor()
    rospy.spin()
