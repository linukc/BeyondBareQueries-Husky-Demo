import os
import json

import cv2
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.75  # Font size
COLOR_CV = (0, 0, 0)  # Text color (BGR: red)
thickness = 2 # Thickness of text

def project_point_cloud_to_image(image, depth_image, camera_pose, intrinsics_matrix, point_cloud):
    """
    Projects a 3D point cloud onto a 2D image using the camera pose, intrinsics, and depth image.
    
    Args:
        image (np.ndarray): The input RGB image (H x W x 3).
        camera_pose (np.ndarray): The 4x4 camera pose matrix (extrinsics).
        intrinsics_matrix (np.ndarray): The 3x3 camera intrinsics matrix.
        point_cloud (np.ndarray): The Nx3 point cloud array, where N is the number of points.
        
    Returns:
        np.ndarray: The image with the projected point cloud.
    """
    
    # Image height and width
    height, width = image.shape[:2]
    
    # Transform point cloud to camera coordinates
    # Add a column of 1s to the point cloud to make it Nx4 for matrix multiplication with pose
    ones = np.ones((point_cloud.shape[0], 1))
    point_cloud_homogeneous = np.hstack((point_cloud, ones))  # (N, 4)
    
    # Apply the camera pose to transform the points to camera coordinates
    point_cloud_camera_coords = (camera_pose @ point_cloud_homogeneous.T).T  # (N, 4)
    point_cloud_camera_coords = point_cloud_camera_coords[:, :3]  # Drop the homogeneous component

    #print(point_cloud_camera_coords)

    # Project points onto the image plane using the intrinsics matrix
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    
    # Only consider points in front of the camera (z > 0)
    #valid_points = point_cloud_camera_coords[:, 2] > 0

    #point_cloud_camera_coords = point_cloud_camera_coords[valid_points]

    # Project 3D points to 2D using the camera intrinsics
    u = (point_cloud_camera_coords[:, 0] * fx / point_cloud_camera_coords[:, 2]) + cx
    v = (point_cloud_camera_coords[:, 1] * fy / point_cloud_camera_coords[:, 2]) + cy
    
    # Round to nearest pixel and stack u, v coordinates
    pixel_coords = np.vstack((u, v)).T
    pixel_coords = np.round(pixel_coords).astype(int)

    # Filter valid pixel coordinates (those within image bounds)
    valid_pixel_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
    )
    
    #pixel_coords = pixel_coords[valid_pixel_mask]

    valid_pixel_mask = np.zeros(len(pixel_coords), dtype=bool)

    # Optionally, check against depth image to handle occlusions
    """
    for i, (u, v) in enumerate(pixel_coords):
        depth_value = depth_image[v, u]
        point_depth = point_cloud_camera_coords[i, 2]
        if np.abs(point_depth-depth_value) <= 0.05:  # Allow some tolerance for depth camera
            # Color the pixel in the image (for simplicity, color it white)
            valid_pixel_mask[i] = True
    pixel_coords = pixel_coords[valid_pixel_mask, :]
    """
    return pixel_coords

def draw_answer(result, targets, anchors, relations, user_query, LLM_answer, save_filename, SAVE_PATH, no_json=True, final_targets=[]):
    json_table = {'objects': [], 'relations': []}
    # camera_pose = np.linalg.inv(pose)

    objects_by_id = {
        obj['id']: obj['bbox_center']
        for obj in result
    }
    # Example list of 3D points and labels
    points = np.array([
        obj['bbox_center'] for obj in result
        if 'A wall on the side of a building' not in obj['description'] and obj['id'] not in targets and obj['id'] not in anchors 
    ])

    print(points.shape)
    labels = [f"{obj['id']}: {obj['description']}" for obj in result
             if 'A wall on the side of a building' not in obj['description'] and obj['id'] not in targets and obj['id'] not in anchors 
             ]  # Labels for each point
    others_ids = [l.split(':')[0] for l in labels]
    # Extract X, Y, Z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Create 3D figure
    # Create a figure with two subplots (one for 3D plot, one for text)
    fig = plt.figure(figsize=(5, 5))  # Wider figure
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.view_init(elev=0, azim=-90) 
    #ax1.set_xlim([-0.15, 2])  # Set X-axis limits from 0 to 1
    #ax1.set_ylim([-0.15, 1])  # Set Y-axis limits from 0 to 1
    #ax1.set_zlim([0.5, 3])  # Set Z-axis limits from 0 to 1
    # Plot points
    ax1.scatter(x, y, z, c='grey', marker='o', s=50)
    
    # Add labels to each point
    for i in range(len(points)):
        ax1.text(x[i], y[i], z[i]+0.02, result[i]['id'], fontsize=12, color='black', ha='center',)
        json_table['objects'].append({'label': labels[i], 'type': 'others'})

    anchors_ids = []
    if len(anchors) > 0:
        points = np.array([
            obj['bbox_center'] for obj in result
            if 'A wall on the side of a building' not in obj['description'] and obj['id'] in anchors 
        ])
        labels = [f"{obj['id']}: {obj['description']}" for obj in result
                if 'A wall on the side of a building' not in obj['description'] and obj['id'] in anchors 
                ]  # Labels for each point
        anchors_ids = [l.split(':')[0] for l in labels]
        # Extract X, Y, Z coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Plot points
        ax1.scatter(x, y, z, c='blue', marker='o', s=50)
        
        # Add labels to each point
        for i in range(len(points)):
            ax1.text(x[i], y[i], z[i]+0.02, anchors[i], fontsize=12, color='black', ha='center',)
            json_table['objects'] = [{'label': labels[i], 'type': 'anchors'}] + json_table['objects']

        # Labels
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

    targets_ids = []
    if len(targets) > 1:
        print(f"Result: {[obj['id'] for obj in result]}")
        print(f"Targets: {targets}")
        print(f"Final targets: {final_targets}")
        points = np.array([
            obj['bbox_center'] for obj in result
            if 'A wall on the side of a building' not in obj['description'] and obj['id'] in targets and obj['id'] not in final_targets
        ])
        labels = [f"{obj['id']}: {obj['description']}" for obj in result
                if 'A wall on the side of a building' not in obj['description'] and obj['id'] in targets and obj['id'] not in final_targets
                ]  # Labels for each point
        targets_ids = [l.split(':')[0] for l in labels]
        # Extract X, Y, Z coordinates
        print(points.shape)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Plot points
        ax1.scatter(x, y, z, c="green", marker='o', s=50)

        # Add labels to each point
        for i in range(len(points)):
            ax1.text(x[i], y[i], z[i]+0.02, targets[i], fontsize=12, color='black', ha='center',)
            json_table['objects'] = [{'label': labels[i], 'type': 'targets'}] + json_table['objects']

    if len(final_targets) > 0:
        points = np.array([
            obj['bbox_center'] for obj in result
            if 'A wall on the side of a building' not in obj['description'] and obj['id'] in final_targets
        ])
        labels = [f"{obj['id']}: {obj['description']}" for obj in result
                if 'A wall on the side of a building' not in obj['description'] and obj['id'] in final_targets
                ]  # Labels for each point
        # Extract X, Y, Z coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Plot points
        ax1.scatter(x, y, z, c="red", marker='o', s=50)

        # Add labels to each point
        for i in range(len(points)):
            ax1.text(x[i], y[i], z[i]+0.02, targets[i], fontsize=12, color='black', ha='center',)
            json_table['objects'] = [{'label': labels[i], 'type': 'answer'}] + json_table['objects']


    def update(angle):
        ax1.view_init(elev=20, azim=angle)
        return fig,
    elev_angles = np.concatenate([
        np.arange(70, 110, 1), 
        np.arange(110, 70, -1) 
    ])
    rot_animation = animation.FuncAnimation(
        fig, update, frames=elev_angles, interval=200, blit=False
    )
    gif_path = os.path.join(SAVE_PATH, f"3d_{save_filename}.gif")
    rot_animation.save(gif_path, dpi=80, writer='pillow')

    information = f"User query: {user_query}\n"

    with open(os.path.join(SAVE_PATH, f"{save_filename}.txt"), "w") as f:
        f.write(information)    
    
    LLM_answer = str(LLM_answer).replace('\n', '').replace('\\', '')
    information = f"LLM answer: {LLM_answer}"

    with open(os.path.join(SAVE_PATH, f"{save_filename}.txt"), "a") as f:
        f.write(information)    

    for rel in relations:
        if rel[0] not in targets:
            continue
      #  x1, y1, z1 = objects_by_id[rel[0]]
      #  x2, y2, z2 = objects_by_id[rel[1]]

      #  line_3d = np.array([
      #      [x1, y1, z1],  # center 1
      #      [x2, y2, z2],  # bbox2
      #  ])

      #  line2d = project_point_cloud_to_image(segmentation, depth, camera_pose, intrinsics, line_3d)
      #  #print(line2d)
      #  cv2.line(segmentation, tuple(line2d[0]), tuple(line2d[1]), (255, 255, 0), 2)

      #  mid_x = int((line2d[0][0] + line2d[1][0]) / 2)
      #  mid_y = int((line2d[0][1] + line2d[1][1]) / 2)
        wrapped_text = textwrap.fill(rel[2], width=15)
        json_table['relations'].append({'sub': rel[0], 'obj': rel[1], 'rel': wrapped_text})

    print("targets_ids: ", targets_ids)
    print("anchors_ids: ", anchors_ids)

    with open(os.path.join(SAVE_PATH, f"table_{save_filename}.json"), "w") as f:
        json.dump(json_table, f)
