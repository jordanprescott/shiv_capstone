from PIL import Image
import numpy as np
import json
import sys
import os


# Add the directory containing `get_yolo_json.py` to the Python path
sys.path.append("/home/jordanprescott/shiv_capstone")

from get_yolo_json import get_json



def get_map_of_specific_depth(depth_map, specific_depth):
    filter_depth_map = depth_map.copy()
    filter_depth_map[depth_map > specific_depth] = 0
    return filter_depth_map


def average_depth_over_bbox(specific_depth_map: np.ndarray, bbox: tuple):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_sdm = specific_depth_map[y_min:y_max, x_min:x_max]
    nonzero = cropped_sdm[cropped_sdm != 0]
    
    avg_depth = np.mean(nonzero) if nonzero.size > 0 else np.nan
    return avg_depth


def get_bboxes(yolo_output_json, im_shape):
    if isinstance(yolo_output_json, str):
        yolo_output_json = json.loads(yolo_output_json)

    bboxes = []
    for detection in yolo_output_json:
        label = detection["label"]
        x_min, y_min, x_max, y_max = map(int, detection['bbox'])
        relative_angle = ((x_max - x_min)/2 + x_min) / im_shape[1]  # Relative position in image width
        bboxes.append((label, (x_min, y_min, x_max, y_max), relative_angle))
    return bboxes


def get_oda(im_path: str, dm_path: str):
    # Load the image as grayscale
    image = Image.open(im_path).convert("L")

    # Load depth map
    data = np.load(dm_path)
    depth = data[data.files[0]]

    # Get YOLO detections
    yolo_output_json = get_json(im_path)

    # Convert YOLO output to bounding boxes
    bounding_boxes = get_bboxes(yolo_output_json, np.array(image).shape)
    
    print(f"Number of detected objects: {len(bounding_boxes)}")

    # List of specific depth thresholds
    specific_depths = [1, 5, 10, 100]
    results = []

    for sd in specific_depths:
        print(f"Depth threshold: {sd}")
        for label, bbox, angle in bounding_boxes[:]:
            sdm = get_map_of_specific_depth(depth, sd)
            avg_depth = average_depth_over_bbox(sdm, bbox)
            if not np.isnan(avg_depth):
                results.append((label, avg_depth, angle))
                bounding_boxes.remove((label, bbox, angle))
                print(f"Object: {label}, Distance: {avg_depth:.2f}, Angle: {angle}")
                
    print(f"Remaining unprocessed objects: {len(bounding_boxes)}")

    # Prepare data for text-to-speech function
    objects = [obj for obj, _, _ in results]
    distances = [distance for _, distance, _ in results]
    positions = [angle for _, _, angle in results]
    importance = [5] * len(objects)  # Assign a default importance value (e.g., 5)

    return objects, distances, positions, importance


# Example Usage
image_path = "./misc/smaller_cars.png"
depth_map_path = "./misc/resized_out.npz"

objects, distances, positions, importance = get_oda(image_path, depth_map_path)

# Print the results to confirm
print("Objects:", objects)
print("Distances:", distances)
print("Positions:", positions)
print("Importance:", importance)
