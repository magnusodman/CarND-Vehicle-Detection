from detections import box_detections
import numpy as np
from scipy.ndimage.measurements import label
import cv2

class CarTrackerAPA:
    track_history = []
    def track(self, bboxes):
        self.track_history.append(bboxes)
        if len(self.track_history) > 10:
            self.track_history.pop(0)
    def trackedCars(self):
        return [((100, 100), (200, 200))]

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def calculate_labeled_bboxes(boxes,size =(1280, 720)):
    heat = np.zeros(size).astype(np.float)
    heat = add_heat(heat,boxes)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    labeled_bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        labeled_bboxes.append(bbox)
    return labeled_bboxes

class Detections:
    index = 0
    detections = box_detections
    def next(self):
        self.index = self.index + 1
        return self.detections[self.index-1]

detections = Detections()

def draw_bboxes(img, bboxes, color = (0, 0, 255)):
    for bbox in bboxes:
        #cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        cv2.rectangle(img, bbox[0], bbox[1], color, 2)
        
    return img

from scratch import CarTracker2
car_tracker = CarTracker2()
def process_image(image):
    boxes = detections.next()
    bboxes  = calculate_labeled_bboxes(boxes, size=(image.shape[0], image.shape[1]))
    draw_img = draw_bboxes(np.copy(image),bboxes)
    car_tracker.track_boxes(bboxes)
    car_boxes = [car.carBbox() for car in car_tracker.detected_cars if car.isDisplayable()]
    draw_img = draw_bboxes(draw_img, car_boxes, color = (255, 0, 0))
    return draw_img



if __name__ == "__main__":
    from moviepy.editor import VideoFileClip
    clip1 = VideoFileClip("test_video.mp4")
    output_video = clip1.fl_image(process_image)
    output_video.write_videofile("test_video_output_detections.mp4", audio=False)