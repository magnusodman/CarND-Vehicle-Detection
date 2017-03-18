import numpy as np
from detections import bbox_detections

import math
def distance(center1, center2):
    return math.sqrt((center1[0]- center2[0])**2 + (center1[1]- center2[1])**2)    

def center(bbox):
    return ((bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2) 

class DetectedCar:

    def __init__(self, bbox):
        self.bboxes= [bbox]
        self.detect_distance = 50

    def tracks(self, bbox):
        if distance(self._center(), center(bbox)) < self.detect_distance:
            return True
        return False

    def _center(self):
        return np.mean([center(bbox)[0] for bbox in self.bboxes]), np.mean([center(bbox)[1] for bbox in self.bboxes])

    def track(self, bbox):
        self.bboxes.append(bbox)

    def notrack(self):
        self.bboxes.pop(0)

    def carBbox(self):
        return self.bboxes[-1]

    def tracked(self):
        return len(self.bboxes) > 0


def track_boxes(boxes, detected_cars):
    
    #Add trackings to existing cars
    non_tracked_boxes = []
    tracked_cars_in_image = set()
    for bbox in boxes:
        tracking_cars_for_box = [car for car in detected_cars if car.tracks(bbox)]
        tracked_cars_in_image = tracked_cars_in_image.union(tracking_cars_for_box)
        if len(tracking_cars_for_box) == 0:
            non_tracked_boxes.append(bbox)
        else:
            tracking_cars_for_box[0].track(bbox)
    
    [detected_car.notrack() for detected_car in detected_cars if detected_car not in tracked_cars_in_image]
    
    #Remove untracked cars
    tracked_cars = []
    for detected_car in detected_cars:
        if detected_car.tracked():
            tracked_cars.append(detected_car)

    #Add cars for untracked boxes
    for non_tracked_box in non_tracked_boxes:
        tracked_cars.append(DetectedCar(non_tracked_box))
        
    return tracked_cars

detected_cars = []
for bboxes in bbox_detections:
    detected_cars = track_boxes(bboxes, detected_cars)
    print("Detected cars: ", len(detected_cars))


