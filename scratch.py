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
        p0_0 = int(np.mean([box[0][0] for box in self.bboxes]))
        p0_1 = int(np.mean([box[0][1] for box in self.bboxes]))
        p1_0 = int(np.mean([box[1][0] for box in self.bboxes]))
        p1_1 = int(np.mean([box[1][1] for box in self.bboxes]))

        return (p0_0, p0_1), (p1_0, p1_1)
        #return self.bboxes[-1]

    def tracked(self):
        return len(self.bboxes) > 0
    
    def isDisplayable(self):
        return len(self.bboxes) > 2

class CarTracker2:
    def __init__(self):
        self.detected_cars = []

    def track_boxes(self, boxes):
        
        #Add trackings to existing cars
        non_tracked_boxes = []
        tracked_cars_in_image = set()
        for bbox in boxes:
            tracking_cars_for_box = [car for car in self.detected_cars if car.tracks(bbox)]
            tracked_cars_in_image = tracked_cars_in_image.union(tracking_cars_for_box)
            if len(tracking_cars_for_box) == 0:
                non_tracked_boxes.append(bbox)
            else:
                tracking_cars_for_box[0].track(bbox)
        
        [detected_car.notrack() for detected_car in self.detected_cars if detected_car not in tracked_cars_in_image]
        
        #Remove untracked cars
        tracked_cars = []
        for detected_car in self.detected_cars:
            if detected_car.tracked():
                tracked_cars.append(detected_car)

        #Add cars for untracked boxes
        for non_tracked_box in non_tracked_boxes:
            tracked_cars.append(DetectedCar(non_tracked_box))
            
        self.detected_cars = tracked_cars

car_tracker2 = CarTracker2()

for bboxes in bbox_detections:
    car_tracker2.track_boxes(bboxes)
    print("Detected cars: ", len(car_tracker2.detected_cars))


