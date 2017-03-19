
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from train_classifier import get_hog_features
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
import numpy as np
import cv2
from skimage.feature import hog
import multiprocessing
from defs import multi_window_search

core_count = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=core_count)

dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def find_boxes(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, parallell=True):
    import time
    #print("Calculating boxes. scale:", scale)
    start = time.time()
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    #print("Setup for calculation. Scale:", scale, time.time()-start)
    start = time.time()
    # Compute individual channel HOG features for the entire image
    hogs = [get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False), get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False), get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)]

    hog1 = hogs[0] #get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = hogs[1] #get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = hogs[2] #get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #print("HOG whole image. Scale:", scale, time.time()-start)
    start = time.time()
    boxes = []
    
    for xb in range(nxsteps):
        args = []
        for yb in range(nysteps):
            args.append((xb, yb, ystart, ystop, pix_per_cell, cells_per_step, nblocks_per_window, window, scale, hog1, hog2, hog3))
        if parallell :
            results = pool.map(multi_window_search, args)
        else:
            results = map(multi_window_search, args)
        results = [result for result in results if result != None]
        boxes.extend(results)
    
    return boxes

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

def calculate_labeled_bboxes(labels):
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
    
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for bbox in calculate_labeled_bboxes(labels):
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 3)
        
    # Return the image
    return img

def add_analysis(img, heatmap):
    heatmap = cv2.resize(heatmap, (int(heatmap.shape[1]/4), int(heatmap.shape[0]/4)))
    heatmap = np.dstack((heatmap, heatmap, heatmap))*255
    img[0:heatmap.shape[0], 0:heatmap.shape[1]] = heatmap
    return img


class HeatTracker:
    
    def __init__(self):
        self.boxes_list = []
        self.HEAT_FRAMES = 6
        self.heatmap = None
    
    def addBoxes(self, boxes):
        self.boxes_list.append(boxes)
        if len(self.boxes_list) > self.HEAT_FRAMES:
            self.boxes_list.pop(0)
        if len(self.boxes_list) > self.HEAT_FRAMES:
            print("TO MANY HEAT FRAMES")
        return max(len(self.boxes_list), self.HEAT_FRAMES)
    
    def allBoxes(self):
        all_boxes = []
        for boxes in self.boxes_list:
            all_boxes.extend(boxes)
        return all_boxes
    
    def setHeatMap(self,heatmap):
        self.heatmap = heatmap

heat_tracker = HeatTracker()

def process_image(image):        
    boxes = []
    
    #Handle 1x. Only scan far away. Time consuming
    """
    ystart = 400
    ystop = 450
    scale = 0.5
    boxes.extend(find_boxes(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block))
    """

    ystart = 350
    ystop = 500
    scale = 1
    boxes.extend(find_boxes(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block))
   

    ystart = 400
    ystop = 656
    for scale in [1.5, 2, 3, 4, 8]:
        boxes.extend(find_boxes(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block))

    threshold_n = heat_tracker.addBoxes(boxes)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,heat_tracker.allBoxes())
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,min(5,threshold_n * 2))

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    add_analysis(draw_img, heatmap)
    return draw_img

if __name__ == "__main__":
    if True:
        from moviepy.editor import VideoFileClip
        clip1 = VideoFileClip("project_video.mp4")
        #clip1 = VideoFileClip("test_video.mp4")
        output_video = clip1.fl_image(process_image)
        output_video.write_videofile("project_video_output.mp4", audio=False)
        #output_video.write_videofile("test_video_output.mp4", audio=False)
    else:
        img = mpimg.imread('./test_images/test1.jpg')
        plt.imshow(process_image(img))
        plt.show()