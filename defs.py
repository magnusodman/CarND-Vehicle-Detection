import numpy as np
import pickle

dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]

def window_search(xb, yb, ystart, ystop, pix_per_cell, cells_per_step, nblocks_per_window, window, scale, hog1, hog2, hog3):
    ypos = yb*cells_per_step
    xpos = xb*cells_per_step
    # Extract HOG for this patch
    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

    xleft = xpos*pix_per_cell
    ytop = ypos*pix_per_cell

    # Scale features and make a prediction
    test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1))    
    test_prediction = svc.predict(test_features)
    
    if test_prediction == 1:
        xbox_left = np.int(xleft*scale)
        ytop_draw = np.int(ytop*scale)
        win_draw = np.int(window*scale)
        box = (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)
        return box
    else:
        return None

def multi_window_search(args):
    return window_search(*args)