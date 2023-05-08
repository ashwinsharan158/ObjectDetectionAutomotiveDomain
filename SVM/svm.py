
# imports
import os
import cv2
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label
import pickle
from glob import glob
import matplotlib.pyplot as plt

# parameters
spatial = 32
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size= (32, 32)
object = 'car'
ystart_ystop_scale = [(405, 510, 1), (400, 600, 1.5), (500, 710, 2.5)]


# load dataset
images = []
neg_images = []
for root, dirs, files in os.walk('./data/' + object + '/'):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            images.append(os.path.join(root, file))
            
for root, dirs, files in os.walk('./data/neg/'):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            neg_images.append(os.path.join(root, file))


# features extract function
# Compute binned color features by scaling images down 
def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel() 
    return features

# Compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


# Return HOG features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, feature_vector=feature_vec)
        return features

# Extract feature wrapper that extracts and combines all features
def extract_features(imgs, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):

    features = []
    for file in imgs:
        image = mpimg.imread(file)
            
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, feature_vec=True))
        hog_features = np.ravel(hog_features)
        
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    return features


# feature extraction
images_features = extract_features(images, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        spatial_size=(spatial, spatial), hist_bins=hist_bins, hist_range=(0, 256))

neg_images_features = extract_features(neg_images,orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        spatial_size=(spatial, spatial), hist_bins=hist_bins, hist_range=(0, 256))

# data preparation
rand_state = np.random.randint(0, 100)
X = np.vstack((images_features, neg_images_features)).astype(np.float64)        
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(images_features)), np.zeros(len(neg_images_features))))

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
print('Feature vector length:', len(X_train[0]))

# classifier
svc = LinearSVC()
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# save data
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial"] = spatial
dist_pickle["hist_bins"] = hist_bins
pickle.dump(dist_pickle, open("svc_pickle_" + object + ".p", 'wb') )


# load data
dist_pickle = pickle.load( open("svc_pickle_" + object + ".p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatia = dist_pickle["spatial"]
hist_bins = dist_pickle["hist_bins"]


# hog sliding window
# Extracts features using hog sub-sampling and make predictions
def find_objs(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, ystart_ystop_scale, h_shift=0):
    bbox_detection_list=[]
    img = img.astype(np.float32)/255

    for (ystart, ystop, scale) in ystart_ystop_scale:
        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
        
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 3
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 

        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                test_prediction = svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bbox_detection_list.append(((xbox_left+h_shift, ytop_draw+ystart),(xbox_left+win_draw+h_shift,ytop_draw+win_draw+ystart)))
    return bbox_detection_list


# heatmap
# Accumulation of labels from last N frames
class Detect_history():
    def __init__ (self):
        self.queue_len = 7
        self.queue = []

    # Put new frame
    def put_labels(self, labels):
        if (len(self.queue) > self.queue_len):
            tmp = self.queue.pop(0)
        self.queue.append(labels)
    
    # Get last N frames hot boxes
    def get_labels(self):
        detections = []
        for label in self.queue:
            detections.extend(label)
        return detections

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], (bbox[1][0]+10,bbox[1][1]-10), (0,0,255) if object is 'car' else (0,255,0), 2)
    return img

def process_image(img): 
    bbox_detection_list = find_objs(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, ystart_ystop_scale)
    blank = np.zeros_like(img[:,:,0]).astype(np.float)

    detect_history.put_labels(bbox_detection_list)
    bbox_detection_list = detect_history.get_labels()
    heatmap = add_heat(blank, bbox_detection_list)
    labels = label(heatmap)
    result = draw_labeled_bboxes(np.copy(img), labels)
    
    return result

# test images
detect_history = Detect_history()
test_images = np.array([plt.imread(i) for i in glob('./data/test_images/*.jpg')])
result = process_image(test_images[0])
plt.figure(figsize = (20,20))
plt.imshow(result)
plt.axis("off")

# test videos
detect_history = Detect_history()
file_name = 'video'
processed_video_path = file_name + '_predict_' + object + '.mp4'
video = VideoFileClip("./videos/" + file_name + ".mp4")
processed_video = video.fl_image(process_image)
processed_video.write_videofile(processed_video_path, audio=False)


