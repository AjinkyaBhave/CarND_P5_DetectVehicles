import matplotlib.pyplot as plt
from sklearn.externals import joblib
from classify_vehicles import *
from scipy.ndimage.measurements import label

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        use_spatial=True, use_hist=True, use_hog=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'LAB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if use_spatial == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if use_hist == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if use_hog == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start=None, x_stop=None, y_start=None, y_stop=None,
                 xy_window=(train_img_width, train_img_height), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start == None:
        x_start = 0
    if x_stop == None:
        x_start_stop[1] = img.shape[1]
    if y_start == None:
        y_start = 0
    if y_stop == None:
        y_stop = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_stop - x_start
    yspan = y_stop - y_start
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, use_spatial=True,
                   use_hist=True, use_hog=True):
    # Create an empty list to receive positive detection windows
    on_windows = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                              (train_img_height, train_img_width))
        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, use_spatial=use_spatial,
                                       use_hist=use_hist, use_hog=use_hog)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # Return windows for positive detections
    return on_windows

# Define a single function that can extract features using hog sub-sampling and make predictions
def subsample_scaled_image(img, scale, svc, X_scaler):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_search = img[y_start:y_stop, :, :]
    if scale != 1:
        img_search = cv2.resize(img_search, (np.int(img_search.shape[1] / scale),
                                             np.int(img_search.shape[0] / scale)))

    if hog_channel == 'ALL':
        ch1 = img_search[:, :, 0]
        ch2 = img_search[:, :, 1]
        ch3 = img_search[:, :, 2]

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        ch1 = img_search[:,:,hog_channel]
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Define blocks in image in x and y
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 pixels was the original training window, with 3 cells and 6 pix per cell
    window = train_img_width
    nwinblocks = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 3  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nwinblocks) // cells_per_step
    nysteps = (nyblocks - nwinblocks) // cells_per_step

    # Initialize a list to append window positions to
    window_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            # Define an empty list to receive features
            img_features = []
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_search[ytop:ytop + window, xleft:xleft + window], (64, 64))
            # Get color features
            if use_spatial == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                img_features.append(spatial_features)
            if use_hist == True:
                hist_features = color_hist(subimg, nbins=hist_bins)
                img_features.append(hist_features)
            if hog_channel == 'ALL':
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nwinblocks, xpos:xpos + nwinblocks].ravel()
                hog_feat2 = hog2[ypos:ypos + nwinblocks, xpos:xpos + nwinblocks].ravel()
                hog_feat3 = hog3[ypos:ypos + nwinblocks, xpos:xpos + nwinblocks].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                # Extract HOG for this patch
                hog_features = hog1[ypos:ypos + nwinblocks, xpos:xpos + nwinblocks].ravel()
            img_features.append(hog_features)
            feature_vector = np.concatenate(img_features).astype(np.float64)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.array(feature_vector).reshape(1, -1))
            # np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                win_scaled = np.int(window * scale)
                startx = np.int(xleft * scale) + x_start
                starty = np.int(ytop * scale) + y_start
                endx = startx + win_scaled
                endy = starty + win_scaled
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
                #cv2.rectangle(img_draw, (startx, starty),(endx, endy), (0, 0, 255), 6)
    return window_list

def create_heatmap(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Zero out pixels below the threshold
    heatmap[heatmap < heat_thresh] = 0
    # Return updated heatmap
    return heatmap

# Define a function to draw bounding boxes
def draw_boxes(img_draw, bboxes, color=(0, 0, 255), thick=3):
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img_draw, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return img_draw

def draw_labeled_boxes(img_draw, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img_draw, bbox[0], bbox[1], (0,0,255), 3)
    # Return the image
    return img_draw

if __name__ == '__main__':
    # Scales to search for vehicle features in image
    scale_list = [2, 1.5, 1]
    # Min and max in y to search in slide_window based on scale
    x_start_stop = [(0, 1200),(200, 850),(400,700)]
    y_start_stop = [(500, 700),(450, 550),(470, 510) ]
    all_detected_windows = []
    use_slow_slide = False

    img = cv2.imread('./test_images/bbox-example-image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Image copy to draw detected vehicle boxes after heat maps
    img_draw = np.copy(img)
    # Image copy to draw detected vehicle boxes before heat maps
    img_boxes = np.copy(img)
    # Heat map to combine multiple scale detections
    img_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Minumum number of times a pixel is present in a bounding box set to accept detection
    heat_thresh = 1

    # Load pre-trained SVM classifier model
    svc = joblib.load(svm_model_path)
    # Fit a per-column scaler
    X_scaler = joblib.load(scaler_model_path)
    print('Load SVM and Scaler')

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255

    t1 = time.time()
    for i, scale in enumerate(scale_list):
        x_start = x_start_stop[i][0]
        x_stop  = x_start_stop[i][1]
        y_start = y_start_stop[i][0]
        y_stop  = y_start_stop[i][1]
        print(scale, x_start, x_stop, y_start, y_stop)

        if use_slow_slide == True:
            win_size = (np.int(scale*train_img_width), np.int(scale*train_img_height))
            windows = slide_window(img, x_start=x_start, x_stop=x_stop, y_start=y_start, y_stop=y_stop,
                               xy_window=win_size, xy_overlap=(0.5, 0.5))

            detected_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, use_spatial=use_spatial,
                                     use_hist=use_hist, use_hog=use_hog)
        else:
            detected_windows = subsample_scaled_image(img, scale, svc, X_scaler)

        all_detected_windows.extend(detected_windows)

    img_heat=create_heatmap(img_heat,all_detected_windows)
    labels = label(img_heat)

    t2 = time.time()
    print('Detection time: ', round(t2-t1,2))
    print(labels[1], 'cars found')
    img_draw  = draw_labeled_boxes(img_draw, labels)
    img_boxes = draw_boxes(img_boxes, all_detected_windows, color=(0, 0, 255), thick=3)

    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(img_boxes)
    plt.title('Car Boxes')
    plt.subplot(132)
    plt.imshow(img_draw)
    plt.title('Car Positions')
    plt.subplot(133)
    plt.imshow(img_heat, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()


