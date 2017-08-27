import matplotlib.pyplot as plt
from sklearn.externals import joblib
from classify_vehicles import *
from scipy.ndimage.measurements import label

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_vehicles(img, scale, clf, scaler):
    # Convert image to colour space used in SVM classifier training
    img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_search = img_conv[y_start:y_stop, x_start:x_stop, :]
    img_local = np.copy(img)

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
        ch1 = img_search[:, :, hog_channel]
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Define blocks in image in x and y
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 pixels was the original training window, with 3 cells and 6 pix per cell
    window = train_img_width
    nwinblocks = (window // pix_per_cell) - cell_per_block + 1
    cells_per_xstep = 4  # Instead of overlap, define how many cells to step
    cells_per_ystep = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nwinblocks) // cells_per_xstep
    nysteps = (nyblocks - nwinblocks) // cells_per_ystep

    # Initialize a list to append window positions to
    window_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            # Define an empty list to receive features
            img_features = []
            ypos = yb * cells_per_ystep
            xpos = xb * cells_per_xstep
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            if use_spatial == True or use_hist == True:
                # Extract the image patch
                subimg = cv2.resize(img_search[ytop:ytop + window, xleft:xleft + window], (train_img_height, train_img_width))
            # Get color features
            if use_spatial == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                img_features.append(spatial_features)
            if use_hist == True:
                hist_features = color_hist(subimg, nbins=hist_bins)
                img_features.append(hist_features)
            if hog_channel == 'ALL':
                # Extract HOG for all channels in this patch
                hog_feat1 = hog1[ypos:ypos + nwinblocks, xpos:xpos + nwinblocks].ravel()
                hog_feat2 = hog2[ypos:ypos + nwinblocks, xpos:xpos + nwinblocks].ravel()
                hog_feat3 = hog3[ypos:ypos + nwinblocks, xpos:xpos + nwinblocks].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                # Extract HOG for single channel in this patch
                hog_features = hog1[ypos:ypos + nwinblocks, xpos:xpos + nwinblocks].ravel()
            img_features.append(hog_features)
            feature_vector = np.concatenate(img_features).astype(np.float64)
            # Scale features and make a prediction
            test_features = scaler.transform(feature_vector.reshape(1, -1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                print('Confidence: ', clf.decision_function(test_features))
                if clf.decision_function(test_features) > 1.0:
                    win_scaled = np.int(window * scale)
                    startx = np.int(xleft * scale) + x_start
                    starty = np.int(ytop * scale) + y_start
                    endx = startx + win_scaled
                    endy = starty + win_scaled
                    # Append window position to list
                    window_list.append(((startx, starty), (endx, endy)))

            if scale ==1:
                colour_tuple = (0, 0, 255)
            elif scale == 1.5:
                colour_tuple = (255, 0, 0)
            else:
                colour_tuple = (0, 255, 0)
            win_scaled = np.int(window * scale)
            startx = np.int(xleft * scale) + x_start
            starty = np.int(ytop * scale) + y_start
            endx = startx + win_scaled
            endy = starty + win_scaled
            cv2.rectangle(img_local, (startx, starty),(endx, endy), colour_tuple, 3)
    plt.imshow(img_local)
    plt.show()
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
def draw_boxes(img_draw, bboxes):
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img_draw, bbox[0], bbox[1], (0, 0, 255), 3)
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
    scale_list = [2,1.5,1]
    # Min and max in y to search in slide_window based on scale
    x_start_stop = [(0, 1280), (0,1280), (0,1280)]#[(0, 1200),(200, 850),(400,700)]
    y_start_stop = [(500, 700),(400, 600),(400, 500) ]#[(400, 700),(400, 600),(470, 510) ]
    all_detected_windows = []
    use_slow_slide = False

    img_names = glob.glob('./test_images/*.*')
    for img_name in img_names:
        img = cv2.imread(img_name)
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

            '''if use_slow_slide == True:
                win_size = (np.int(train_img_width/scale), np.int(train_img_height/scale))
                windows = slide_window(img, x_start=x_start, x_stop=x_stop, y_start=y_start, y_stop=y_stop,
                                   xy_window=win_size, xy_overlap=(0.5, 0.5))

                detected_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, use_spatial=use_spatial,
                                         use_hist=use_hist, use_hog=use_hog)
            else:'''
            detected_windows = find_vehicles(img, scale, svc, X_scaler)
            all_detected_windows.extend(detected_windows)
        img_heat=create_heatmap(img_heat,all_detected_windows)
        labels = label(img_heat)

        t2 = time.time()
        print('Detection time: ', round(t2-t1,2))
        print(labels[1], 'cars found')
        img_draw  = draw_labeled_boxes(img_draw, labels)
        img_boxes = draw_boxes(img_boxes, all_detected_windows)

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


