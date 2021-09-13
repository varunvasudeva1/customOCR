# Package import
import cv2
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json

####################################################################################################
#
#       MODEL TRAINING - Everything until the next section is code to develop the model
#       The next section will be code that can be given an image input and yield a JSON
#       output with the text data from the input
#
####################################################################################################

# Lists that collect training images and training labels
train_data = []
train_labels = []

# Takes raw coordinate positions of boxes (not in px)
positions = []

# Loop through images and append image in bounding box and text to train_data and train_labels respectively
for index in range(1, 6):
    path = fr'/Users/varunvasudeva/Desktop/ds/DL/rule14problem/Rule14 Interview Sample Problem/Images/testimage{index}.jpeg'
    img = cv2.imread(path)
    img_copy = cv2.imread(path)

    # Defining image height and width (to calculate bounding boxes)
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Conversion to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding to binarize the image
    # (if pixel value < threshold, pixel value = max, else min)
    _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

    # Denoising to remove potential noise in the background
    cv2.medianBlur(img, 5)

    # Inverting image
    img = cv2.bitwise_not(img)

    # cv2.imshow('processed', img)
    # cv2.waitKey()

    # ==================================================================================
    # USING PYTESSERACT
    # import pytesseract as pyt
    # from pytesseract import Output
    #
    #
    # d = pyt.image_to_data(img_bin, output_type=Output.DICT)
    # print(d.keys())
    #
    # n_boxes = len(d['text'])
    # for i in range(n_boxes):
    #     if int(d['conf'][i]) > 60:
    #         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # cv2.imshow('img', img)
    # cv2.waitKey()
    #
    # Doesn't detect bounding boxes *at all*
    #
    # ==================================================================================
    # USING CUSTOM IMPLEMENTATION

    json_path = fr'/Users/varunvasudeva/Desktop/ds/DL/rule14problem/Rule14 Interview Sample Problem/OCR/testimage{index}.json'
    o = open(json_path)
    img_data = json.load(o)

    # Stores positions of bounding polygons per image and gets appended to
    # `positions` at the end of the loop
    image_pos = []

    for i in range(len(img_data['Blocks'])):
        # Selecting only those blocks that correspond to words
        if img_data["Blocks"][i]['BlockType'] == 'WORD':
            # Coordinate data for each bounding polygon
            c1 = img_data["Blocks"][i]['Geometry']['Polygon'][0]
            c2 = img_data["Blocks"][i]['Geometry']['Polygon'][1]
            c3 = img_data["Blocks"][i]['Geometry']['Polygon'][2]
            c4 = img_data["Blocks"][i]['Geometry']['Polygon'][3]

            # Setting the actual coordinate by multiplying by image dimensions
            x_min = float(min(c1['X'], c2['X'], c3['X'], c4['X']))
            x_max = float(max(c1['X'], c2['X'], c3['X'], c4['X']))
            y_min = float(min(c1['Y'], c2['Y'], c3['Y'], c4['Y']))
            y_max = float(max(c1['Y'], c2['Y'], c3['Y'], c4['Y']))

            image_pos.append([x_min, x_max, y_min, y_max])

            x_min = int(img_width * x_min)
            x_max = int(img_width * x_max)
            y_min = int(img_height * y_min)
            y_max = int(img_height * y_max)

            crop = img[y_min:y_max, x_min:x_max]
            crop = cv2.Canny(crop, 100, 200)
            target = img_data["Blocks"][i]['Text']

            # Appending input and corresponding output to training data

            # Resizing to have uniform array lengths
            crop = cv2.resize(crop, (60, 60))
            train_data.append(crop)
            train_labels.append(target)

        positions.append(image_pos)


train_data = np.array(train_data, dtype=object)
train_labels = np.array(train_labels)
positions = np.array(positions, dtype=object)

nsamples, nx, ny = train_data.shape
train_data = train_data.reshape((nsamples, nx * ny))

# x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2)
# The data split above gave us results for the models below

# print(train_data.shape)
# print(train_labels)

# Training the KNN model for OCR
# OpenCV kNN algorithm
# knn_cv2 = cv2.ml.KNearest_create()
# knn_cv2.train(samples=x_train, responses=y_train, layout=ROW_SAMPLE)
# ret, result, neighbours, dist = knn_cv2.findNearest(x_test, k=5)
#
# # Testing accuracy
# matches = result == y_test
# correct = np.count_nonzero(matches)
# accuracy = correct*100.0/result.size
# print(accuracy)

# Testing a bunch of different classifiers from the sklearn suite

knn_skl = KNeighborsClassifier(n_neighbors=3)
knn_skl.fit(train_data, train_labels)
accuracy_knn = knn_skl.score(train_data, train_labels)
# print(f'KNN Accuracy: {accuracy_knn}')

dt = DecisionTreeClassifier()
dt.fit(train_data, train_labels)
accuracy_dt = dt.score(train_data, train_labels)
# print(f'Decision Tree Accuracy: {accuracy_dt}')  # Lowest so far

rf = RandomForestClassifier()
rf.fit(train_data, train_labels)
accuracy_rf = rf.score(train_data, train_labels)

# print(f'Random Forest Accuracy: {accuracy_rf}')  # Does about as well as KNN, usually a tad worse

####################################################################################################
#
#       MODEL INFERENCE - This section houses code that can be given an image input and yield a JSON
#       output with the text data from the input
#       def image_to_text(image_path, dictionary):
#
####################################################################################################


# Function to append OCR output from an image to a dictionary
def OCR(image_path, dictionary):
    # This gets appended to the output dictionary
    text = []

    name = image_path[len(image_path) - 15: len(image_path) - 5]

    image = cv2.imread(image_path)
    image_copy = image.copy()
    image_height = image.shape[0]
    image_width = image.shape[1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    cv2.medianBlur(image, 5)
    image = cv2.bitwise_not(image)

    # Loop through images to generate output JSON

    # ==================================================================================
    # Approach 1 - Detecting Contours via OpenCV
    # contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # idx = 0
    # for cnt in contours:
    #     idx += 1
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     roi = image[y:y + h, x:x + w]
    #     roi = cv2.resize(roi, (60, 60))
    #     detected_images.append(roi)
    #
    # for img in detected_images:
    #     text.append(knn_skl.predict(img))
    #
    # for element in text:
    #     print(element)
    # ==================================================================================
    # Approach 2 - Find boxes by manually using training JSON bounding polygon data

    # Average positions of identifiable words in test images per line will serve as a basis for the positions
    # in the incoming input images

    # dictionary <- text <- line
    # multiple 'line's go into text, multiple 'text's go into dictionary with key 'name'

    line = []
    prev_y = 0
    # count = 0

    for box in positions[index]:
        x_lo = int(image_width * box[0])
        x_hi = int(image_width * box[1])
        y_lo = int(image_height * box[2])
        y_hi = int(image_height * box[3])

        # if count == 12:
        #     text.append(line)
        #     line.clear()
        #     count = 0

        if y_lo - prev_y > 40:
            text.append(line)
            line.clear()
            prev_y = y_lo

        cropped = image[y_lo:y_hi, x_lo:x_hi]
        cropped = cv2.resize(cropped, (60, 60))
        cropped = cv2.Canny(cropped, 100, 200)

        # cv2.imshow(f'image{index}_crop', cropped)
        # cv2.waitKey()

        nx, ny = cropped.shape
        cropped = np.reshape(cropped, (1, nx * ny))

        # If there's no character in the image, we don't bother putting it through the classifier
        percentWhite = cv2.countNonZero(cropped) / cropped.size * 100
        if percentWhite < 7:
            element = ''
        else:
            element = str(rf.predict(cropped)[0]).replace('/', '')
        line.append(element)

        # count += 1

    # text = np.asarray(text)
    # nsamples, nx, ny = text.shape
    # text = np.reshape(text, (nsamples, nx * ny))
    # print(text.shape)
    # for line in text:
    #     print(line)
    # ==================================================================================
    dictionary[name] = text


# Main output dictionary that gets converted to JSON
output_JSON = {}
for index in range(1, 6):
    path = fr'/Users/varunvasudeva/Desktop/ds/DL/rule14problem/Rule14 Interview Sample Problem/Images/testimage{index}.jpeg'
    OCR(path, output_JSON)

# for img in output_JSON.values():
#     for line in img:
#         for s in line:
#             s = s.replace('/', '')

# Writing output JSON
with open("Vasudeva_Varun_results.json", "w") as outfile:
    json.dump(output_JSON, outfile)


