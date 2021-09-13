# customOCR
OCR system using a Random Forest algorithm to categorize images

This script uses OpenCV to make images grayscale, invert them, detect edges, etc. and then sklearn's Random Forest Classifier to classify the digits. For training data, 5 images and their OCR outputs were given to be used in building this model. 5 images is really not all that much training data but that's part of the challenge. With more time, I'd have loved to add many versions per bounded image involving transformations, skew, and distortion - this would perhaps help mitigate the lack of training data. The classifier isn't altogether too accurate but the pipeline allows for very quick modification and is flexible to change in all code sections. This was my first time using OpenCV for genuine image manipulation and actually creating an image training data set manually, as compared to having a clean and pretty set to work with.

![testimage1](https://user-images.githubusercontent.com/37934117/133106387-8fbdf095-cfcc-4a2e-a534-bc40217cc5b4.jpeg)

The above image is a template of what is to be fed into the OCR program. PyTesseract tried and failed to find bounding boxes, as did OpenCV's various contour detection features. This meant using the output OCR data given to us was necessary in order to find the bounding polygons and create a label for that image.
