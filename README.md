# Age-Gender-detection
Age and Gender Detection
1. Pre-trained Deep Learning Models
Face Detection Model:

Model files: opencv_face_detector.pbtxt, opencv_face_detector_uint8.pb
Framework: OpenCV DNN
Function: Detects faces in the image.
Gender Classification Model:

Model files: gender_deploy.prototxt, gender_net.caffemodel
Framework: Caffe
Function: Predicts gender as either "Male" or "Female".
Age Prediction Model:

Model files: age_deploy.prototxt, age_net.caffemodel
Framework: Caffe
Function: Predicts the age range of a detected face.
2. Key Components in Your Code
Face Detection (detect_faces)

Uses OpenCV DNN's face detector.
Converts the image into a blob and detects faces using the face model.
Filters detections based on a confidence threshold (0.7).
Gender Prediction (predict_gender)

Takes a cropped face.
Converts it into a blob and passes it through the gender model.
Returns either "Male" or "Female".
Age Prediction (predict_age)

Takes a cropped face.
Converts it into a blob and passes it through the age model.
Returns an age category
