1) Models Used
Face Detection: Detects faces using OpenCV's DNN model (opencv_face_detector.pbtxt, .pb).
Gender Classification: Predicts "Male" or "Female" (gender_deploy.prototxt, .caffemodel).
Age Prediction: Estimates an age range (age_deploy.prototxt, .caffemodel).
2)Key Functions
detect_faces() → Detects faces in an image using a confidence threshold of 0.7.
predict_gender() → Classifies the gender from a cropped face.
predict_age() → Predicts the age category from a cropped face.
3) Process
Load and resize the image.
Detect faces and extract each face.
Predict gender and age.
Draw bounding boxes and labels.
Save and display the processed image.
