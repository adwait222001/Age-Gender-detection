import cv2
import numpy as np

# Model files and mean values
FACE_PROTO = r"C:\Users\Admin\Desktop\pdfprint1\github folder 1\opencv_face_detector (1).pbtxt"
FACE_MODEL = r"C:\Users\Admin\Desktop\pdfprint1\github folder 1\opencv_face_detector_uint8 (1).pb"
GENDER_PROTO = r"C:\Users\Admin\Desktop\pdfprint1\github folder 1\gender_deploy (1).prototxt"
GENDER_MODEL = r"C:\Users\Admin\Desktop\pdfprint1\github folder 1\gender_net (1).caffemodel"
AGE_PROTO = r"C:\Users\Admin\Desktop\pdfprint1\github folder 1\age_deploy (1).prototxt"
AGE_MODEL = r"C:\Users\Admin\Desktop\pdfprint1\github folder 1\age_net (1).caffemodel"

# Load models
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

# Mean values for normalization
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDERS = ['Male', 'Female']
AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Function to detect faces
def detect_faces(frame, confidence_threshold=0.7):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            face_boxes.append(box.astype(int))

    return face_boxes

# Function to predict gender
def predict_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    return GENDERS[gender_preds[0].argmax()]

# Function to predict age
def predict_age(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    return AGE_RANGES[age_preds[0].argmax()]

# Load image
image_path = r"C:\Users\Admin\Desktop\pdfprint1\github folder 1\input.jpg"
output_path = r"C:\Users\Admin\Desktop\pdfprint1\github folder 1\output.jpg"  # Path to save the result
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Image not found!")
else:
    # Resize the frame to 500x500
    frame = cv2.resize(frame, (500, 500))

    face_boxes = detect_faces(frame)

    if len(face_boxes) == 0:
        print("No faces detected.")
    else:
        for idx, (startX, startY, endX, endY) in enumerate(face_boxes):
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.resize(face, (227, 227))
            gender = predict_gender(face)
            age = predict_age(face)

            # Print the detected age in console
            print(f"Face {idx+1}: Age - {age}")

            # Draw a rectangle around the face
            color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Add gender label above the face
            label = f"{gender}"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save the result image
    cv2.imwrite(output_path, frame)
    print(f"Output image saved at: {output_path}")

    # Show the processed image
    cv2.imshow('Gender Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
