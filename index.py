import cv2
import argparse

# Define command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# Load YOLOv3 network
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Define the output layers of the network
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load the input image
image = cv2.imread(args["image"])

# Resize the image to fit the network input size
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)

# Set the blob as input to the network and perform a forward pass
net.setInput(blob)
layerOutputs = net.forward(ln)

# Initialize lists to store the detected faces and ages
faces = []
ages = []

# Loop over each of the layer outputs
for output in layerOutputs:
    # Loop over each detection in the output
    for detection in output:
        # Extract the confidence and class IDs
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # Check if the detected object is a face and if the confidence is high enough
        if classID == 0 and confidence > 0.5:
            # Compute the coordinates of the bounding box
            box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (x, y, w, h) = box.astype("int")

            # Extract the face ROI and append it to the list of faces
            face = image[y:y+h, x:x+w]
            faces.append(face)

            # Use a pre-trained model to estimate the age of the detected face
            age_model = models.load_model('age_model.h5')
            age = age_model.predict(face)
            age = int(age)

            # Append the age to the list of ages
            ages.append(age)

# Display the input image with the detected faces and ages
for i, face in enumerate(faces):
    cv2.putText(image, str(ages[i]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
