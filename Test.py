import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from autocorrect import Speller

# videocapture variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# bbox variables
offset = 20
imgSize = 300

# data collection variables
folder = "Data/C"
counter = 0
labels = ["A", "B", "C"]

# data prediction variables
predictions = []

# spell checker (not yet implemented)
spell = Speller()

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img,)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            predictions.append(labels[index])  # Append prediction to list
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            predictions.append(labels[index])  # Append prediction to list

        cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w + offset, y+h + offset), (255, 0, 255), 4)

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("image", imgOutput)

    key = cv2.waitKey(1)
    if key == ord('q'):
        pred_img = np.ones((200, 400, 3), np.uint8) * 255  # Create a blank image
        text_x = 30  # Initial x-coordinate for text
        text_y = 100  # y-coordinate for text
        count = 0  # Counter for consecutive same predictions
        prev_pred = None  # Variable to store the previous prediction
        for pred in predictions:
            if pred == prev_pred:
                count += 1
                if count >= 1:
                    continue  # Skip drawing if there are more than 3 consecutive same predictions
            else:
                count = 0  # Reset the counter if prediction changes
                prev_pred = pred  # Update previous prediction

            cv2.putText(pred_img, f"{pred}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            text_size = cv2.getTextSize(f"{pred}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x += text_size[0] + 10  # Increment x-coordinate for next prediction
            print(pred_img)  # print on console
        cv2.imshow("Predictions", pred_img)
        cv2.waitKey(1)

    if key == ord("s"):
        break

cv2.destroyAllWindows()
cap.release()