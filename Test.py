import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from autocorrect import Speller

# Video capture variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Bbox variables
offset = 20
imgSize = 300

# Data collection variables
folder = "Data/C"
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Data prediction variables
predictions = []
predictions_array = []

# Spell checker (not yet implemented)
spell = Speller()

# Timer variables
start_time = time.time()
last_save_time = start_time  # Initialize last save time
letter_appearance_time = 3  # Time interval for each letter appearance in seconds
next_letter_time = start_time + letter_appearance_time  # Initialize next letter appearance time
stop_time = False  # Flag to stop time counting
stop_spelling = False  # Flag to stop spelling letters
spell_box_color = (0, 255, 0)  # Green color for spelled box
time_box_color = (0, 255, 0)  # Green color for time indicator box

while True:
    success, img = cap.read()
    if not success:  # Check if frame is successfully captured
        continue  # Skip the current iteration if frame is not successfully captured
    imgOutput = img.copy()
    hands, img = detector.findHands(img, )
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        current_time = time.time()  # Get the current time

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if not stop_spelling:  # Check if spelling is not stopped
                if not stop_time:  # Check if time counting is not stopped
                    if 0 <= index < len(labels):  # Check if the index is within the valid range
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
            if not stop_spelling:  # Check if spelling is not stopped
                if not stop_time:  # Check if time counting is not stopped
                    if 0 <= index < len(labels):  # Check if the index is within the valid range
                        predictions.append(labels[index])  # Append prediction to list
                        print(prediction, index)

        if current_time - last_save_time >= 3:  # Check if 3 seconds have passed since the last save
            last_save_time = current_time  # Update the last save time
            if not stop_spelling:  # Check if spelling is not stopped
                if not stop_time:  # Check if time counting is not stopped
                    predictions_array.append(predictions[-1])  # Save the latest prediction
                    print("Saved prediction to predictions_array")
                    next_letter_time = current_time + letter_appearance_time  # Update next letter appearance time

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255),
                      cv2.FILLED)
        if 0 <= index < len(labels):  # Check if the index is within the valid range
            cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Display spelled letters at the top of the window
        word = ''.join(predictions_array)
        spelled_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        spelled_x = (img.shape[1] - spelled_size[0]) // 2
        spelled_y = 50
        cv2.rectangle(imgOutput, (spelled_x - 10, spelled_y - 10), (spelled_x + spelled_size[0] + 10, spelled_y + spelled_size[1] + 10), spell_box_color, -1)
        cv2.putText(imgOutput, word, (spelled_x, spelled_y + spelled_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display timer for next letter appearance
        remaining_time = int(next_letter_time - current_time) if not stop_spelling else letter_appearance_time
        timer_text = f"Time: {remaining_time}s"
        timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        timer_x = 20  # Position time indicator to the left side
        timer_y = 100
        time_box_color = (0, 0, 255) if stop_spelling else (0, 255, 0)  # Change box color based on spelling state
        cv2.rectangle(imgOutput, (timer_x - 10, timer_y - 10), (timer_x + timer_size[0] + 10, timer_y + timer_size[1] + 10), time_box_color, -1)
        cv2.putText(imgOutput, timer_text, (timer_x, timer_y + timer_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("image", imgOutput)

    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
    elif key == ord("s"):
        stop_time = not stop_time  # Toggle stop time flag
        if stop_time:
            last_save_time = time.time()  # Save the current time when stopping
    elif key == ord("e"):  # Pause or resume spelling
        stop_time = False
        last_save_time = time.time()  # Save the current time when resuming
        next_letter_time = last_save_time + letter_appearance_time
        stop_spelling = not stop_spelling  # Toggle stop spelling flag
        spell_box_color = (0, 0, 255) if stop_spelling else (0, 255, 0)  # Change box color based on spelling state
    elif key == ord("r") and stop_spelling:  # Erase the last letter if spelling is paused
        if predictions_array:  # Check if there are spelled letters to erase
            predictions_array.pop()  # Remove the last letter from the spelled letters