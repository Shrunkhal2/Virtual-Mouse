import cv2
import numpy as np
import HandTrackingModule as htm  # Ensure this module exists
import time
import autopy  # For controlling the mouse pointer

##########################
wCam, hCam = 640, 480  # Camera resolution
frameR = 100  # Frame reduction for smoother control
smoothening = 7  # Smoothening factor for mouse movement
##########################

pTime = 0
plocX, plocY = 0, 0  # Previous locations of the mouse pointer
clocX, clocY = 0, 0  # Current locations of the mouse pointer

# Camera setup
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
cap.set(3, wCam)  # Set width
cap.set(4, hCam)  # Set height

# Hand detector
detector = htm.handDetector(maxHands=1)

# Get screen size
wScr, hScr = autopy.screen.size()

while True:
    # 1. Capture the frame and find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Check if landmarks are detected
    if len(lmList) != 0:
        # Get coordinates of the index and middle finger tips
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # Draw a rectangle for the frame boundary
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # 4. Moving Mode: Only the index finger is up
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates to screen size
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen mouse movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move the mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # Update previous location
            plocX, plocY = clocX, clocY

        # 5. Clicking Mode: Both index and middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between the two fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # Perform click if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 6. Frame Rate Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # 7. Display the frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()