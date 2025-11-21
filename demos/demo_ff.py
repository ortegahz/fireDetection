from collections import deque

import cv2

# Open the video file or capture device
cap = cv2.VideoCapture('/home/manu/mnt/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/jiu-shiwai/J-D-20m-003.mp4')

# Get the FPS from the video
FPS = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize a deque to store the previous frames (up to 3 seconds)
previous_frames = deque(maxlen=3 * FPS + 1)

# Create named windows for displaying the frames in full screen
cv2.namedWindow('Combined RGB', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)

# Set windows to full screen
cv2.setWindowProperty('Combined RGB', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Original Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, current_frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Add the current frame to the deque
    previous_frames.append(current_gray)

    # Only process the frame if we have enough history (3 seconds of frames)
    if len(previous_frames) >= 3 * FPS + 1:
        # Calculate differences with 1-second, 2-second, and 3-second previous frames
        # diff_1s = cv2.absdiff(current_gray, previous_frames[-1 - FPS])
        # diff_2s = cv2.absdiff(current_gray, previous_frames[-1 - 2 * FPS])
        # diff_3s = cv2.absdiff(current_gray, previous_frames[-1 - 3 * FPS])
        diff_1s = previous_frames[-1 - FPS]
        diff_2s = previous_frames[-1 - 2 * FPS]
        diff_3s = previous_frames[-1 - 3 * FPS]

        # Convert each difference to a 3-channel image
        diff_1s_rgb = cv2.cvtColor(diff_1s, cv2.COLOR_GRAY2BGR)
        diff_2s_rgb = cv2.cvtColor(diff_2s, cv2.COLOR_GRAY2BGR)
        diff_3s_rgb = cv2.cvtColor(diff_3s, cv2.COLOR_GRAY2BGR)

        # Combine them into a single RGB image (use diff_1s for R, diff_2s for G, diff_3s for B)
        combined_rgb = cv2.merge([diff_1s_rgb[:, :, 0], diff_2s_rgb[:, :, 1], diff_3s_rgb[:, :, 2]])

        # Display the combined RGB image
        cv2.imshow('Combined RGB', combined_rgb)

    # Display the original frame as well (optional)
    cv2.imshow('Original Frame', current_frame)

    # Breaking condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
