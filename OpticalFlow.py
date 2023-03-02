import numpy as np
import cv2 as cv

cap = cv.VideoCapture("./vtest.avi.mp4")

#params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15), # size of the search window at each pyramid level.
                  maxLevel = 2, # 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on.
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) # parameter, specifying the termination criteria of the iterative search algorithm.
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
# Convert the first frame to gray scale
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# Extract the key points Shi-Tomashi Corner Detection
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# print(type(p0))
# print(p0.shape)
print("p0", p0)
# p0 = np.array([[[310, 115]], [[380,165]], [[140,145]], [[435,175]], [[460,185]]], dtype=np.float32)
# np.array(p0)
# print(type(p0))
# print(p0.shape)
# print("p0", p01)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    # Create all the frames from the video
    ret, frame = cap.read()
    # Condition that is generate at the end of the video
    if not ret:
        print('No frames grabbed!')
        break
    # Convert in gray all the frames
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow with the previos and current frame, the initial key points,
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel() # Create a complete list join the small ones
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        #start_point, end_point, color, thickness
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        # image, center_coordinates, radius, color, thickness
    img = cv.add(frame, mask) # sum of arrays
    cv.imshow("Frame", img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2) # The new shape should be compatible with the original shape
# #cv.destroyAllWindows()