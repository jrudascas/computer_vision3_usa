import numpy as np
import cv2 as cv



cap = cv.VideoCapture(cv.samples.findFile('./Electiva_III/videos/vtest.mp4'))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255 # saturacion y brillo para que  sea el maximo 

# Seleccionar la región de interés en el primer fotograma
r, h, c, w = 122, 100, 341, 70  #1=> 120, 120, 350, 70 #2=> 157, 110, 371, 70 # 3=> 122, 100, 341, 70 
track_window = (c, r, w, h)
roi = frame1[r:r+h, c:c+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Aplicar MeanShift en cada uno de los fotogramas
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

frame_count = 0
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_count += 1
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)  
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Aplicar MeanShift en la ROI
    hsv = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    ret, track_window = cv.meanShift(dst, track_window, term_crit)

    # Dibujar el rectángulo en la imagen bgr
    x, y, w, h = track_window
    cv.rectangle(bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)

    if frame_count == 100:
        concatenated_image = cv.hconcat([frame2, bgr])
        cv.imwrite('./Electiva_III/imagenes/meanshift_fr_{}.png'.format(frame_count), concatenated_image)


    cv.imshow('MeanShift', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('./Electiva_III/imagenes/opticalfb1_mean.png', frame2)
        cv.imwrite('./Electiva_III/imagenes/opticalhsv1_mean.png', bgr)
    prvs = next
cv.destroyAllWindows()
