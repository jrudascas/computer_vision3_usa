import numpy as np
import cv2 as cv

# Parámetros de detección de características
feature_params = dict(maxCorners=20, qualityLevel=0.3, minDistance=7, blockSize=7)

cap = cv.VideoCapture(cv.samples.findFile('./Electiva_III/videos/vtest.mp4'))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

frame_count = 0
while True:
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_count += 1
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Detectar características en la imagen anterior
    prev_points = cv.goodFeaturesToTrack(prvs, mask=None, **feature_params)

    # Calcular flujo óptico denso
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calcular ángulo y magnitud del flujo
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Dibujar los puntos de características en la imagen
    for point in prev_points:
        x, y = point.ravel()
        cv.circle(hsv, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Convertir la imagen HSV a BGR y mostrarla
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('Optical_Flow', bgr)
    if frame_count == 100:
        concatenated_image = cv.hconcat([frame2, bgr])
        cv.imwrite('./Electiva_III/imagenes/OpticalFlow_fr_{}.png'.format(frame_count), concatenated_image)
        
    cv.imshow('Optical_Flow', bgr)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = next

cv.destroyAllWindows()
