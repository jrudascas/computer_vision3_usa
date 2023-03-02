import numpy as np
import cv2 as cv

# Se lee el archivo de video
cap = cv.VideoCapture(cv.samples.findFile('./Electiva_III/videos/vtest.mp4'))

# Se lee el primer frame del video
ret, frame1 = cap.read()

# Se convierte a escala de grises
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# Se crea una matriz con las mismas dimensiones que el frame
hsv = np.zeros_like(frame1) 
# print(hsv.ndim) # 3 dim filas,columnas, color

# Se establece la saturación en 255
hsv[..., 1] = 255 #Al utilizar ... en lugar de especificar las tres dimensiones completas de la matriz, 
                  #se está indicando que se conserven las dos primeras dimensiones (altura y ancho), 
                  # pero que se modifique el valor de la tercera dimensión (color) en su totalidad.
                  #255==> Por lo tanto, el píxel se verá más brillante y más intenso en su color.

# Se entra en un ciclo que procesa cada frame del video
cont = 0
while(1):
    # Se lee el siguiente frame
    ret, frame2 = cap.read()

    # Si no hay más frames, se sale del ciclo
    if not ret:
        print(cont)
        print('No frames grabbed!')
        
        break
    

    # Se convierte el frame a escala de grises
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Se calcula el flujo óptico mediante el método Farneback
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0) # (frame1, frame2, None => Mask, Piramide)

    # Se calculan la magnitud y el ángulo del flujo óptico
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1]) #mag cantidad de movimiento entre dos frames consecutivos en cada punto de la imagen
                                                          #ang direccion movimiento
    # Se convierte el ángulo de radianes a grados
    hsv[..., 0] = ang*180/np.pi/2

    # Se normaliza la magnitud del flujo óptico y se establece como el componente de brillo en la matriz HSV
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Se convierte la matriz HSV a BGR para visualizarla
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    

    # Se muestra el frame con el flujo óptico calculado
    cv.imshow('frame2', bgr)

    # Se espera 30 milisegundos y se espera por una tecla
    k = cv.waitKey(30) & 0xff

    cont += 1

    # Si se presiona ESC, se sale del ciclo
    if k == 27:
        print(cont)
        break

    # Si se presiona 's', se guardan los frames actual y de flujo óptico en archivos PNG
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)

    # El frame actual se establece como el frame anterior para el siguiente ciclo
    prvs = next

# Se cierran todas las ventanas abiertas
cv.destroyAllWindows()