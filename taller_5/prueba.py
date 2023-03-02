import cv2

# Abrir el archivo de video
cap = cv2.VideoCapture('./Electiva_III/videos/vtest.mp4')

# Obtener el número de frames en el video
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Cerrar el archivo de video
cap.release()

print('El video tiene {} frames.'.format(num_frames))
