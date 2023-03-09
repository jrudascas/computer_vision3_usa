import cv2
import dlib
import numpy as np

# Función para extraer descriptores de rostros
def extract_face_descriptors(image): # reciba arreglo de arreglos en Numpy

    # # # Cargar imagen
    # image = cv2.imread(image)

    # Inicializar detector de rostros y puntos faciales
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./Electiva_III/taller_8/shape_predictor_68_face_landmarks.dat")

    
    faces = detector(image, 1)

    # Verificar si se detectó un rostro en la imagen
    if len(faces) > 0:
        # Obtener puntos faciales del primer rostro detectado
        landmarks = predictor(image, faces[0])

        # Extraer descriptores de rostro
        descriptors = []

        # 1. Distancia entre los ojos ok
        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
        eye_distance = np.linalg.norm(left_eye - right_eye)
        descriptors.append(eye_distance)

        # 2. Anchura de la cara
        face_width = np.linalg.norm(np.array([landmarks.part(16).x, landmarks.part(16).y]) - np.array([landmarks.part(0).x, landmarks.part(0).y]))
        descriptors.append(face_width)

        # 3. Altura de la cara
        face_height = np.linalg.norm(np.array([landmarks.part(8).x, landmarks.part(8).y]) - np.array([landmarks.part(27).x, landmarks.part(27).y]))
        descriptors.append(face_height)

        # 4. Distancia entre la nariz y la barbilla
        nose_to_chin = np.linalg.norm(np.array([landmarks.part(33).x, landmarks.part(33).y]) - np.array([landmarks.part(8).x, landmarks.part(8).y]))
        descriptors.append(nose_to_chin)

        # 5. Distancia entre las cejas
        brow_distance = np.linalg.norm(np.array([landmarks.part(17).x, landmarks.part(17).y]) - np.array([landmarks.part(26).x, landmarks.part(26).y]))
        descriptors.append(brow_distance)

        # 6. Distancia entre la boca y la nariz
        nose_to_mouth = np.linalg.norm(np.array([landmarks.part(33).x, landmarks.part(33).y]) - np.array([landmarks.part(51).x, landmarks.part(51).y]))
        descriptors.append(nose_to_mouth)

        # 7. Distancia entre las comisuras de la boca
        mouth_width = np.linalg.norm(np.array([landmarks.part(48).x, landmarks.part(48).y]) - np.array([landmarks.part(54).x, landmarks.part(54).y]))
        descriptors.append(mouth_width)

        # 8. Distancia entre el labio superior y el inferior
        mouth_height = np.linalg.norm(np.array([landmarks.part(62).x, landmarks.part(62).y]) - np.array([landmarks.part(66).x, landmarks.part(66).y]))
        descriptors.append(mouth_height)
                                                                                                                                            
        # 9. Distancia entre las aletas de la nariz
        nose_wings = np.linalg.norm(np.array([landmarks.part(31).x, landmarks.part(31).y]) - np.array([landmarks.part(35).x, landmarks.part(35).y]))
        descriptors.append(nose_wings)

        # 10. Relación entre la altura de la cara y la distancia entre las cejas
        face_height_brow_distance_ratio = face_height / brow_distance
        descriptors.append(face_height_brow_distance_ratio)

        # cv2.circle(image, left_eye, 2, (0, 0, 255), -1)
        # cv2.circle(image, right_eye, 2, (0, 0, 255), -1)

        # Pintar los descriptores en la imagen
        # Pintar landmarks en la imagen
        # for i in range(68):
        #     x = landmarks.part(i).x
        #     y = landmarks.part(i).y
        #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


        # # cv2.circle(image, (int(x+w*0.75), int(y+h*0.25)), 2, (0, 0, 0), -1)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return descriptors

    else:
        # Si no se detectó ningún rostro, retornar una lista vacía
        return [0,0,0,0,0,0,0,0,0,0]

# extract_face_descriptors('./Electiva_III/imagenes/05_hombre.png')


