## Introducción
----    
El algoritmo de density flow, también conocido como flujo denso, es una técnica utilizada en el procesamiento de imágenes que permite calcular el movimiento de los píxeles en un video. Este algoritmo es capaz de detectar la dirección y magnitud del movimiento en cada píxel, lo que resulta en una representación visual del flujo de movimiento en el video.


## Ejercicio
----
En este ejercicio se realiza la implementación de tres algoritmos de seguimiento de objetos en el contexto del algoritmo de density flow: MeanShift, CamShift y optical flow de Lucas-Kanade. Cada uno de estos algoritmos tiene una técnica única para realizar el seguimiento de los objetos en un video.

- MeanShift: Este algoritmo utiliza una ventana de tamaño fijo para realizar el seguimiento del objeto en un video. La ventana se desplaza en el video y se actualiza en función de la región de interés (ROI) para ajustarse a la posición actual del objeto.

- CamShift: Este algoritmo es una variante del algoritmo MeanShift que se adapta automáticamente al tamaño y orientación del objeto a medida que se mueve. Este algoritmo utiliza un modelo de histograma para realizar el seguimiento del objeto y ajustar automáticamente la ventana.

- Optical flow de Lucas-Kanade: Este algoritmo se utiliza para estimar el movimiento de los píxeles en un video. Este algoritmo utiliza una técnica de flujo óptico para calcular el movimiento de los píxeles en cada fotograma del video.

Para la implementación de estos algoritmos se utilizó el [Video](https://www.youtube.com/watch?v=oIFFnyD4TA0). Se aplicaron los tres algoritmos para realizar el seguimiento de las personas en el video y se comparó su rendimiento.


## Resultados
----
> Después de comparar los resultados de los tres algoritmos de seguimiento de objetos, se encontró que el algoritmo que se comportó mejor fue MeanShift. Al seleccionar unas coordenadas dentro de la imagen para el tracking, MeanShift logró acercarse al objeto del tracking dependiendo del tamaño de la ventana, mientras que CamShift nunca pudo moldear la ventana al objeto en movimiento aunque en algunas situaciones realizó el seguimiento. El algoritmo de Optical Flow de Lucas-Kanade también tuvo dificultades para seguir el objeto en movimiento debido a la presencia de objetos que se cruzaban entre ellos y solo logró detectar el objeto cuando no había interferencia de objetos. En conclusión, se encontró que MeanShift fue el algoritmo más adecuado para realizar el seguimiento de objetos en el video.

A continuación, se presentan las imágenes obtenidas de cada uno de los algoritmos comparados exactamente en el frame número 100 del video:

### Optical Flow
----
![](https://drive.google.com/uc?id=1vE3aTigAQVN2RGAF2fovE9FqySYMI8JQ)

### MeanShift
----
![](https://drive.google.com/uc?id=1MPrkXsW9LcGBwdVXmx1mZOXyaCsPwC09)

### CamShift
----
![](https://drive.google.com/uc?id=1D4c_IkXmYQXkixoZmEkeQq9yIUrr-BX6)
===
### Presentado por:
- Fredy Quesada Vivas
- Angello Perilla Ampudia