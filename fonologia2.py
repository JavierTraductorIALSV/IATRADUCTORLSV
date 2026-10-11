import cv2
import mediapipe as mp
import numpy as np

def Etiqueta(idx, mano, results, image_width, image_height):
    aux = None
    for _, clase in enumerate(results.multi_handedness):
        if clase.classification[0].index == idx:
            label = clase.classification[0].label
            texto = '{}'.format(label)

            coords = tuple(np.multiply(
                np.array((mano.landmark[mp_hands.HandLandmark.WRIST].x,
                         mano.landmark[mp_hands.HandLandmark.WRIST].y)),
                [image_width, image_height]).astype(int))
            
            aux = texto, coords
    return aux

def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Configurar resolución de la cámara para mejor calidad
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Diccionario para nombres de dedos
nombres_dedos = {
    4: "PUG",
    8: "1",
    12: "2",
    16: "3",
    20: "4"
}

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2) as hands:
    
    change = False
    change2 = False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Obtener dimensiones de la imagen
        image_height, image_width, _ = image.shape
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Obtener puntos de referencia de los dedos
                index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                                    int(hand_landmarks.landmark[8].y * image_height))
               
                index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                                    int(hand_landmarks.landmark[6].y * image_height))
                
                thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                             int(hand_landmarks.landmark[4].y * image_height))
                thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                             int(hand_landmarks.landmark[2].y * image_height))
                
                middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                                     int(hand_landmarks.landmark[12].y * image_height))
                middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                                     int(hand_landmarks.landmark[10].y * image_height))
                
                ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                                   int(hand_landmarks.landmark[16].y * image_height))
                ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                                   int(hand_landmarks.landmark[14].y * image_height))
                
                pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                             int(hand_landmarks.landmark[20].y * image_height))
                pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                             int(hand_landmarks.landmark[18].y * image_height))
                
                wrist = (int(hand_landmarks.landmark[0].x * image_width),
                         int(hand_landmarks.landmark[0].y * image_height))
                
                # Mostrar números de dedos cerca de sus puntas (sin voltear)
                for landmark_id, nombre in nombres_dedos.items():
                    landmark = hand_landmarks.landmark[landmark_id]
                    coord_x = int(landmark.x * image_width)
                    coord_y = int(landmark.y * image_height)
                    
                    # Mostrar el número/PUG en negro, sin fondo
                    cv2.putText(image, nombre, (coord_x + 10, coord_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
                
                # Detección de manos para diálogo
                etiqueta_info = Etiqueta(num, hand_landmarks, results, image_width, image_height)
                if etiqueta_info and len(results.multi_hand_landmarks) == 2:
                    text, coords = etiqueta_info
                    print(text, coords)
                    
                    if text == "Right":
                        text = "IZQUIERDA"
                        index_finger_tip_r = (int(hand_landmarks.landmark[8].x * image_width),
                                              int(hand_landmarks.landmark[8].y * image_height))
                        change = True
                    
                    if text == "Left":
                        text = "DERECHA"
                        index_finger_tip_l = (int(hand_landmarks.landmark[8].x * image_width),
                                              int(hand_landmarks.landmark[8].y * image_height))
                        wrist = (int(hand_landmarks.landmark[0].x * image_width),
                                 int(hand_landmarks.landmark[0].y * image_height))
                        change2 = True

                        if change2 and change:
                            if distancia_euclidiana(index_finger_tip_l, wrist) < 170.0:
                                cv2.putText(image, "Me puedes decir la hora?", (100, 150),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            3.0, (0, 0, 255), 6)
                                cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 
                                           1, (255, 255, 255), 2, cv2.LINE_AA)

        # Voltear la imagen horizontalmente para selfie-view
        flipped_image = cv2.flip(image, 1)
        
        # Dibujar leyendas en la parte inferior (después de voltear)
        cv2.putText(flipped_image, "LEYENDAS: 1=INDICE 2=MEDIO 3=ANULAR 4=MEÑIQUE PUG=PULGAR", 
                   (10, image_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Redimensionar la imagen para que sea más grande
        scale_percent = 150  # 150% del tamaño original
        width = int(flipped_image.shape[1] * scale_percent / 100)
        height = int(flipped_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(flipped_image, dim, interpolation=cv2.INTER_AREA)
        
        # Crear una ventana con tamaño ajustable
        cv2.namedWindow('MediaPipe Hands - Reconocimiento de Dedos', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MediaPipe Hands - Reconocimiento de Dedos', width, height)
        
        cv2.imshow('MediaPipe Hands - Reconocimiento de Dedos', resized_image)
        
        # Salir con ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()