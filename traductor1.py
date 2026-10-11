import cv2
import mediapipe as mp
import numpy as np
import time

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

# Configurar para pantalla completa
cv2.namedWindow('Detección de Lengua de Señas - Presiona ESC para salir', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Detección de Lengua de Señas - Presiona ESC para salir', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Variables para subtítulos
current_caption = ""
caption_timer = 0
caption_duration = 30  # Número de frames que se mostrará el subtítulo

# Variables para FPS
prev_time = 0
fps = 0

# Colores
COLORS = {
    'BIEN': (0, 255, 0),
    'MAL': (0, 0, 255),
    'TE AMO': (255, 0, 0),
    'DEFAULT': (255, 255, 255)
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

        # Calcular FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Voltear la imagen horizontalmente para selfie-view desde el principio
        image = cv2.flip(image, 1)
        
        # Obtener dimensiones de la imagen
        image_height, image_width, _ = image.shape
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Dibujar área de subtítulos (barra inferior)
        caption_bar_height = 80
        cv2.rectangle(image, (0, image_height - caption_bar_height), 
                     (image_width, image_height), (0, 0, 0), -1)
        cv2.rectangle(image, (0, image_height - caption_bar_height), 
                     (image_width, image_height), (255, 255, 255), 2)

        # Dibujar información de FPS
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Dibujar instrucciones
        cv2.putText(image, 'Presiona ESC para salir', (image_width - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Obtener puntos de referencia de los dedos (coordenadas ya volteadas)
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
                
                # Detectar gestos y establecer subtítulos
                # Gesto "BIEN"
                if (thumb_pip[1] - thumb_tip[1] > 0 and 
                    thumb_pip[1] - index_finger_tip[1] < 0 and
                    thumb_pip[1] - middle_finger_tip[1] < 0 and 
                    thumb_pip[1] - ring_finger_tip[1] < 0 and
                    thumb_pip[1] - pinky_tip[1] < 0):
                    current_caption = "BIEN"
                    caption_timer = caption_duration
                    cv2.putText(image, 'BIEN', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0, COLORS['BIEN'], 4)
                
                # Gesto "MAL"
                elif (thumb_pip[1] - thumb_tip[1] < 0 and 
                      thumb_pip[1] - index_finger_tip[1] > 0 and
                      thumb_pip[1] - middle_finger_tip[1] > 0 and 
                      thumb_pip[1] - ring_finger_tip[1] > 0 and
                      thumb_pip[1] - pinky_tip[1] > 0):
                    current_caption = "MAL"
                    caption_timer = caption_duration
                    cv2.putText(image, 'MAL', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0, COLORS['MAL'], 4)
                
                # Gesto "TE AMO" (simplificado)
                elif (thumb_pip[1] - thumb_tip[1] > 0 and 
                      index_finger_pip[1] - index_finger_tip[1] > 0 and
                      pinky_pip[1] - pinky_tip[1] > 0):
                    current_caption = "TE AMO"
                    caption_timer = caption_duration
                    cv2.putText(image, 'TE AMO', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0, COLORS['TE AMO'], 4)
                
                # Detección de manos para diálogo
                etiqueta_info = Etiqueta(num, hand_landmarks, results, image_width, image_height)
                if etiqueta_info and len(results.multi_hand_landmarks) == 2:
                    text, coords = etiqueta_info
                    
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
                                current_caption = "Me puedes decir la hora?"
                                caption_timer = caption_duration

        # Mostrar subtítulo actual en la barra inferior
        if caption_timer > 0:
            # Calcular posición centrada para el texto
            text_size = cv2.getTextSize(current_caption, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (image_width - text_size[0]) // 2
            text_y = image_height - (caption_bar_height // 2) + 10
            
            # Dibujar texto del subtítulo
            cv2.putText(image, current_caption, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS.get(current_caption.split()[0], COLORS['DEFAULT']), 3)
            
            # Reducir el temporizador
            caption_timer -= 1
        else:
            current_caption = ""  # Limpiar subtítulo cuando el temporizador expire

        # Mostrar la imagen (ya no es necesario voltearla al final)
        cv2.imshow('Detección de Lengua de Señas - Presiona ESC para salir', image)
        
        # Salir con ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()