
def detectar_facil_20250919_02(self, landmarks):
    """Detectar seña: FACIL_20250919_02"""
    try:
        # Verificar si hay landmarks de manos
        if 'manos' not in landmarks or not landmarks['manos'] or not landmarks['manos'][0]:
            return False
            
        mano_data = landmarks['manos'][0]
        
        # Verificar que tenemos suficientes puntos (21 puntos * 3 coordenadas)
        if len(mano_data) < 21 * 3:
            return False
            
        # Obtener coordenadas de puntos clave (como en tu ejemplo)
        # Puntos de referencia: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
        landmarks_dict = {}
        for i in range(21):
            x = mano_data[i*3]
            y = mano_data[i*3 + 1]
            z = mano_data[i*3 + 2]
            landmarks_dict[i] = (x, y, z)
        
        # Obtener puntos específicos (como en tu ejemplo)
        wrist = landmarks_dict[0]          # Muñeca
        thumb_tip = landmarks_dict[4]      # Punta del pulgar
        index_tip = landmarks_dict[8]      # Punta del índice
        middle_tip = landmarks_dict[12]    # Punta del medio
        ring_tip = landmarks_dict[16]      # Punta del anular
        pinky_tip = landmarks_dict[20]     # Punta del meñique
        
        thumb_pip = landmarks_dict[2]      # Base del pulgar
        index_pip = landmarks_dict[6]      # Base del índice
        middle_pip = landmarks_dict[10]    # Base del medio
        ring_pip = landmarks_dict[14]      # Base del anular
        pinky_pip = landmarks_dict[18]     # Base del meñique
        
        # DETECCIÓN DE GESTOS ESPECÍFICOS (como en tu ejemplo)
        
        # Gesto "BIEN" (pulgar arriba, otros dedos abajo)
        if (thumb_pip[1] > thumb_tip[1] and    # Pulgar extendido hacia arriba
            index_pip[1] < index_tip[1] and    # Índice flexionado hacia abajo
            middle_pip[1] < middle_tip[1] and  # Medio flexionado hacia abajo
            ring_pip[1] < ring_tip[1] and      # Anular flexionado hacia abajo
            pinky_pip[1] < pinky_tip[1]):      # Meñique flexionado hacia abajo
            return True
        
        # Gesto "MAL" (pulgar abajo, otros dedos arriba)
        if (thumb_pip[1] < thumb_tip[1] and    # Pulgar flexionado hacia abajo
            index_pip[1] > index_tip[1] and    # Índice extendido hacia arriba
            middle_pip[1] > middle_tip[1] and  # Medio extendido hacia arriba
            ring_pip[1] > ring_tip[1] and      # Anular extendido hacia arriba
            pinky_pip[1] > pinky_tip[1]):      # Meñique extendido hacia arriba
            return True
        
        # Gesto "TE AMO" (pulgar, índice y meñique extendidos)
        if (thumb_pip[1] > thumb_tip[1] and    # Pulgar extendido
            index_pip[1] > index_tip[1] and    # Índice extendido
            middle_pip[1] < middle_tip[1] and  # Medio flexionado
            ring_pip[1] < ring_tip[1] and      # Anular flexionado
            pinky_pip[1] > pinky_tip[1]):      # Meñique extendido
            return True
            
        # Para otras señas, implementa lógica específica aquí
        # Basado en la posición relativa de los dedos
        
        return False
        
    except Exception as e:
        print(f"Error en detección de FACIL_20250919_02: {e}")
        return False
