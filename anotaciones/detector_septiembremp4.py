
def detectar_septiembremp4(self, landmarks):
    """Detectar seña: SEPTIEMBRE.mp4"""
    try:
        # Verificar si hay landmarks de manos
        if 'manos' not in landmarks or not landmarks['manos']:
            return False
            
        mano_data = landmarks['manos'][0]
        
        # ANÁLISIS BÁSICO DE LA POSICIÓN DE LA MANO
        # Esto es un ejemplo - debes ajustarlo según el gesto real
        
        # Obtener coordenadas de puntos clave
        # Dedos: pulgar (4), índice (8), medio (12), anular (16), meñique (20)
        if len(mano_data) >= 21 * 3:  # 21 puntos * 3 coordenadas
            # Ejemplo: detectar si la mano está abierta
            # Calculamos distancias entre dedos
            puntas_dedos = [8, 12, 16, 20]  # Índice, medio, anular, meñique
            bases_dedos = [6, 10, 14, 18]   # Bases de los dedos
            
            dedos_extendidos = 0
            for i in range(4):
                # Coordenadas de la punta del dedo
                tip_idx = puntas_dedos[i] * 3
                # Coordenadas de la base del dedo
                base_idx = bases_dedos[i] * 3
                
                # Verificar si el dedo está extendido (punta más alta que base)
                if tip_idx + 1 < len(mano_data) and base_idx + 1 < len(mano_data):
                    if mano_data[tip_idx + 1] < mano_data[base_idx + 1]:  # Y coordinate (inverted)
                        dedos_extendidos += 1
            
            # Para este ejemplo, consideramos que es la seña si hay al menos 3 dedos extendidos
            if dedos_extendidos >= 3:
                return True
                
        return False
        
    except Exception as e:
        print(f"Error en detección de SEPTIEMBRE.mp4: {e}")
        return False
