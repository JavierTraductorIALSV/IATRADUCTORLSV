
    # Reconocimiento automático para: SIEMPRE
    def detectar_siempre(self, landmarks):
        """Detectar seña: SIEMPRE"""
        # Implementar lógica específica basada en los landmarks
        
        # Ejemplo básico - verificar si hay landmarks de manos
        if 'manos' in landmarks and len(landmarks['manos']) > 0:
            # Aquí iría la lógica de detección específica
            # basada en los datos recolectados
            
            # Por ahora, un placeholder que siempre retorna False
            # Debes implementar la lógica real basada en tus datos
            return False
        
        return False
        