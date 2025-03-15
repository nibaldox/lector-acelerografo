import numpy as np
from scipy import signal

class EventDetector:
    def __init__(self, sampling_rate):
        """
        Inicializa el detector de eventos
        Args:
            sampling_rate: Frecuencia de muestreo en Hz
        """
        self.sampling_rate = sampling_rate
        
    def sta_lta(self, data, sta_window=1.0, lta_window=10.0, trigger_ratio=3.0):
        """
        Implementa el algoritmo STA/LTA (Short Time Average / Long Time Average)
        Args:
            data: Array de datos
            sta_window: Ventana corta en segundos
            lta_window: Ventana larga en segundos
            trigger_ratio: Ratio de disparo STA/LTA
        Returns:
            triggers: Lista de tiempos donde se detectaron eventos
            sta_lta_ratio: Array con el ratio STA/LTA
        """
        # Convertir ventanas de segundos a muestras
        sta_samples = int(sta_window * self.sampling_rate)
        lta_samples = int(lta_window * self.sampling_rate)
        
        # Calcular energía de la señal
        energy = data ** 2
        
        # Calcular STA y LTA usando convolución
        sta = np.zeros_like(data)
        lta = np.zeros_like(data)
        
        # Calcular promedios usando convolución con padding
        sta_conv = np.convolve(energy, np.ones(sta_samples)/sta_samples, mode='same')
        lta_conv = np.convolve(energy, np.ones(lta_samples)/lta_samples, mode='same')
        
        # Asignar los resultados asegurando dimensiones iguales
        sta = sta_conv[:len(data)]
        lta = lta_conv[:len(data)]
        
        # Evitar división por cero y valores muy pequeños
        min_val = np.max(lta) * 1e-10
        lta[lta < min_val] = min_val
        
        # Calcular ratio STA/LTA
        ratio = np.zeros_like(data)
        ratio = sta / lta
        
        # Encontrar eventos donde el ratio supera el umbral
        triggers = []
        trigger_on = False
        # Manejar valores no válidos
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Aplicar suavizado para reducir falsos positivos
        ratio = np.convolve(ratio, np.ones(5)/5, mode='same')
        
        for i in range(len(ratio)):
            if ratio[i] > trigger_ratio:
                if not trigger_on:
                    triggers.append(i / self.sampling_rate)  # Convertir a segundos
                    trigger_on = True
            else:
                trigger_on = False
                
        return triggers, ratio
    
    def peak_detection(self, data, threshold=None, distance=None):
        """
        Detecta picos en la señal que superan un umbral
        Args:
            data: Array de datos
            threshold: Umbral de amplitud (si es None, se usa 3*std)
            distance: Distancia mínima entre picos en segundos
        Returns:
            peaks: Índices de los picos detectados
            properties: Propiedades de los picos
        """
        if threshold is None:
            threshold = 3 * np.std(data)
            
        if distance is None:
            distance = int(0.5 * self.sampling_rate)  # 0.5 segundos por defecto
        else:
            distance = int(distance * self.sampling_rate)
            
        # Usar el valor absoluto de los datos y asegurar que son números reales
        abs_data = np.abs(np.real(data))
        
        # Detectar picos que superan el umbral
        peaks, properties = signal.find_peaks(
            abs_data,
            height=threshold,
            distance=distance
        )
        
        # Asegurar que se encontraron picos
        if len(peaks) == 0:
            return np.array([]), {'peak_heights': np.array([])}
            
        return peaks, properties
    
    def calculate_event_features(self, data, event_time, window=5.0):
        """
        Calcula características del evento
        Args:
            data: Array de datos
            event_time: Tiempo del evento en segundos
            window: Ventana de análisis en segundos
        Returns:
            features: Diccionario con características del evento
        """
        # Convertir tiempo a índices
        event_idx = int(event_time * self.sampling_rate)
        window_samples = int(window * self.sampling_rate)
        
        # Extraer ventana de datos
        start = max(0, event_idx - window_samples//2)
        end = min(len(data), event_idx + window_samples//2)
        event_data = data[start:end]
        
        # Calcular características
        features = {
            'peak_amplitude': np.max(np.abs(event_data)),
            'rms': np.sqrt(np.mean(event_data**2)),
            'duration': len(event_data) / self.sampling_rate,
            'energy': np.sum(event_data**2),
            'zero_crossings': len(np.where(np.diff(np.signbit(event_data)))[0])
        }
        
        return features
