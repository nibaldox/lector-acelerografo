import numpy as np
from scipy import signal

class SignalFilter:
    def __init__(self, sampling_rate):
        """
        Inicializa el filtro
        Args:
            sampling_rate: Frecuencia de muestreo en Hz
        """
        self.fs = sampling_rate
        
    def butter_lowpass(self, cutoff, order=4):
        """
        Diseña un filtro pasa bajos Butterworth
        Args:
            cutoff: Frecuencia de corte en Hz
            order: Orden del filtro
        Returns:
            b, a: Coeficientes del filtro
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_highpass(self, cutoff, order=4):
        """
        Diseña un filtro pasa altos Butterworth
        Args:
            cutoff: Frecuencia de corte en Hz
            order: Orden del filtro
        Returns:
            b, a: Coeficientes del filtro
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_bandpass(self, lowcut, highcut, order=4):
        """
        Diseña un filtro pasa banda Butterworth
        Args:
            lowcut: Frecuencia de corte inferior en Hz
            highcut: Frecuencia de corte superior en Hz
            order: Orden del filtro
        Returns:
            b, a: Coeficientes del filtro
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        return b, a

    def apply_filter(self, data, filter_type='lowpass', **kwargs):
        """
        Aplica el filtro seleccionado a los datos
        Args:
            data: Array de datos a filtrar
            filter_type: Tipo de filtro ('lowpass', 'highpass', 'bandpass')
            **kwargs: Argumentos específicos del filtro (cutoff, order, etc.)
        Returns:
            filtered_data: Datos filtrados
        """
        # Eliminar tendencia lineal
        detrended = signal.detrend(data)
        
        # Aplicar el filtro seleccionado
        if filter_type == 'lowpass':
            cutoff = kwargs.get('cutoff', 10.0)
            order = kwargs.get('order', 4)
            b, a = self.butter_lowpass(cutoff, order)
            
        elif filter_type == 'highpass':
            cutoff = kwargs.get('cutoff', 0.1)
            order = kwargs.get('order', 4)
            b, a = self.butter_highpass(cutoff, order)
            
        elif filter_type == 'bandpass':
            lowcut = kwargs.get('lowcut', 0.1)
            highcut = kwargs.get('highcut', 10.0)
            order = kwargs.get('order', 4)
            b, a = self.butter_bandpass(lowcut, highcut, order)
        else:
            raise ValueError(f"Tipo de filtro no soportado: {filter_type}")

        # Aplicar filtro con fase cero (forward-backward)
        filtered_data = signal.filtfilt(b, a, detrended)
        
        return filtered_data

    def get_filter_response(self, filter_type='lowpass', **kwargs):
        """
        Obtiene la respuesta en frecuencia del filtro
        Args:
            filter_type: Tipo de filtro
            **kwargs: Argumentos específicos del filtro
        Returns:
            w: Frecuencias normalizadas
            h: Respuesta en frecuencia
        """
        if filter_type == 'lowpass':
            b, a = self.butter_lowpass(kwargs.get('cutoff', 10.0), kwargs.get('order', 4))
        elif filter_type == 'highpass':
            b, a = self.butter_highpass(kwargs.get('cutoff', 0.1), kwargs.get('order', 4))
        elif filter_type == 'bandpass':
            b, a = self.butter_bandpass(
                kwargs.get('lowcut', 0.1),
                kwargs.get('highcut', 10.0),
                kwargs.get('order', 4)
            )
        else:
            raise ValueError(f"Tipo de filtro no soportado: {filter_type}")
            
        w, h = signal.freqz(b, a)
        # Convertir frecuencias normalizadas a Hz
        freqs = w * self.fs / (2 * np.pi)
        return freqs, np.abs(h)
