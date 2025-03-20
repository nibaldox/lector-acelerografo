import numpy as np
from scipy import signal
from logger import get_logger
import time as time_module

class SignalFilter:
    def __init__(self, sampling_rate):
        """
        Inicializa el filtro
        
        Args:
            sampling_rate (float): Frecuencia de muestreo en Hz
        """
        self.fs = sampling_rate
        self.logger = get_logger("SignalFilter")
        self.logger.info(f"Inicializando SignalFilter con frecuencia de muestreo {sampling_rate} Hz")
        
    def butter_lowpass(self, cutoff, order=4):
        """
        Diseña un filtro pasa bajos Butterworth
        
        Args:
            cutoff (float): Frecuencia de corte en Hz
            order (int): Orden del filtro
            
        Returns:
            tuple: (b, a) Coeficientes del filtro
            
        Raises:
            ValueError: Si los parámetros no son válidos
        """
        if cutoff <= 0:
            self.logger.error(f"Frecuencia de corte inválida: {cutoff}")
            raise ValueError("La frecuencia de corte debe ser mayor que cero")
            
        if order < 1:
            self.logger.warning(f"Orden de filtro inválido: {order}. Usando valor por defecto 4.")
            order = 4
            
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        
        if normal_cutoff >= 1.0:
            self.logger.warning(f"Frecuencia de corte ({cutoff} Hz) mayor que la frecuencia de Nyquist ({nyq} Hz). Ajustando.")
            normal_cutoff = 0.99
            
        self.logger.debug(f"Diseñando filtro pasa bajos: cutoff={cutoff} Hz, order={order}")
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_highpass(self, cutoff, order=4):
        """
        Diseña un filtro pasa altos Butterworth
        
        Args:
            cutoff (float): Frecuencia de corte en Hz
            order (int): Orden del filtro
            
        Returns:
            tuple: (b, a) Coeficientes del filtro
            
        Raises:
            ValueError: Si los parámetros no son válidos
        """
        if cutoff <= 0:
            self.logger.error(f"Frecuencia de corte inválida: {cutoff}")
            raise ValueError("La frecuencia de corte debe ser mayor que cero")
            
        if order < 1:
            self.logger.warning(f"Orden de filtro inválido: {order}. Usando valor por defecto 4.")
            order = 4
            
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        
        if normal_cutoff >= 1.0:
            self.logger.warning(f"Frecuencia de corte ({cutoff} Hz) mayor que la frecuencia de Nyquist ({nyq} Hz). Ajustando.")
            normal_cutoff = 0.99
            
        self.logger.debug(f"Diseñando filtro pasa altos: cutoff={cutoff} Hz, order={order}")
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_bandpass(self, lowcut, highcut, order=4):
        """
        Diseña un filtro pasa banda Butterworth
        
        Args:
            lowcut (float): Frecuencia de corte inferior en Hz
            highcut (float): Frecuencia de corte superior en Hz
            order (int): Orden del filtro
            
        Returns:
            tuple: (b, a) Coeficientes del filtro
            
        Raises:
            ValueError: Si los parámetros no son válidos
        """
        if lowcut <= 0 or highcut <= 0:
            self.logger.error(f"Frecuencias de corte inválidas: lowcut={lowcut}, highcut={highcut}")
            raise ValueError("Las frecuencias de corte deben ser mayores que cero")
            
        if lowcut >= highcut:
            self.logger.error(f"Frecuencia de corte inferior ({lowcut}) mayor o igual que la superior ({highcut})")
            raise ValueError("La frecuencia de corte inferior debe ser menor que la superior")
            
        if order < 1:
            self.logger.warning(f"Orden de filtro inválido: {order}. Usando valor por defecto 4.")
            order = 4
            
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        if high >= 1.0:
            self.logger.warning(f"Frecuencia de corte superior ({highcut} Hz) mayor que la frecuencia de Nyquist ({nyq} Hz). Ajustando.")
            high = 0.99
            
        self.logger.debug(f"Diseñando filtro pasa banda: lowcut={lowcut} Hz, highcut={highcut} Hz, order={order}")
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        return b, a

    def apply_filter(self, data, filter_type='lowpass', **kwargs):
        """
        Aplica el filtro seleccionado a los datos
        
        Args:
            data (numpy.ndarray): Array de datos a filtrar
            filter_type (str): Tipo de filtro ('lowpass', 'highpass', 'bandpass')
            **kwargs: Argumentos específicos del filtro (cutoff, order, etc.)
            
        Returns:
            numpy.ndarray: Datos filtrados
            
        Raises:
            ValueError: Si los datos o parámetros no son válidos
        """
        if data is None or len(data) == 0:
            self.logger.error("Los datos de entrada son nulos o vacíos")
            raise ValueError("Los datos de entrada no pueden ser nulos o vacíos")
            
        start_time = time_module.time()
        self.logger.info(f"Aplicando filtro {filter_type} a {len(data)} muestras")
        
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
            self.logger.error(f"Tipo de filtro no soportado: {filter_type}")
            raise ValueError(f"Tipo de filtro no soportado: {filter_type}")

        # Aplicar filtro con fase cero (forward-backward)
        try:
            filtered_data = signal.filtfilt(b, a, detrended)
        except Exception as e:
            self.logger.error(f"Error al aplicar filtro: {str(e)}")
            raise ValueError(f"Error al aplicar filtro: {str(e)}")
        
        end_time = time_module.time()
        self.logger.debug(f"Filtrado completado en {end_time - start_time:.4f} segundos")
        
        return filtered_data

    def get_filter_response(self, filter_type='lowpass', **kwargs):
        """
        Obtiene la respuesta en frecuencia del filtro
        
        Args:
            filter_type (str): Tipo de filtro
            **kwargs: Argumentos específicos del filtro
            
        Returns:
            tuple: (freqs, response) Frecuencias en Hz y respuesta en magnitud
            
        Raises:
            ValueError: Si el tipo de filtro no es soportado
        """
        self.logger.debug(f"Obteniendo respuesta en frecuencia para filtro {filter_type}")
        
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
            self.logger.error(f"Tipo de filtro no soportado: {filter_type}")
            raise ValueError(f"Tipo de filtro no soportado: {filter_type}")
            
        w, h = signal.freqz(b, a)
        # Convertir frecuencias normalizadas a Hz
        freqs = w * self.fs / (2 * np.pi)
        return freqs, np.abs(h)
