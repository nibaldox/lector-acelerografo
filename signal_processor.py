import numpy as np
from scipy import signal
from filters import SignalFilter

class SignalProcessor:
    def __init__(self, sampling_rate):
        """
        Inicializa el procesador de señales
        Args:
            sampling_rate: Frecuencia de muestreo en Hz
        """
        self.fs = sampling_rate
        self.filter = SignalFilter(sampling_rate)
        
    def remove_baseline(self, data, polynomial_order=3):
        """
        Elimina la tendencia de línea base usando un ajuste polinomial
        Args:
            data: Array de datos de entrada
            polynomial_order: Orden del polinomio para el ajuste
        Returns:
            corrected_data: Datos con línea base corregida
        """
        # Crear array de tiempo normalizado para mejor estabilidad numérica
        t = np.linspace(0, 1, len(data))
        
        # Ajustar polinomio
        coeffs = np.polyfit(t, data, polynomial_order)
        baseline = np.polyval(coeffs, t)
        
        # Restar línea base
        corrected_data = data - baseline
        
        return corrected_data
    
    def integrate_acceleration(self, acceleration, time, highpass_freq=0.1):
        """
        Integra aceleración para obtener velocidad con corrección de línea base
        Args:
            acceleration: Array de datos de aceleración
            time: Array de tiempo correspondiente
            highpass_freq: Frecuencia de corte para filtro pasa altos (Hz)
        Returns:
            velocity: Array de velocidad integrada
        """
        # Remover línea base
        acc_corrected = self.remove_baseline(acceleration)
        
        # Aplicar filtro pasa altos para eliminar deriva de baja frecuencia
        acc_filtered = self.filter.apply_filter(
            acc_corrected, 
            filter_type='highpass', 
            cutoff=highpass_freq
        )
        
        # Integración trapezoidal
        dt = time[1] - time[0]  # Intervalo de tiempo
        velocity = np.zeros_like(acc_filtered)
        
        # Método trapezoidal: v(n) = v(n-1) + (a(n) + a(n-1))*dt/2
        for i in range(1, len(velocity)):
            velocity[i] = velocity[i-1] + (acc_filtered[i] + acc_filtered[i-1]) * dt / 2
        
        # Corrección de línea base para la velocidad
        velocity = self.remove_baseline(velocity)
        
        return velocity
    
    def integrate_velocity(self, velocity, time, highpass_freq=0.05):
        """
        Integra velocidad para obtener desplazamiento con corrección de línea base
        Args:
            velocity: Array de datos de velocidad
            time: Array de tiempo correspondiente
            highpass_freq: Frecuencia de corte para filtro pasa altos (Hz)
        Returns:
            displacement: Array de desplazamiento integrado
        """
        # Remover línea base
        vel_corrected = self.remove_baseline(velocity)
        
        # Aplicar filtro pasa altos para eliminar deriva de baja frecuencia
        vel_filtered = self.filter.apply_filter(
            vel_corrected, 
            filter_type='highpass', 
            cutoff=highpass_freq
        )
        
        # Integración trapezoidal
        dt = time[1] - time[0]  # Intervalo de tiempo
        displacement = np.zeros_like(vel_filtered)
        
        # Método trapezoidal: d(n) = d(n-1) + (v(n) + v(n-1))*dt/2
        for i in range(1, len(displacement)):
            displacement[i] = displacement[i-1] + (vel_filtered[i] + vel_filtered[i-1]) * dt / 2
        
        # Corrección de línea base para el desplazamiento
        displacement = self.remove_baseline(displacement)
        
        return displacement
    
    def process_acceleration_data(self, acceleration, time):
        """
        Procesa datos de aceleración para obtener velocidad y desplazamiento
        Args:
            acceleration: Array de datos de aceleración
            time: Array de tiempo correspondiente
        Returns:
            dict: Diccionario con arrays de aceleración, velocidad y desplazamiento
        """
        # Integrar aceleración para obtener velocidad
        velocity = self.integrate_acceleration(acceleration, time)
        
        # Integrar velocidad para obtener desplazamiento
        displacement = self.integrate_velocity(velocity, time)
        
        return {
            'acceleration': acceleration,
            'velocity': velocity,
            'displacement': displacement,
            'time': time
        }
    
    def compute_response_spectrum(self, acceleration, time, periods=None, damping_ratio=0.05):
        """
        Calcula el espectro de respuesta de aceleración, velocidad y desplazamiento.
        
        Args:
            acceleration (numpy.array): Datos de aceleración
            time (numpy.array): Vector de tiempo
            periods (numpy.array, opcional): Periodos para calcular la respuesta
            damping_ratio (float, opcional): Razón de amortiguamiento (default: 5%)
            
        Returns:
            dict: Periodos y espectros de respuesta (Sa, Sv, Sd)
        """
        if periods is None:
            periods = np.logspace(-2, 1, 100)  # 0.01s a 10s
            
        dt = time[1] - time[0]
        omega = 2 * np.pi / periods
        Sa = np.zeros_like(periods)
        Sv = np.zeros_like(periods)
        Sd = np.zeros_like(periods)
        
        for i, T in enumerate(periods):
            # Parámetros del sistema de 1GDL
            w = 2 * np.pi / T
            c = 2 * damping_ratio * w
            k = w * w
            
            # Resolver ecuación diferencial usando método de Newmark-Beta
            u = np.zeros_like(acceleration)  # Desplazamiento
            v = np.zeros_like(acceleration)  # Velocidad
            
            # Parámetros de Newmark-Beta (promedio constante de aceleración)
            gamma = 0.5
            beta = 0.25
            
            # Constantes para el método
            a1 = 1 / (beta * dt * dt) + (gamma * c) / (beta * dt)
            a2 = 1 / (beta * dt)
            a3 = 1 / (2 * beta) - 1
            
            for j in range(1, len(acceleration)):
                # Predictor
                dp = -k * u[j-1] - c * v[j-1] - acceleration[j]
                
                # Corrector
                du = dp / (k + a1)
                u[j] = u[j-1] + du
                v[j] = v[j-1] + a2 * du
            
            # Calcular valores máximos
            Sd[i] = np.max(np.abs(u))
            Sv[i] = np.max(np.abs(v))
            Sa[i] = w * w * Sd[i]  # Relación entre Sa y Sd
        
        return {
            'periods': periods,
            'Sa': Sa,
            'Sv': Sv,
            'Sd': Sd
        }
    
    def compute_power_spectrum(self, data, sampling_rate):
        """
        Calcula el espectro de potencia de la señal.
        
        Args:
            data (numpy.array): Datos de la señal
            sampling_rate (float): Frecuencia de muestreo en Hz
            
        Returns:
            dict: Frecuencias y espectro de potencia
        """
        # Calcular la FFT
        n = len(data)
        freq = np.fft.fftfreq(n, d=1/sampling_rate)
        fft_vals = np.fft.fft(data)
        
        # Calcular el espectro de potencia (solo frecuencias positivas)
        pos_freq_idx = freq >= 0
        frequencies = freq[pos_freq_idx]
        power_spectrum = np.abs(fft_vals[pos_freq_idx])**2 / n
        
        return {
            'frequencies': frequencies,
            'power_spectrum': power_spectrum
        }
    
    def compute_autocorrelation(self, data, max_lag=None):
        """
        Calcula la función de autocorrelación de la señal.
        
        Args:
            data (numpy.array): Datos de la señal
            max_lag (int, opcional): Máximo desfase a considerar
            
        Returns:
            dict: Desfases y coeficientes de autocorrelación
        """
        if max_lag is None:
            max_lag = len(data) // 2
            
        # Normalizar los datos
        data = (data - np.mean(data)) / np.std(data)
        
        # Calcular autocorrelación
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Solo la parte positiva
        
        # Normalizar por la autocorrelación en lag=0
        autocorr = autocorr / autocorr[0]
        
        # Limitar al máximo desfase
        lags = np.arange(min(len(autocorr), max_lag))
        autocorr = autocorr[:max_lag]
        
        return {
            'lags': lags,
            'autocorr': autocorr
        }
    
    def compute_combined_response(self, data_x, data_y, data_z, time, method='SRSS', damping_ratio=0.05):
        """
        Calcula la respuesta combinada de múltiples componentes.
        
        Args:
            data_x (numpy.array): Datos de la componente X (Norte-Sur)
            data_y (numpy.array): Datos de la componente Y (Este-Oeste)
            data_z (numpy.array): Datos de la componente Z (Vertical)
            time (numpy.array): Vector de tiempo
            method (str): Método de combinación ('SRSS' o 'Porcentual')
            damping_ratio (float): Razón de amortiguamiento (default: 0.05)
            
        Returns:
            dict: Respuesta combinada y sus componentes
        """
        if method not in ['SRSS', 'Porcentual']:
            raise ValueError("El método debe ser 'SRSS' o 'Porcentual'")
        
        # Calcular espectros de respuesta individuales
        periods = np.logspace(-2, 1, 100)  # 0.01s a 10s
        resp_x = self.compute_response_spectrum(data_x, time, periods, damping_ratio)
        resp_y = self.compute_response_spectrum(data_y, time, periods, damping_ratio)
        resp_z = self.compute_response_spectrum(data_z, time, periods, damping_ratio)
        
        # Inicializar arrays para la respuesta combinada
        Sa_comb = np.zeros_like(periods)
        Sv_comb = np.zeros_like(periods)
        Sd_comb = np.zeros_like(periods)
        
        if method == 'SRSS':
            # Método de la raíz cuadrada de la suma de cuadrados
            Sa_comb = np.sqrt(resp_x['Sa']**2 + resp_y['Sa']**2 + resp_z['Sa']**2)
            Sv_comb = np.sqrt(resp_x['Sv']**2 + resp_y['Sv']**2 + resp_z['Sv']**2)
            Sd_comb = np.sqrt(resp_x['Sd']**2 + resp_y['Sd']**2 + resp_z['Sd']**2)
        else:  # Método Porcentual (30%)
            # Todas las combinaciones posibles
            combinations = [
                (1.0, 0.3, 0.3),
                (0.3, 1.0, 0.3),
                (0.3, 0.3, 1.0)
            ]
            
            # Calcular cada combinación y tomar el máximo
            for cx, cy, cz in combinations:
                Sa_temp = np.abs(cx * resp_x['Sa']) + np.abs(cy * resp_y['Sa']) + np.abs(cz * resp_z['Sa'])
                Sv_temp = np.abs(cx * resp_x['Sv']) + np.abs(cy * resp_y['Sv']) + np.abs(cz * resp_z['Sv'])
                Sd_temp = np.abs(cx * resp_x['Sd']) + np.abs(cy * resp_y['Sd']) + np.abs(cz * resp_z['Sd'])
                
                Sa_comb = np.maximum(Sa_comb, Sa_temp)
                Sv_comb = np.maximum(Sv_comb, Sv_temp)
                Sd_comb = np.maximum(Sd_comb, Sd_temp)
        
        return {
            'periods': periods,
            'Sa_combined': Sa_comb,
            'Sv_combined': Sv_comb,
            'Sd_combined': Sd_comb,
            'Sa_x': resp_x['Sa'],
            'Sa_y': resp_y['Sa'],
            'Sa_z': resp_z['Sa'],
            'Sv_x': resp_x['Sv'],
            'Sv_y': resp_y['Sv'],
            'Sv_z': resp_z['Sv'],
            'Sd_x': resp_x['Sd'],
            'Sd_y': resp_y['Sd'],
            'Sd_z': resp_z['Sd']
        }

    def compute_particle_orbit(self, data_x, data_y, time, start_time=None, end_time=None):
        """
        Calcula la órbita de partículas entre dos componentes.
        
        Args:
            data_x (numpy.array): Datos de la primera componente
            data_y (numpy.array): Datos de la segunda componente
            time (numpy.array): Vector de tiempo
            start_time (float, opcional): Tiempo inicial para el análisis
            end_time (float, opcional): Tiempo final para el análisis
            
        Returns:
            dict: Datos para graficar la órbita de partículas
        """
        # Si no se especifican tiempos, usar todo el registro
        if start_time is None:
            start_time = time[0]
        if end_time is None:
            end_time = time[-1]
            
        # Filtrar datos por tiempo
        mask = (time >= start_time) & (time <= end_time)
        x_filtered = data_x[mask]
        y_filtered = data_y[mask]
        time_filtered = time[mask]
        
        return {
            'x': x_filtered,
            'y': y_filtered,
            'time': time_filtered
        }
    
    def compute_fourier_amplitude_ratio(self, data_x, data_y, time):
        """
        Calcula la relación de amplitud de Fourier entre dos componentes.
        
        Args:
            data_x (numpy.array): Datos de la primera componente
            data_y (numpy.array): Datos de la segunda componente
            time (numpy.array): Vector de tiempo
            
        Returns:
            dict: Frecuencias y relación de amplitud
        """
        # Calcular FFT para ambas componentes
        n = len(data_x)
        freq = np.fft.fftfreq(n, d=1/self.fs)
        fft_x = np.fft.fft(data_x)
        fft_y = np.fft.fft(data_y)
        
        # Calcular amplitudes
        amp_x = np.abs(fft_x)
        amp_y = np.abs(fft_y)
        
        # Calcular relación (evitar división por cero)
        ratio = np.zeros_like(amp_x)
        mask = amp_y > 0
        ratio[mask] = amp_x[mask] / amp_y[mask]
        
        # Solo frecuencias positivas
        pos_freq_idx = freq >= 0
        frequencies = freq[pos_freq_idx]
        ratio = ratio[pos_freq_idx]
        
        return {
            'frequencies': frequencies,
            'ratio': ratio
        }
    
    def compute_phase_difference(self, data_x, data_y, time):
        """
        Calcula la diferencia de fase entre dos componentes.
        
        Args:
            data_x (numpy.array): Datos de la primera componente
            data_y (numpy.array): Datos de la segunda componente
            time (numpy.array): Vector de tiempo
            
        Returns:
            dict: Frecuencias y diferencia de fase
        """
        # Calcular FFT para ambas componentes
        n = len(data_x)
        freq = np.fft.fftfreq(n, d=1/self.fs)
        fft_x = np.fft.fft(data_x)
        fft_y = np.fft.fft(data_y)
        
        # Calcular fases
        phase_x = np.angle(fft_x)
        phase_y = np.angle(fft_y)
        
        # Calcular diferencia de fase (en grados)
        phase_diff = np.degrees(phase_x - phase_y)
        
        # Normalizar a rango [-180, 180]
        phase_diff = (phase_diff + 180) % 360 - 180
        
        # Solo frecuencias positivas
        pos_freq_idx = freq >= 0
        frequencies = freq[pos_freq_idx]
        phase_diff = phase_diff[pos_freq_idx]
        
        return {
            'frequencies': frequencies,
            'phase_difference': phase_diff
        }
    
    def compute_cross_power_spectrum(self, data_x, data_y, time):
        """
        Calcula el espectro de potencia cruzada entre dos componentes.
        
        Args:
            data_x (numpy.array): Datos de la primera componente
            data_y (numpy.array): Datos de la segunda componente
            time (numpy.array): Vector de tiempo
            
        Returns:
            dict: Frecuencias y espectro de potencia cruzada
        """
        # Calcular FFT para ambas componentes
        n = len(data_x)
        freq = np.fft.fftfreq(n, d=1/self.fs)
        fft_x = np.fft.fft(data_x)
        fft_y = np.fft.fft(data_y)
        
        # Calcular espectro de potencia cruzada
        cross_power = fft_x * np.conjugate(fft_y) / n
        
        # Solo frecuencias positivas
        pos_freq_idx = freq >= 0
        frequencies = freq[pos_freq_idx]
        cross_power = cross_power[pos_freq_idx]
        
        return {
            'frequencies': frequencies,
            'cross_power_real': np.real(cross_power),
            'cross_power_imag': np.imag(cross_power),
            'cross_power_magnitude': np.abs(cross_power)
        }
    
    def compute_cross_correlation(self, data_x, data_y, max_lag=None):
        """
        Calcula la correlación cruzada entre dos componentes.
        
        Args:
            data_x (numpy.array): Datos de la primera componente
            data_y (numpy.array): Datos de la segunda componente
            max_lag (int, opcional): Máximo desfase a considerar
            
        Returns:
            dict: Desfases y coeficientes de correlación cruzada
        """
        if max_lag is None:
            max_lag = len(data_x) // 2
            
        # Normalizar los datos
        data_x_norm = (data_x - np.mean(data_x)) / np.std(data_x)
        data_y_norm = (data_y - np.mean(data_y)) / np.std(data_y)
        
        # Calcular correlación cruzada
        cross_corr = np.correlate(data_x_norm, data_y_norm, mode='full')
        
        # Centrar en lag=0
        mid_point = len(cross_corr) // 2
        lags = np.arange(-mid_point, mid_point + 1)
        
        # Normalizar por las autocorrelaciones en lag=0
        auto_x = np.correlate(data_x_norm, data_x_norm, mode='valid')[0]
        auto_y = np.correlate(data_y_norm, data_y_norm, mode='valid')[0]
        norm_factor = np.sqrt(auto_x * auto_y)
        cross_corr = cross_corr / norm_factor
        
        # Limitar al máximo desfase
        if max_lag < len(lags) // 2:
            start_idx = mid_point - max_lag
            end_idx = mid_point + max_lag + 1
            lags = lags[start_idx:end_idx]
            cross_corr = cross_corr[start_idx:end_idx]
        
        return {
            'lags': lags,
            'cross_corr': cross_corr
        }
    
    def compute_coherence(self, data_x, data_y, time, nperseg=256):
        """
        Calcula la función de coherencia entre dos componentes.
        
        Args:
            data_x (numpy.array): Datos de la primera componente
            data_y (numpy.array): Datos de la segunda componente
            time (numpy.array): Vector de tiempo
            nperseg (int, opcional): Longitud de cada segmento para el cálculo
            
        Returns:
            dict: Frecuencias y función de coherencia
        """
        # Calcular coherencia usando scipy.signal
        freq, coherence = signal.coherence(
            data_x, 
            data_y, 
            fs=self.fs, 
            nperseg=nperseg
        )
        
        return {
            'frequencies': freq,
            'coherence': coherence
        }
