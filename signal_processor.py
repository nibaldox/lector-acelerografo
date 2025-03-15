import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
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
    
    def compute_response_spectrum(self, acceleration, time, periods, damping=0.05):
        """
        Calcula el espectro de respuesta para diferentes períodos
        Args:
            acceleration: Array de datos de aceleración
            time: Array de tiempo correspondiente
            periods: Lista de períodos para calcular la respuesta (segundos)
            damping: Coeficiente de amortiguamiento (por defecto 5%)
        Returns:
            dict: Diccionario con espectros de respuesta de aceleración, velocidad y desplazamiento
        """
        dt = time[1] - time[0]  # Intervalo de tiempo
        
        # Inicializar arrays para almacenar resultados
        Sa = np.zeros(len(periods))  # Espectro de respuesta de aceleración
        Sv = np.zeros(len(periods))  # Espectro de respuesta de velocidad
        Sd = np.zeros(len(periods))  # Espectro de respuesta de desplazamiento
        
        # Para cada período, calcular la respuesta máxima
        for i, period in enumerate(periods):
            if period > 0:  # Evitar división por cero
                # Frecuencia natural en rad/s
                omega = 2 * np.pi / period
                
                # Constantes para el método de Newmark-beta
                c = 2 * damping * omega  # Coeficiente de amortiguamiento
                k = omega**2  # Rigidez
                
                # Inicializar desplazamiento y velocidad
                u = np.zeros_like(time)
                v = np.zeros_like(time)
                a = np.zeros_like(time)
                
                # Condiciones iniciales
                u[0] = 0
                v[0] = 0
                a[0] = -acceleration[0]
                
                # Método de Newmark-beta (aceleración promedio constante)
                gamma = 0.5
                beta = 0.25
                
                # Constantes para el método
                a1 = 1 / (beta * dt**2)
                a2 = 1 / (beta * dt)
                a3 = 1 / (2 * beta) - 1
                a4 = gamma / (beta * dt)
                a5 = 1 - gamma / beta
                a6 = dt * (1 - gamma / (2 * beta))
                
                # Resolver ecuación de movimiento paso a paso
                for j in range(1, len(time)):
                    # Predicción
                    u_pred = u[j-1] + dt * v[j-1] + dt**2 / 2 * a[j-1]
                    v_pred = v[j-1] + dt * a[j-1]
                    
                    # Residuo
                    r = -k * u_pred - c * v_pred - acceleration[j]
                    
                    # Corrección
                    delta_a = r / (1 + c * gamma * dt + k * beta * dt**2)
                    delta_v = gamma * dt * delta_a
                    delta_u = beta * dt**2 * delta_a
                    
                    # Actualizar
                    a[j] = a[j-1] + delta_a
                    v[j] = v_pred + delta_v
                    u[j] = u_pred + delta_u
                
                # Guardar valores máximos (en valor absoluto)
                Sd[i] = np.max(np.abs(u))
                Sv[i] = np.max(np.abs(v))
                Sa[i] = np.max(np.abs(a + acceleration))  # Aceleración total
        
        return {
            'periods': periods,
            'Sa': Sa,
            'Sv': Sv,
            'Sd': Sd
        }
