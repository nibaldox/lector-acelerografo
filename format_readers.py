"""
Módulo para la lectura de diferentes formatos de archivos sísmicos.
Proporciona clases para leer y convertir datos de formatos comunes en sismología.
"""

import numpy as np
import pandas as pd
import os
import struct
from pathlib import Path
from datetime import datetime
import warnings

class BaseReader:
    """Clase base para todos los lectores de formatos"""
    
    def __init__(self, file_path):
        """
        Inicializa el lector base
        
        Args:
            file_path: Ruta al archivo a leer
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    def read_data(self):
        """
        Método para leer datos que debe ser implementado por las subclases
        
        Returns:
            dict: Diccionario con los datos leídos
        """
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def _extract_metadata(self):
        """
        Método para extraer metadatos que debe ser implementado por las subclases
        
        Returns:
            dict: Diccionario con los metadatos
        """
        raise NotImplementedError("Las subclases deben implementar este método")


class SACReader(BaseReader):
    """Clase para leer archivos en formato SAC (Seismic Analysis Code)"""
    
    def __init__(self, file_path):
        """
        Inicializa el lector de archivos SAC
        
        Args:
            file_path: Ruta al archivo SAC
        """
        super().__init__(file_path)
        self.header = None
        self.data = None
    
    def read_data(self):
        """
        Lee los datos de un archivo SAC
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        try:
            with open(self.file_path, 'rb') as f:
                # Leer encabezado
                self.header = self._read_header(f)
                
                # Leer datos
                npts = self.header['npts']
                f.seek(632)  # Posición donde comienzan los datos
                data = np.fromfile(f, dtype=np.float32, count=npts)
                
                # Crear array de tiempo
                delta = self.header['delta']
                time = np.arange(0, npts * delta, delta)
                
                # Extraer metadatos relevantes
                metadata = self._extract_metadata()
                
                # Determinar la componente basada en el nombre del archivo o metadata
                component = self._determine_component()
                
                # Crear diccionario de salida en formato compatible
                result = {
                    'time': time,
                    component: data,
                    'dt': delta,
                    'components': [component],
                    'metadata': metadata,
                    'name': self.file_path.name
                }
                
                return result
                
        except Exception as e:
            raise IOError(f"Error al leer archivo SAC: {str(e)}")
    
    def _read_header(self, file_handle):
        """
        Lee el encabezado de un archivo SAC
        
        Args:
            file_handle: Manejador de archivo abierto
            
        Returns:
            dict: Diccionario con los valores del encabezado
        """
        # Estructura del encabezado SAC (versión simplificada)
        header = {}
        
        # Leer valores float (70 valores de 4 bytes cada uno)
        float_values = np.fromfile(file_handle, dtype=np.float32, count=70)
        
        # Leer valores int (40 valores de 4 bytes cada uno)
        file_handle.seek(280)
        int_values = np.fromfile(file_handle, dtype=np.int32, count=40)
        
        # Leer valores char (192 bytes en total)
        file_handle.seek(440)
        char_data = file_handle.read(192)
        
        # Asignar valores importantes al diccionario
        header['delta'] = float_values[0]  # Intervalo de muestreo
        header['npts'] = int(int_values[9])  # Número de puntos
        header['b'] = float_values[5]  # Tiempo de inicio
        header['e'] = float_values[6]  # Tiempo de fin
        
        # Más campos pueden ser añadidos según sea necesario
        
        return header
    
    def _extract_metadata(self):
        """
        Extrae metadatos del encabezado SAC
        
        Returns:
            dict: Diccionario con metadatos en formato estándar
        """
        if self.header is None:
            with open(self.file_path, 'rb') as f:
                self.header = self._read_header(f)
        
        # Convertir metadatos SAC a formato estándar
        metadata = {
            'sampling_rate': 1.0 / self.header['delta'],
            'npts': self.header['npts'],
            'start_time': self.header['b'],
            'end_time': self.header['e'],
            'duration': self.header['e'] - self.header['b'],
            'format': 'SAC',
            'unit': 'm/s/s',  # Asumimos aceleración, pero podría variar
            'sensor_name': 'Unknown',  # Podría extraerse de otros campos
            'station': 'Unknown',  # Podría extraerse de otros campos
        }
        
        return metadata
    
    def _determine_component(self):
        """
        Determina la componente (N, E, Z) basada en el nombre del archivo o metadatos
        
        Returns:
            str: Componente ('N', 'E', o 'Z')
        """
        # Estrategia simple basada en el nombre del archivo
        filename = self.file_path.name.upper()
        
        if 'NORTH' in filename or 'NS' in filename or '_N_' in filename or filename.endswith('N'):
            return 'N'
        elif 'EAST' in filename or 'EW' in filename or '_E_' in filename or filename.endswith('E'):
            return 'E'
        elif 'VERT' in filename or 'UP' in filename or '_Z_' in filename or filename.endswith('Z'):
            return 'Z'
        else:
            # Si no podemos determinar, asumimos componente N
            return 'N'


class MiniSEEDReader(BaseReader):
    """Clase para leer archivos en formato miniSEED"""
    
    def __init__(self, file_path):
        """
        Inicializa el lector de archivos miniSEED
        
        Args:
            file_path: Ruta al archivo miniSEED
        """
        super().__init__(file_path)
        
        # Verificar si obspy está disponible
        try:
            import obspy
            self.obspy_available = True
        except ImportError:
            self.obspy_available = False
            warnings.warn("La biblioteca obspy no está instalada. Se usará un lector básico para miniSEED.")
    
    def read_data(self):
        """
        Lee los datos de un archivo miniSEED
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        if self.obspy_available:
            return self._read_with_obspy()
        else:
            return self._read_basic()
    
    def _read_with_obspy(self):
        """
        Lee el archivo usando obspy (método preferido)
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        try:
            import obspy
            
            # Leer el archivo con obspy
            st = obspy.read(str(self.file_path))
            
            # Obtener la primera traza
            tr = st[0]
            
            # Crear array de tiempo
            delta = tr.stats.delta
            npts = tr.stats.npts
            time = np.arange(0, npts * delta, delta)
            
            # Determinar la componente
            component = self._determine_component(tr)
            
            # Extraer metadatos
            metadata = {
                'sampling_rate': tr.stats.sampling_rate,
                'npts': tr.stats.npts,
                'start_time': str(tr.stats.starttime),
                'end_time': str(tr.stats.endtime),
                'duration': tr.stats.endtime - tr.stats.starttime,
                'format': 'miniSEED',
                'unit': 'm/s/s',  # Asumimos aceleración
                'sensor_name': tr.stats.get('sensor', 'Unknown'),
                'station': tr.stats.station,
                'network': tr.stats.network,
                'channel': tr.stats.channel,
                'location': tr.stats.location
            }
            
            # Crear diccionario de salida en formato compatible
            result = {
                'time': time,
                component: tr.data,
                'dt': delta,
                'components': [component],
                'metadata': metadata,
                'name': self.file_path.name
            }
            
            return result
            
        except Exception as e:
            raise IOError(f"Error al leer archivo miniSEED con obspy: {str(e)}")
    
    def _read_basic(self):
        """
        Implementación básica para leer miniSEED sin obspy
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        # Esta es una implementación muy básica y limitada
        # miniSEED es un formato complejo y se recomienda usar obspy
        
        try:
            with open(self.file_path, 'rb') as f:
                # Leer los primeros bytes para identificar el formato
                header = f.read(48)  # Primeros 48 bytes contienen información básica
                
                # Verificar si es un archivo miniSEED válido
                if header[0:6] != b'000001':
                    raise ValueError("No parece ser un archivo miniSEED válido")
                
                # Extraer información básica
                # Esto es muy simplificado y no funcionará para todos los archivos miniSEED
                sample_rate_factor = struct.unpack('>h', header[36:38])[0]
                sample_rate_mult = struct.unpack('>h', header[38:40])[0]
                
                if sample_rate_factor > 0 and sample_rate_mult > 0:
                    sampling_rate = sample_rate_factor * sample_rate_mult
                elif sample_rate_factor > 0 and sample_rate_mult < 0:
                    sampling_rate = sample_rate_factor / abs(sample_rate_mult)
                elif sample_rate_factor < 0 and sample_rate_mult > 0:
                    sampling_rate = -1.0 / (sample_rate_factor * sample_rate_mult)
                else:
                    sampling_rate = -1.0 / (sample_rate_factor / sample_rate_mult)
                
                # Crear metadatos básicos
                metadata = {
                    'sampling_rate': sampling_rate,
                    'format': 'miniSEED',
                    'unit': 'm/s/s',  # Asumimos aceleración
                    'sensor_name': 'Unknown',
                    'station': 'Unknown'
                }
                
                # Advertir sobre la implementación limitada
                warnings.warn("Usando implementación básica para miniSEED. Se recomienda instalar obspy.")
                
                # Crear un conjunto de datos simulado
                # Esto es solo un placeholder, no datos reales
                time = np.arange(0, 10, 1/sampling_rate)
                data = np.zeros_like(time)
                
                # Determinar componente basado en el nombre del archivo
                component = self._determine_component()
                
                # Crear diccionario de salida
                result = {
                    'time': time,
                    component: data,
                    'dt': 1/sampling_rate,
                    'components': [component],
                    'metadata': metadata,
                    'name': self.file_path.name
                }
                
                return result
                
        except Exception as e:
            raise IOError(f"Error al leer archivo miniSEED: {str(e)}")
    
    def _determine_component(self, trace=None):
        """
        Determina la componente (N, E, Z) basada en el nombre del archivo o metadatos
        
        Args:
            trace: Objeto de traza de obspy (opcional)
            
        Returns:
            str: Componente ('N', 'E', o 'Z')
        """
        if trace is not None and hasattr(trace.stats, 'channel'):
            # Usar el código de canal si está disponible
            channel = trace.stats.channel
            if channel:
                last_char = channel[-1].upper()
                if last_char == 'N' or last_char == '1':
                    return 'N'
                elif last_char == 'E' or last_char == '2':
                    return 'E'
                elif last_char == 'Z':
                    return 'Z'
        
        # Estrategia basada en el nombre del archivo
        filename = self.file_path.name.upper()
        
        if 'NORTH' in filename or 'NS' in filename or '_N_' in filename or filename.endswith('N'):
            return 'N'
        elif 'EAST' in filename or 'EW' in filename or '_E_' in filename or filename.endswith('E'):
            return 'E'
        elif 'VERT' in filename or 'UP' in filename or '_Z_' in filename or filename.endswith('Z'):
            return 'Z'
        else:
            # Si no podemos determinar, asumimos componente N
            return 'N'


class ASCIIReader(BaseReader):
    """Clase para leer archivos en formato ASCII (columnas de datos)"""
    
    def __init__(self, file_path, delimiter=None, skiprows=0, time_column=0, data_column=1):
        """
        Inicializa el lector de archivos ASCII
        
        Args:
            file_path: Ruta al archivo ASCII
            delimiter: Delimitador de columnas (None para autodetectar)
            skiprows: Número de filas a omitir al inicio
            time_column: Índice de la columna de tiempo (0-based)
            data_column: Índice de la columna de datos (0-based)
        """
        super().__init__(file_path)
        self.delimiter = delimiter
        self.skiprows = skiprows
        self.time_column = time_column
        self.data_column = data_column
    
    def read_data(self):
        """
        Lee los datos de un archivo ASCII
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        try:
            # Leer archivo con pandas
            df = pd.read_csv(
                self.file_path, 
                delimiter=self.delimiter, 
                skiprows=self.skiprows,
                header=None
            )
            
            # Verificar que hay suficientes columnas
            if len(df.columns) <= max(self.time_column, self.data_column):
                raise ValueError(f"El archivo no tiene suficientes columnas. Se esperaban al menos {max(self.time_column, self.data_column)+1}")
            
            # Extraer columnas de tiempo y datos
            time = df.iloc[:, self.time_column].values
            data = df.iloc[:, self.data_column].values
            
            # Calcular dt (intervalo de tiempo)
            if len(time) > 1:
                dt = np.mean(np.diff(time))
            else:
                dt = 0.01  # Valor predeterminado
            
            # Determinar la componente basada en el nombre del archivo
            component = self._determine_component()
            
            # Crear metadatos básicos
            metadata = {
                'sampling_rate': 1.0 / dt,
                'npts': len(data),
                'duration': time[-1] - time[0] if len(time) > 1 else 0,
                'format': 'ASCII',
                'unit': 'm/s/s',  # Asumimos aceleración
                'sensor_name': 'Unknown',
                'station': 'Unknown'
            }
            
            # Crear diccionario de salida en formato compatible
            result = {
                'time': time,
                component: data,
                'dt': dt,
                'components': [component],
                'metadata': metadata,
                'name': self.file_path.name
            }
            
            return result
            
        except Exception as e:
            raise IOError(f"Error al leer archivo ASCII: {str(e)}")
    
    def _extract_metadata(self):
        """
        Extrae metadatos del archivo ASCII
        
        Returns:
            dict: Diccionario con metadatos en formato estándar
        """
        # Para archivos ASCII, los metadatos son limitados
        # y se calculan principalmente en read_data()
        return {}
    
    def _determine_component(self):
        """
        Determina la componente (N, E, Z) basada en el nombre del archivo
        
        Returns:
            str: Componente ('N', 'E', o 'Z')
        """
        filename = self.file_path.name.upper()
        
        if 'NORTH' in filename or 'NS' in filename or '_N_' in filename or filename.endswith('N'):
            return 'N'
        elif 'EAST' in filename or 'EW' in filename or '_E_' in filename or filename.endswith('E'):
            return 'E'
        elif 'VERT' in filename or 'UP' in filename or '_Z_' in filename or filename.endswith('Z'):
            return 'Z'
        else:
            # Si no podemos determinar, asumimos componente N
            return 'N'


class SEGYReader(BaseReader):
    """Clase para leer archivos en formato SEG-Y"""
    
    def __init__(self, file_path):
        """
        Inicializa el lector de archivos SEG-Y
        
        Args:
            file_path: Ruta al archivo SEG-Y
        """
        super().__init__(file_path)
        
        # Verificar si obspy está disponible
        try:
            import obspy
            self.obspy_available = True
        except ImportError:
            self.obspy_available = False
            warnings.warn("La biblioteca obspy no está instalada. Se usará un lector básico para SEG-Y.")
    
    def read_data(self):
        """
        Lee los datos de un archivo SEG-Y
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        if self.obspy_available:
            return self._read_with_obspy()
        else:
            return self._read_basic()
    
    def _read_with_obspy(self):
        """
        Lee el archivo usando obspy (método preferido)
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        try:
            import obspy
            from obspy.io.segy.segy import _read_segy
            
            # Leer el archivo con obspy
            st = _read_segy(str(self.file_path))
            
            # Obtener la primera traza
            tr = st[0]
            
            # Crear array de tiempo
            delta = tr.stats.delta
            npts = tr.stats.npts
            time = np.arange(0, npts * delta, delta)
            
            # Determinar la componente
            component = self._determine_component()
            
            # Extraer metadatos
            metadata = {
                'sampling_rate': tr.stats.sampling_rate,
                'npts': tr.stats.npts,
                'duration': npts * delta,
                'format': 'SEG-Y',
                'unit': 'm/s/s',  # Asumimos aceleración
                'sensor_name': 'Unknown',
                'station': 'Unknown'
            }
            
            # Crear diccionario de salida en formato compatible
            result = {
                'time': time,
                component: tr.data,
                'dt': delta,
                'components': [component],
                'metadata': metadata,
                'name': self.file_path.name
            }
            
            return result
            
        except Exception as e:
            raise IOError(f"Error al leer archivo SEG-Y con obspy: {str(e)}")
    
    def _read_basic(self):
        """
        Implementación básica para leer SEG-Y sin obspy
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        # Esta es una implementación muy básica y limitada
        # SEG-Y es un formato complejo y se recomienda usar obspy
        
        try:
            with open(self.file_path, 'rb') as f:
                # Leer encabezado textual (3200 bytes)
                textual_header = f.read(3200)
                
                # Leer encabezado binario (400 bytes)
                binary_header = f.read(400)
                
                # Extraer información básica del encabezado binario
                sample_interval = struct.unpack('>h', binary_header[16:18])[0]
                if sample_interval > 0:
                    dt = sample_interval / 1000000.0  # Convertir de microsegundos a segundos
                else:
                    dt = 0.001  # Valor predeterminado
                
                # Advertir sobre la implementación limitada
                warnings.warn("Usando implementación básica para SEG-Y. Se recomienda instalar obspy.")
                
                # Crear un conjunto de datos simulado
                # Esto es solo un placeholder, no datos reales
                time = np.arange(0, 10, dt)
                data = np.zeros_like(time)
                
                # Determinar componente basado en el nombre del archivo
                component = self._determine_component()
                
                # Crear metadatos básicos
                metadata = {
                    'sampling_rate': 1.0 / dt,
                    'format': 'SEG-Y',
                    'unit': 'm/s/s',  # Asumimos aceleración
                    'sensor_name': 'Unknown',
                    'station': 'Unknown'
                }
                
                # Crear diccionario de salida
                result = {
                    'time': time,
                    component: data,
                    'dt': dt,
                    'components': [component],
                    'metadata': metadata,
                    'name': self.file_path.name
                }
                
                return result
                
        except Exception as e:
            raise IOError(f"Error al leer archivo SEG-Y: {str(e)}")
    
    def _extract_metadata(self):
        """
        Extrae metadatos del archivo SEG-Y
        
        Returns:
            dict: Diccionario con metadatos en formato estándar
        """
        # Para SEG-Y, los metadatos se extraen principalmente en read_data()
        return {}
    
    def _determine_component(self):
        """
        Determina la componente (N, E, Z) basada en el nombre del archivo
        
        Returns:
            str: Componente ('N', 'E', o 'Z')
        """
        filename = self.file_path.name.upper()
        
        if 'NORTH' in filename or 'NS' in filename or '_N_' in filename or filename.endswith('N'):
            return 'N'
        elif 'EAST' in filename or 'EW' in filename or '_E_' in filename or filename.endswith('E'):
            return 'E'
        elif 'VERT' in filename or 'UP' in filename or '_Z_' in filename or filename.endswith('Z'):
            return 'Z'
        else:
            # Si no podemos determinar, asumimos componente N
            return 'N'


class MSReader(BaseReader):
    """Clase para leer archivos en formato MS/SS (formato propietario de acelerógrafos)"""
    
    def __init__(self, file_path):
        """
        Inicializa el lector de archivos MS
        
        Args:
            file_path: Ruta al archivo MS
        """
        super().__init__(file_path)
    
    def read_data(self):
        """
        Lee el archivo binario .ms y retorna los datos de aceleración
        
        Returns:
            dict: Diccionario con los datos y metadatos
        """
        try:
            with open(self.file_path, 'rb') as f:
                # Leer todo el contenido binario
                data = f.read()
                
            # Primeros bytes podrían ser encabezado
            header_size = 32  # Asumimos 32 bytes de encabezado
            data_without_header = data[header_size:]
            
            # Leer datos crudos como int32
            raw_data = np.frombuffer(data_without_header, dtype=np.int32)
            
            # Si el tamaño no es divisible por 3, ajustamos
            total_samples = len(raw_data)
            samples_per_channel = total_samples // 3
            
            # Reorganizar los datos en tres canales
            channels = []
            for i in range(3):
                start = i * samples_per_channel
                end = (i + 1) * samples_per_channel
                channels.append(raw_data[start:end])
            
            data_array = np.column_stack(channels)
            samples = samples_per_channel  # Número de muestras por canal
            
            # Leer configuración del archivo .ss
            ss_file_path = str(self.file_path).replace('.ms', '.ss')
            metadata = self._extract_metadata(ss_file_path)

            # Obtener offsets para cada canal
            zero_offset_E = float(metadata.get('zero_offset_E', '0'))
            zero_offset_N = float(metadata.get('zero_offset_N', '0'))
            zero_offset_Z = float(metadata.get('zero_offset_Z', '0'))
            
            # Obtener ganancias y sensibilidades específicas
            gain_E = float(metadata.get('gain_A_0', '1.0'))
            gain_N = float(metadata.get('gain_A_1', '1.0'))
            gain_Z = float(metadata.get('gain_A_2', '1.0'))
            sens_E = float(metadata.get('custom_sensitivity_A_0', metadata.get('sens', '1.0')))
            sens_N = float(metadata.get('custom_sensitivity_A_1', metadata.get('sens', '1.0')))
            sens_Z = float(metadata.get('custom_sensitivity_A_2', metadata.get('sens', '1.0')))
            
            # Aplicar offsets, sensibilidad y ganancia a cada canal
            # Convertir a m/s² (aceleración de la gravedad ≈ 9.81 m/s²)
            acceleration = np.zeros_like(data_array, dtype=np.float64)
            
            # Para cada canal: (valor - offset) * sensibilidad * ganancia
            # La sensibilidad está en V/g, por lo que multiplicamos por 9.81 para obtener m/s²
            g = 9.81  # aceleración de la gravedad en m/s²
            acceleration[:, 0] = (data_array[:, 0] - zero_offset_E) * sens_E * gain_E * g
            acceleration[:, 1] = (data_array[:, 1] - zero_offset_N) * sens_N * gain_N * g
            acceleration[:, 2] = (data_array[:, 2] - zero_offset_Z) * sens_Z * gain_Z * g

            sampling_rate = float(metadata.get('sampling_rate', '100'))
            time_array = np.arange(samples_per_channel) / sampling_rate
            
            # Crear diccionario de salida en formato compatible
            result = {
                'time': time_array,
                'E': acceleration[:, 0],  # Canal Este
                'N': acceleration[:, 1],  # Canal Norte
                'Z': acceleration[:, 2],  # Canal Vertical
                'components': ['E', 'N', 'Z'],
                'metadata': metadata,
                'name': Path(self.file_path).name
            }
            
            return result
            
        except Exception as e:
            raise IOError(f"Error al leer el archivo {self.file_path}: {str(e)}")
    
    def _extract_metadata(self, ss_file_path):
        """
        Extrae metadatos del archivo .ss asociado
        
        Args:
            ss_file_path: Ruta al archivo .ss
            
        Returns:
            dict: Diccionario con metadatos en formato estándar
        """
        metadata = {}
        try:
            with open(ss_file_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        metadata[key.strip('"')] = value.strip('"')
                        
            # Añadir información adicional
            metadata['format'] = 'MS/SS'
            metadata['unit'] = 'm/s/s'
            
            # Si no hay sampling_rate, usar un valor por defecto
            if 'sampling_rate' not in metadata:
                metadata['sampling_rate'] = '100'
                
        except Exception as e:
            # Si no se puede leer el archivo .ss, crear metadatos básicos
            metadata = {
                'format': 'MS/SS',
                'unit': 'm/s/s',
                'sampling_rate': '100',
                'sensor_name': 'Unknown',
                'station': 'Unknown',
            }
            
        return metadata


def get_reader_for_file(file_path):
    """
    Devuelve el lector adecuado para el tipo de archivo
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        BaseReader: Instancia del lector apropiado
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.ms':
        return MSReader(file_path)
    elif extension == '.sac':
        return SACReader(file_path)
    elif extension == '.mseed' or extension == '.miniseed':
        return MiniSEEDReader(file_path)
    elif extension == '.sgy' or extension == '.segy':
        return SEGYReader(file_path)
    elif extension in ['.txt', '.csv', '.dat', '.asc']:
        return ASCIIReader(file_path)
    else:
        # Intentar determinar el formato por el contenido
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                
                # Verificar si es SAC
                if len(header) >= 4 and struct.unpack('<f', header[0:4])[0] > 0:
                    return SACReader(file_path)
                
                # Verificar si es miniSEED
                if header.startswith(b'000001') or header.startswith(b'000002'):
                    return MiniSEEDReader(file_path)
                
                # Verificar si es SEG-Y
                f.seek(0)
                if len(f.read(3200)) == 3200:  # Encabezado textual de SEG-Y
                    return SEGYReader(file_path)
                
                # Por defecto, intentar como ASCII
                return ASCIIReader(file_path)
                
        except Exception:
            # Si no podemos determinar, intentar como ASCII
            return ASCIIReader(file_path)
