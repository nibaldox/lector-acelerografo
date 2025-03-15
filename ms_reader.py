import numpy as np
import struct

class MSReader:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def read_data(self):
        """
        Lee el archivo binario .ms y retorna los datos de aceleración
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
            ss_file_path = self.file_path.replace('.ms', '.ss')
            metadata = {}
            try:
                with open(ss_file_path, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            metadata[key.strip('"')] = value.strip('"')
            except:
                pass

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
            
            return {
                'time': time_array,
                'E': acceleration[:, 0],  # Canal Este
                'N': acceleration[:, 1],  # Canal Norte
                'Z': acceleration[:, 2],  # Canal Vertical
                'metadata': metadata
            }
            
        except Exception as e:
            raise Exception(f"Error al leer el archivo {self.file_path}: {str(e)}")
    
    @staticmethod
    def get_sampling_rate(ss_file_path):
        """
        Lee la frecuencia de muestreo del archivo .ss asociado
        """
        try:
            with open(ss_file_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'sampling_rate' in line:
                        return int(line.split('=')[1])
            return 100  # valor por defecto
        except:
            return 100  # valor por defecto si hay error
