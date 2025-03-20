"""
Módulo que define las estructuras de datos estándar para la aplicación.

Este módulo proporciona clases de datos para representar registros sísmicos,
metadatos y resultados de análisis, asegurando consistencia en toda la aplicación.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime
from pathlib import Path

@dataclass
class SeismicMetadata:
    """Metadatos de un registro sísmico."""
    
    # Información del evento
    event_name: str = ""
    event_date: Optional[datetime] = None
    event_location: str = ""
    event_magnitude: Optional[float] = None
    event_depth: Optional[float] = None
    
    # Información de la estación
    station_name: str = ""
    station_location: str = ""
    station_latitude: Optional[float] = None
    station_longitude: Optional[float] = None
    station_elevation: Optional[float] = None
    
    # Información del instrumento
    instrument_type: str = ""
    instrument_serial: str = ""
    sensor_type: str = ""
    sampling_rate: float = 0.0
    units: str = "m/s²"
    gain: float = 1.0
    sensitivity: float = 1.0
    
    # Información del archivo
    file_format: str = ""
    file_path: Optional[Path] = None
    original_file: Optional[Path] = None
    
    # Campos adicionales
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte los metadatos a un diccionario."""
        result = {}
        for key, value in self.__dict__.items():
            if key == 'extra_fields':
                result.update(value)
            elif key == 'event_date' and value is not None:
                result[key] = value.isoformat()
            elif key in ('file_path', 'original_file') and value is not None:
                result[key] = str(value)
            elif value is not None:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeismicMetadata':
        """Crea una instancia de SeismicMetadata a partir de un diccionario."""
        # Extraer campos conocidos
        known_fields = {k: v for k, v in data.items() 
                       if k in cls.__annotations__ and k != 'extra_fields'}
        
        # Convertir campos específicos
        if 'event_date' in known_fields and isinstance(known_fields['event_date'], str):
            try:
                known_fields['event_date'] = datetime.fromisoformat(known_fields['event_date'])
            except ValueError:
                known_fields['event_date'] = None
                
        if 'file_path' in known_fields and isinstance(known_fields['file_path'], str):
            known_fields['file_path'] = Path(known_fields['file_path'])
            
        if 'original_file' in known_fields and isinstance(known_fields['original_file'], str):
            known_fields['original_file'] = Path(known_fields['original_file'])
        
        # Extraer campos extra
        extra_fields = {k: v for k, v in data.items() 
                       if k not in cls.__annotations__}
        
        return cls(**known_fields, extra_fields=extra_fields)


@dataclass
class SeismicComponent:
    """Datos de un componente sísmico (N, E, Z)."""
    
    name: str  # Nombre del componente (N, E, Z, etc.)
    acceleration: np.ndarray  # Datos de aceleración
    velocity: Optional[np.ndarray] = None  # Datos de velocidad (calculados)
    displacement: Optional[np.ndarray] = None  # Datos de desplazamiento (calculados)
    
    # Estadísticas
    pga: Optional[float] = None  # Peak Ground Acceleration
    pgv: Optional[float] = None  # Peak Ground Velocity
    pgd: Optional[float] = None  # Peak Ground Displacement
    
    # Información adicional
    orientation: Optional[float] = None  # Orientación en grados
    is_filtered: bool = False
    filter_params: Dict[str, Any] = field(default_factory=dict)
    
    def compute_stats(self):
        """Calcula estadísticas básicas para el componente."""
        if self.acceleration is not None and len(self.acceleration) > 0:
            self.pga = np.max(np.abs(self.acceleration))
        
        if self.velocity is not None and len(self.velocity) > 0:
            self.pgv = np.max(np.abs(self.velocity))
            
        if self.displacement is not None and len(self.displacement) > 0:
            self.pgd = np.max(np.abs(self.displacement))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el componente a un diccionario (sin arrays)."""
        result = {}
        for key, value in self.__dict__.items():
            if key in ('acceleration', 'velocity', 'displacement'):
                # No incluir arrays grandes
                result[f"{key}_length"] = len(value) if value is not None else 0
            elif key == 'filter_params':
                result.update(value)
            elif value is not None:
                result[key] = value
        return result


@dataclass
class SeismicRecord:
    """Registro sísmico completo con todos sus componentes y metadatos."""
    
    time: np.ndarray  # Vector de tiempo
    components: Dict[str, SeismicComponent] = field(default_factory=dict)  # Componentes por nombre
    metadata: SeismicMetadata = field(default_factory=SeismicMetadata)
    
    # Resultados de análisis
    fft_results: Dict[str, Any] = field(default_factory=dict)
    response_spectrum: Dict[str, Any] = field(default_factory=dict)
    detected_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_component(self, name: str, data: np.ndarray):
        """
        Añade un componente al registro.
        
        Args:
            name (str): Nombre del componente (N, E, Z)
            data (np.ndarray): Datos de aceleración
        """
        component = SeismicComponent(name=name, acceleration=data)
        component.compute_stats()
        self.components[name] = component
    
    def get_component_names(self) -> List[str]:
        """Obtiene los nombres de los componentes disponibles."""
        return list(self.components.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el registro a un diccionario compatible con el formato
        anterior para mantener compatibilidad.
        """
        result = {
            'time': self.time,
            'name': self.metadata.event_name or "Sin nombre",
            'metadata': self.metadata.to_dict()
        }
        
        # Añadir componentes en formato compatible
        for name, component in self.components.items():
            result[f'{name}_aceleracion'] = component.acceleration
            if component.velocity is not None:
                result[f'{name}_velocidad'] = component.velocity
            if component.displacement is not None:
                result[f'{name}_desplazamiento'] = component.displacement
        
        # Añadir resultados de análisis
        if self.fft_results:
            result['fft_results'] = self.fft_results
        if self.response_spectrum:
            result['response_spectrum'] = self.response_spectrum
        if self.detected_events:
            result['detected_events'] = self.detected_events
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeismicRecord':
        """
        Crea una instancia de SeismicRecord a partir del formato de diccionario
        utilizado anteriormente.
        """
        # Extraer tiempo
        time = data.get('time', np.array([]))
        
        # Extraer metadatos
        metadata_dict = data.get('metadata', {})
        if not metadata_dict and 'name' in data:
            metadata_dict = {'event_name': data['name']}
        metadata = SeismicMetadata.from_dict(metadata_dict)
        
        # Crear registro
        record = cls(time=time, metadata=metadata)
        
        # Extraer componentes
        for comp in ['N', 'E', 'Z']:
            acc_key = f'{comp}_aceleracion'
            vel_key = f'{comp}_velocidad'
            disp_key = f'{comp}_desplazamiento'
            
            if acc_key in data:
                component = SeismicComponent(
                    name=comp,
                    acceleration=data[acc_key],
                    velocity=data.get(vel_key),
                    displacement=data.get(disp_key)
                )
                component.compute_stats()
                record.components[comp] = component
        
        # Extraer resultados de análisis
        if 'fft_results' in data:
            record.fft_results = data['fft_results']
        if 'response_spectrum' in data:
            record.response_spectrum = data['response_spectrum']
        if 'detected_events' in data:
            record.detected_events = data['detected_events']
            
        return record
