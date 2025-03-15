import pandas as pd
import json
import plotly
import numpy as np
from pathlib import Path

class DataExporter:
    def __init__(self, output_dir="exports"):
        """
        Inicializa el exportador de datos
        Args:
            output_dir: Directorio donde se guardarán las exportaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_raw_data(self, data, filename, format='csv'):
        """
        Exporta datos crudos a diferentes formatos
        Args:
            data: Diccionario con los datos del registro
            filename: Nombre base del archivo
            format: Formato de exportación ('csv', 'excel', 'json')
        Returns:
            path: Ruta del archivo guardado
        """
        # Crear DataFrame
        df = pd.DataFrame({
            'time': data['time'],
            'E': data['E'],
            'N': data['N'],
            'Z': data['Z']
        })
        
        # Exportar según formato
        if format == 'csv':
            output_path = self.output_dir / f"{filename}.csv"
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            output_path = self.output_dir / f"{filename}.xlsx"
            df.to_excel(output_path, index=False)
        elif format == 'json':
            output_path = self.output_dir / f"{filename}.json"
            df.to_json(output_path, orient='records')
        else:
            raise ValueError(f"Formato no soportado: {format}")
            
        return output_path
    
    def export_analysis_results(self, data, analysis_type, results, filename):
        """
        Exporta resultados de análisis
        Args:
            data: Datos originales
            analysis_type: Tipo de análisis ('fft', 'events', 'filtered')
            results: Resultados del análisis
            filename: Nombre base del archivo
        Returns:
            path: Ruta del archivo guardado
        """
        output_path = self.output_dir / f"{filename}_{analysis_type}_results.json"
        
        export_data = {
            'type': analysis_type,
            'metadata': data.get('metadata', {}),
            'results': {}
        }
        
        if analysis_type == 'fft':
            export_data['results'] = {
                'frequencies': results['frequencies'].tolist(),
                'magnitudes': results['magnitudes'].tolist(),
                'phase': results['phase'].tolist() if 'phase' in results else None
            }
        elif analysis_type == 'events':
            export_data['results'] = {
                'events': [
                    {
                        'time': event['time'],
                        'features': event['features']
                    }
                    for event in results['events']
                ]
            }
        elif analysis_type == 'filtered':
            export_data['results'] = {
                'original': data[results['component']].tolist(),
                'filtered': results['filtered_data'].tolist(),
                'filter_params': results['filter_params']
            }
            
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        return output_path
    
    def export_plot(self, fig, filename):
        """
        Exporta gráficos de Plotly
        Args:
            fig: Figura de Plotly
            filename: Nombre base del archivo
        Returns:
            paths: Lista de rutas de los archivos guardados
        """
        paths = []
        
        # Exportar como HTML interactivo
        html_path = self.output_dir / f"{filename}.html"
        fig.write_html(html_path)
        paths.append(html_path)
        
        # Exportar como imagen estática
        png_path = self.output_dir / f"{filename}.png"
        fig.write_image(png_path)
        paths.append(png_path)
        
        # Exportar datos del gráfico como JSON
        json_path = self.output_dir / f"{filename}_plot_data.json"
        with open(json_path, 'w') as f:
            json.dump(fig.to_plotly_json(), f, indent=2)
        paths.append(json_path)
        
        return paths
