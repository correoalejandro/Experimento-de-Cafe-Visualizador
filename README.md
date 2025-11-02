# Experimento Sensorial de Café

Aplicación Streamlit para explorar y analizar los resultados de un experimento sensorial de café. La interfaz permite cargar datos propios o utilizar el conjunto de ejemplo incluido (`MuestreoCafe_merged.csv`) para realizar análisis exploratorio y pruebas de hipótesis.

## Requisitos

La aplicación está lista para desplegarse en [Streamlit Community Cloud](https://streamlit.io/cloud). Las dependencias necesarias están definidas en `requirements.txt`:

```text
streamlit>=1.32
pandas>=2.0
numpy>=1.24
scipy>=1.10
altair>=5.0
```

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_experimento_cafe.py
```

La aplicación cargará automáticamente el archivo de ejemplo `MuestreoCafe_merged.csv` si no se proporciona uno propio (puedes sustituirlo desde la barra lateral con un archivo subido o una ruta local).

## Despliegue en Streamlit Cloud

Antes de publicar, verifica esta lista rápida:

- `app_experimento_cafe.py` en la raíz del repositorio (es el entrypoint que Streamlit ejecutará).
- `requirements.txt` con las dependencias de la app.
- `MuestreoCafe_merged.csv` (incluido) o tu propio CSV de ejemplo para que la app tenga datos iniciales.

Con todo listo:

1. Sube este repositorio a GitHub.
2. Crea una nueva aplicación en Streamlit Cloud y selecciona el repositorio.
3. Indica `app_experimento_cafe.py` como archivo principal.
4. Publica la aplicación.

Streamlit instalará automáticamente las dependencias usando `requirements.txt`, ejecutará el script y la barra lateral ofrecerá la descarga del CSV de ejemplo para validar el formato.
