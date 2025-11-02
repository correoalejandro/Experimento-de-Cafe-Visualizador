# Experimento Sensorial de Café

Aplicación Streamlit para explorar y analizar los resultados de un experimento sensorial de café. La interfaz permite cargar datos propios o utilizar el conjunto de ejemplo incluido (`MuestreoCafe.csv`) para realizar análisis exploratorio y pruebas de hipótesis.

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

La aplicación cargará automáticamente el archivo de ejemplo si no se proporciona uno propio.

## Despliegue en Streamlit Cloud

1. Sube este repositorio a GitHub.
2. Crea una nueva aplicación en Streamlit Cloud y selecciona el repositorio.
3. Indica `app_experimento_cafe.py` como archivo principal.
4. Publica la aplicación.

Streamlit instalará automáticamente las dependencias usando `requirements.txt` y ejecutará la aplicación.
