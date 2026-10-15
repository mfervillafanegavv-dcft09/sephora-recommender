# Monitoreo de NEXA

La API y la interfaz Streamlit registran las inferencias reales en
`reports\inference_history.csv`. El archivo no se crea ni se llena con datos de
ejemplo: aparece después de la primera recomendación generada por la interfaz
o de la primera solicitud exitosa a `POST /recommend`.

Abrir o recargar Streamlit no cuenta como una inferencia. Se registra una fila
cuando se envía el formulario **Ver recomendaciones** y el recomendador devuelve
un resultado.

## Iniciar la API

Desde PowerShell:

```powershell
Set-Location "D:\Martin\henry\PROYECTO FINAL"
& "C:\Users\Usuario\AppData\Local\Python\pythoncore-3.15-64\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Deja esta ventana abierta mientras usas la API. Consulta `http://127.0.0.1:8000/docs`
para enviar solicitudes a `POST /recommend`. Se pueden consultar categorías
válidas con `GET /categories`.

Por cada recomendación exitosa, la API agrega una fila al CSV con:

- `timestamp_utc`: fecha y hora de registro en UTC;
- `category`: categoría solicitada;
- `budget_usd`: presupuesto enviado;
- `mode_used`: modo efectivo devuelto por el recomendador.

No se guarda el `user_id`. Las solicitudes inválidas o fallidas no se registran.
La API o Streamlit crean la carpeta `reports` si hace falta, al guardar la
primera inferencia.

## Usar la interfaz Streamlit

Desde PowerShell, ejecuta:

```powershell
Set-Location "D:\Martin\henry\PROYECTO FINAL\Nexa.v4"
& "C:\Users\Usuario\AppData\Local\Python\pythoncore-3.15-64\python.exe" -m streamlit run "app\nexa_streamlit.py"
```

Selecciona categoría y presupuesto, y pulsa **Ver recomendaciones**. Cada
envío registrado corresponde a una ejecución del recomendador; una recarga de
la página sin enviar el formulario no agrega una fila.

## Ejecutar el monitor

Con la API detenida o en otra ventana de PowerShell, ejecuta:

```powershell
& "C:\Users\Usuario\AppData\Local\Python\pythoncore-3.15-64\python.exe" "D:\Martin\henry\PROYECTO FINAL\Sephora_Monitoring.py"
```

Si todavía no hubo solicitudes exitosas, el monitor avisará que no encuentra
`reports\inference_history.csv` y terminará sin generar gráficos. Tras generar
tráfico real, vuelve a ejecutarlo. Los gráficos se guardan en
`reports\monitoring_charts`.

El informe muestra el total de recomendaciones registradas y la distribución
de `mode_used`. También calcula PSI comparando los presupuestos registrados
con los precios del catálogo del modelo; no es una comparación entre dos
períodos históricos de solicitudes.
