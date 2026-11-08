# YaLlegué ✅

Checador de asistencia con reconocimiento facial. Detecta personas al entrar/salir, registra hora automáticamente y muestra bienvenida en pantalla — sin internet, sin tarjetas, sin huellas.

---

## ¿Qué hace?

| Pantalla | Función |
|----------|---------|
| `/pantalla` | Pantalla de bienvenida — muestra nombre + foto + hora al detectar a alguien |
| `/admin` | Panel admin — personas registradas, registros de asistencia, control de cámara |

**Flujo:**
1. Persona pasa frente a la cámara
2. InsightFace genera embedding facial y lo compara con la BD
3. Si hay match → registra entrada o salida en `attendance`
4. Pantalla muestra bienvenida con nombre y foto
5. Admin puede bloquear personas o ver historial completo

---

## Stack

- **Python 3.10+** con Flask
- **InsightFace** (buffalo_sc) — reconocimiento facial por embeddings coseno
- **OpenCV** — captura de cámara / video
- **SQLite** — personas, asistencia
- **100% offline** — modelos bundleados, sin llamadas a internet

---

## Instalación

```bash
# 1. Clonar
git clone https://github.com/JoanLinux/yallegue.git
cd yallegue

# 2. Entorno virtual
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Dependencias
pip install -r requirements.txt

# 4. Modelos InsightFace (descargar y colocar en models/)
# models/insightface/models/buffalo_sc/
#   • det_500m.onnx
#   • w600k_mbf.onnx
```

---

## Dependencias (`requirements.txt`)

```
flask>=3.0
insightface>=0.7
onnxruntime>=1.17
opencv-python>=4.9
numpy>=1.24
werkzeug>=3.0
```

---

## Correr

```bat
start.bat
```

O directamente:

```bash
python server.py
```

Accesos:
- **Admin:** http://localhost:5252/admin
- **Pantalla:** http://localhost:5252/pantalla

---

## Estructura

```
yallegue/
├── server.py              # Backend Flask + lógica de visión
├── start.bat              # Lanzador Windows
├── requirements.txt       # Dependencias Python
├── static/
│   ├── admin.html         # Panel de administración
│   └── pantalla.html      # Vista de bienvenida
├── uploads/
│   └── faces/             # Fotos de personas registradas (gitignored)
├── models/
│   └── insightface/       # Modelos InsightFace (gitignored)
└── yallegue.db            # Base de datos SQLite (gitignored)
```

---

## Base de datos

| Tabla | Columnas clave | Descripción |
|-------|----------------|-------------|
| `persons` | name, role, face_path, embedding, blocked | Personas registradas con embedding facial |
| `attendance` | person_id, type (entrada/salida), timestamp | Registro de asistencia con hora exacta |

---

## API principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/video_feed` | Stream MJPEG de la cámara |
| GET | `/status` | Estado del sistema + persona detectada actualmente |
| POST | `/start` | Iniciar detección |
| POST | `/stop` | Detener detección |
| GET | `/api/persons` | Listar personas registradas |
| POST | `/api/persons` | Registrar persona + foto |
| DELETE | `/api/persons/<id>` | Eliminar persona |
| GET | `/api/attendance` | Historial de asistencia |
| POST | `/api/attendance/export` | Exportar asistencia |

---

## Tipos de cámara

El sistema soporta:
- **Webcam USB** — `source: "0"` (o índice de cámara)
- **Archivo de video** — ruta al `.mp4` para pruebas
- **Cámara IP** — URL RTSP

Configurable desde el panel admin o via API con `cam_type: "entrada"` o `"salida"`.

---

## Requisitos de hardware

- Cámara USB o IP
- CPU moderna — no requiere GPU
- ~300 MB RAM para modelos

---

## Proyecto relacionado

- [TuPlatilloFue](https://github.com/JoanLinux/tuplatillofue) — reconocimiento facial para restaurantes con la misma tecnología
