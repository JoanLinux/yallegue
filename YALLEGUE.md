# YaLlegué — Checador de Asistencia Facial

Sistema de control de asistencia por reconocimiento facial en tiempo real.  
Stack: **Python / Flask · InsightFace · OpenCV · SQLite · HTML/JS vanilla**

---

## Arquitectura

```
yallegue/
├── server.py           # Backend Flask (puerto 5252)
├── static/
│   ├── admin.html      # Panel de administración
│   └── pantalla.html   # Pantalla TV (recepción / lobby)
├── uploads/faces/      # Fotos de personas registradas
├── yallegue.db         # SQLite (persons + attendance)
├── start.bat           # Launcher Windows
└── requirements.txt
```

### Threads en server.py

| Thread | Función |
|--------|---------|
| `stream_worker` | Lee frames de la cámara/video, publica en `latest_frame` |
| `detect_worker` | Lee `latest_frame`, corre InsightFace cada 0.5 s, actualiza `_last_faces` y `state["current"]` |
| Flask | Sirve `/api/frame` JPEG y todos los endpoints REST |

La separación stream/detect evita que la inferencia CPU bloquee el flujo de video.

---

## Base de datos

### `persons`
| col | tipo | notas |
|-----|------|-------|
| id | INTEGER PK | |
| name | TEXT | |
| role | TEXT | |
| face_path | TEXT | filename en `uploads/faces/` |
| embedding | BLOB | float32 InsightFace (512 dims) |
| blocked | INTEGER | 0=activo · 1=acceso denegado |
| created_at | TIMESTAMP | |

### `attendance`
| col | tipo | notas |
|-----|------|-------|
| id | INTEGER PK | |
| person_id | INTEGER FK | |
| type | TEXT | `"entrada"` ó `"salida"` |
| timestamp | TIMESTAMP | |

---

## Lógica de detección

```
detect_worker (cada 0.5 s)
│
├─ load_embeddings()          ← incluye blocked=1
├─ fa.get(frame_640x360)      ← InsightFace buffalo_sc
│
├─ para cada cara detectada:
│   ├─ cosine_sim vs todos los embeddings
│   ├─ umbral: sim > 0.42 → matched
│   │
│   ├─ [blocked=1 AND cam_type="entrada"]
│   │     → state["current"] = {access_denied: True}
│   │     → NO INSERT en attendance
│   │
│   ├─ [blocked=1 AND cam_type="salida"]
│   │     → flujo normal (puede registrar salida)
│   │
│   └─ [no bloqueado]
│         → checa último registro del día
│         → INSERT solo si last_type != cam_type   ← alternancia entrada/salida
│         → state["current"] = {access_denied: False}
│
├─ _currently_present[pid] = now     ← presencia activa
├─ expirar pids sin ver > 5 s        ← PRESENCE_TIMEOUT
└─ state["current"] = None si nadie presente
```

---

## Endpoints principales

| método | ruta | descripción |
|--------|------|-------------|
| POST | `/start` | Inicia stream. Body: `{source, loop, cam_type}` |
| POST | `/stop` | Detiene stream, limpia estado |
| POST | `/api/set-cam-type` | Cambia entrada/salida en caliente. Limpia `_currently_present` |
| GET | `/status` | Estado actual: `{running, fps, current, cam_type, error}` |
| GET | `/api/frame` | JPEG único (canvas polling, evita spinner de browser) |
| GET | `/api/persons` | Lista personas con campo `blocked` |
| POST | `/api/persons` | Registra persona (multipart: name, role, face, face_bbox) |
| PUT | `/api/persons/<id>` | Edita nombre/rol/foto. Llama `load_embeddings()` al terminar |
| DELETE | `/api/persons/<id>` | Elimina persona y asistencia |
| POST | `/api/persons/<id>/block` | Body: `{blocked: true/false}` |
| GET | `/api/attendance/by-person` | `?date=YYYY-MM-DD` → personas + accesos del día |
| POST | `/api/attendance/manual` | Acceso manual desde modal de historial |
| POST | `/api/detect-face-box` | Detección de cajas para preview de registro |

---

## Pantallas

### `admin.html` — Panel de administración

**Tabs:**
- **Cámara** — preview de video (canvas polling 100 ms), toggle entrada/salida, estado de detección en tiempo real
- **Personas** — registro con detección automática de cara, selección multi-cara, lista con badge BLOQUEADO y botón Editar
- **Asistencia** — grid por persona del día actual, modal con historial + acceso manual
- **Historial** — igual que asistencia pero para día anterior (configurable)

**Modal Editar persona:**  
Nombre · Puesto · Nueva foto (con selección de cara igual que registro) · Bloquear/Desbloquear · Eliminar

### `pantalla.html` — TV de lobby

| estado | pantalla |
|--------|---------|
| Sin actividad | Screensaver: reloj + fecha |
| Entrada exitosa | Foto · "Ya llegó" · hora · ✅ Asistencia registrada |
| Salida exitosa | Foto · "Salió" · hora · 🚪 Salida registrada |
| Acceso denegado | Foto de fondo · badge rojo 🚫 "Acceso denegado" (sin panel de info) |

**Timeout screensaver:** 5 minutos sin detección → vuelve al reloj.  
La foto de la última persona queda en pantalla hasta que llegue alguien más o pasen los 5 min.

---

## Reglas de negocio clave

1. **Una entrada por llegada:** persona visible → se registra solo al aparecer (`_currently_present`). Mientras siga en frame no se vuelve a registrar.
2. **Alternancia entrada/salida:** no se puede registrar dos entradas seguidas; el sistema solo inserta si el tipo es diferente al último registro del día.
3. **Acceso denegado:** solo en cámara de entrada. En salida, persona bloqueada sale normalmente.
4. **Sin duplicados en pantalla:** `_shownEvents` Set browser-side previene mostrar la misma detección dos veces en la misma sesión.
5. **Cambio de tipo en caliente:** toggle entrada/salida llama `/api/set-cam-type` inmediatamente, limpia presencias activas para re-evaluar.

---

## Requisitos

```
flask>=3.0
insightface>=0.7
onnxruntime>=1.17   # CPU — para GPU instalar onnxruntime-gpu cuando CUDA 13 sea soportado
opencv-python>=4.9
numpy>=1.24
werkzeug>=3.0
```

> **Nota GPU:** El sistema tiene RTX 5080 + CUDA 13.3, pero `onnxruntime-gpu` no tiene build  
> oficial para CUDA 13.x aún. Cuando esté disponible: `pip install onnxruntime-gpu`.  
> Con GPU la inferencia baja de ~150 ms → ~10 ms por frame.

---

## Iniciar

```bat
start.bat
```

- Admin: http://localhost:5252/admin  
- Pantalla TV: http://localhost:5252/pantalla
