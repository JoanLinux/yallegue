"""
YaLlegué — Checador de asistencia con detección facial
Accede en: http://localhost:5252
Pantalla:   http://localhost:5252/pantalla
Admin:      http://localhost:5252/admin
"""

import os
os.environ["INSIGHTFACE_HOME"] = os.path.join(os.path.dirname(__file__), "models", "insightface")

import threading, time, json, sqlite3, base64, logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

# Suppress noisy poll endpoints from console log
class _SilentPollFilter(logging.Filter):
    _SKIP = {"/status", "/video_feed", "/api/frame"}
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self._SKIP)

logging.getLogger("werkzeug").addFilter(_SilentPollFilter())
import numpy as np
import cv2

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "yallegue.db"
FACES_DIR  = BASE_DIR / "uploads" / "faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")

# ── STATE ─────────────────────────────────────────────────────────────────────
state = {
    "running":   False,
    "current":   None,   # last detected person (shown on pantalla)
    "fps":       0.0,
    "error":     "",
    "source":    "0",
    "loop":      False,
    "cam_type":  "entrada",
}
frame_lock     = threading.Lock()
latest_frame   = None
stream_thread  = None
detect_thread  = None
_detect_frame  = None   # frame queued for detection (set by capture thread)
_detect_lock   = threading.Lock()

# ── DB ────────────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            role       TEXT DEFAULT "",
            face_path  TEXT,
            embedding  BLOB,
            blocked    INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id  INTEGER NOT NULL,
            type       TEXT NOT NULL DEFAULT 'entrada',
            timestamp  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            FOREIGN KEY(person_id) REFERENCES persons(id)
        );
    """)
    con.commit()
    # migration: add blocked column to existing DBs
    try:
        con.execute("ALTER TABLE persons ADD COLUMN blocked INTEGER NOT NULL DEFAULT 0")
        con.commit()
    except Exception:
        pass
    con.close()

init_db()

def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

# ── FACE ENGINE ───────────────────────────────────────────────────────────────
_face_app = None
_face_lock = threading.Lock()

def get_face_app():
    global _face_app
    with _face_lock:
        if _face_app is None:
            try:
                from insightface.app import FaceAnalysis
                _face_app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
                _face_app.prepare(ctx_id=0, det_size=(640,640))
                print("[INFO] InsightFace listo.")
            except Exception as e:
                print(f"[ERROR] InsightFace: {e}")
                _face_app = None
    return _face_app

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def load_embeddings():
    con = get_db()
    rows = con.execute("SELECT id, name, role, face_path, embedding, blocked FROM persons WHERE embedding IS NOT NULL").fetchall()
    con.close()
    result = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        result.append({"id": r["id"], "name": r["name"], "role": r["role"],
                       "face_path": r["face_path"], "embedding": emb,
                       "blocked": bool(r["blocked"])})
    return result

# ── STREAM THREAD ─────────────────────────────────────────────────────────────
PRESENCE_TIMEOUT = 5.0   # seconds without detection → person left scene
_currently_present = {}  # person_id -> last_seen_time (active in frame)
_last_faces = []         # [{x1,y1,x2,y2,name,matched}] drawn on every frame

def stream_worker(source):
    global latest_frame, _detect_frame
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        state["error"] = f"No se pudo abrir: {source}"
        state["running"] = False
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps and src_fps > 0:
        frame_interval = 1.0 / min(src_fps, 30.0)
    else:
        frame_interval = 1.0 / 30.0
    is_file = not isinstance(src, int) and not str(src).startswith("rtsp") and not str(src).startswith("http")

    state["error"] = ""
    fps_counter, fps_start = 0, time.time()
    last_frame_time = 0.0

    while state["running"]:
        if is_file:
            now_wait = time.time()
            elapsed = now_wait - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            if state["loop"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_frame_time = 0.0
                continue
            time.sleep(0.05)
            continue

        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 2:
            state["fps"] = round(fps_counter / elapsed, 1)
            fps_counter, fps_start = 0, time.time()

        with frame_lock:
            latest_frame = frame.copy()
        # post frame for detect thread (non-blocking: drop if detect is busy)
        with _detect_lock:
            _detect_frame = frame.copy()

    cap.release()
    state["running"] = False


def detect_worker():
    global _detect_frame
    fa = get_face_app()
    last_detect_time = 0.0

    while state["running"]:
        # throttle: run detection at most every 0.5s
        now_t = time.time()
        if now_t - last_detect_time < 0.5:
            time.sleep(0.05)
            continue

        with _detect_lock:
            frame = _detect_frame
            _detect_frame = None

        if frame is None or not fa:
            time.sleep(0.05)
            continue

        last_detect_time = time.time()
        known = load_embeddings()
        if not known:
            continue

        small = cv2.resize(frame, (640, 360))
        faces = fa.get(small)
        scale_x = frame.shape[1] / 640.0
        scale_y = frame.shape[0] / 360.0
        new_faces = []
        detected_pids = set()
        now_t2 = time.time()

        for face in faces:
            emb = face.embedding
            best_id, best_sim = None, 0.0
            for k in known:
                sim = cosine_sim(emb, k["embedding"])
                if sim > best_sim:
                    best_sim, best_id = sim, k
            b = face.bbox.astype(int)
            new_faces.append({
                "x1": int(b[0]*scale_x), "y1": int(b[1]*scale_y),
                "x2": int(b[2]*scale_x), "y2": int(b[3]*scale_y),
                "name": best_id["name"] if (best_id and best_sim > 0.42) else "?",
                "matched": best_id is not None and best_sim > 0.42,
            })
            if best_id and best_sim > 0.42:
                pid = best_id["id"]
                detected_pids.add(pid)
                if best_id.get("blocked") and state["cam_type"] == "entrada":
                    # blocked on entry only — show denial, never record
                    if pid not in _currently_present:
                        state["current"] = {
                            "person_id":   pid,
                            "name":        best_id["name"],
                            "role":        best_id["role"],
                            "face_path":   best_id["face_path"],
                            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
                            "detected_at": now_t2,
                            "access_denied": True,
                        }
                    _currently_present[pid] = now_t2
                else:
                    if pid not in _currently_present:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        con = get_db()
                        cam_type = state["cam_type"]
                        last_row = con.execute(
                            "SELECT type FROM attendance WHERE person_id=? ORDER BY timestamp DESC LIMIT 1",
                            (pid,)
                        ).fetchone()
                        last_type = last_row["type"] if last_row else None
                        if last_type != cam_type and state["running"]:
                            con.execute(
                                "INSERT INTO attendance(person_id, type, timestamp) VALUES(?,?,?)",
                                (pid, cam_type, ts)
                            )
                            con.commit()
                            con.close()
                            state["current"] = {
                                "person_id":   pid,
                                "name":        best_id["name"],
                                "role":        best_id["role"],
                                "face_path":   best_id["face_path"],
                                "timestamp":   ts,
                                "detected_at": now_t2,
                                "access_denied": False,
                            }
                        else:
                            con.close()
                    _currently_present[pid] = now_t2

        _last_faces[:] = new_faces
        expired = [pid for pid, last in _currently_present.items()
                   if now_t2 - last > PRESENCE_TIMEOUT]
        for pid in expired:
            del _currently_present[pid]

        if state["current"] and not _currently_present:
            state["current"] = None

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "admin.html")

@app.route("/pantalla")
def pantalla():
    return send_from_directory("static", "pantalla.html")

@app.route("/admin")
def admin():
    return send_from_directory("static", "admin.html")

@app.route("/status")
def status():
    alive = stream_thread is not None and stream_thread.is_alive()
    if state["running"] and not alive:
        state["running"] = False
    return jsonify({
        "running":  state["running"],
        "fps":      state["fps"],
        "error":    state["error"],
        "current":  state["current"],
        "cam_type": state["cam_type"],
    })

@app.route("/start", methods=["POST"])
def start():
    global stream_thread, detect_thread
    if state["running"]:
        return jsonify({"ok": True, "msg": "ya corriendo"})
    data = request.json or {}
    state["source"]   = data.get("source", "0")
    state["loop"]     = bool(data.get("loop", False))
    state["cam_type"] = data.get("cam_type", "entrada")
    state["running"] = True
    state["error"]   = ""
    state["current"] = None
    _currently_present.clear()
    _last_faces.clear()
    stream_thread = threading.Thread(target=stream_worker, args=(state["source"],), daemon=True)
    stream_thread.start()
    detect_thread = threading.Thread(target=detect_worker, daemon=True)
    detect_thread.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    global _detect_frame
    state["running"] = False
    state["current"] = None
    _currently_present.clear()
    _last_faces.clear()
    _detect_frame = None
    return jsonify({"ok": True})

@app.route("/api/set-cam-type", methods=["POST"])
def set_cam_type():
    t = (request.json or {}).get("cam_type", "entrada")
    if t in ("entrada", "salida"):
        state["cam_type"] = t
        # clear presence so next detection re-evaluates with new type
        _currently_present.clear()
        state["current"] = None
    return jsonify({"ok": True, "cam_type": state["cam_type"]})

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with frame_lock:
                f = latest_frame
            if f is None:
                time.sleep(0.05)
                continue
            draw = f.copy()
            for face in _last_faces:
                if not face["matched"]:
                    continue
                color = (99, 179, 237)
                cv2.rectangle(draw, (face["x1"], face["y1"]), (face["x2"], face["y2"]), color, 2)
                label = face["name"]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(draw, (face["x1"], face["y1"]-th-10), (face["x1"]+tw+8, face["y1"]), color, -1)
                cv2.putText(draw, label, (face["x1"]+4, face["y1"]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10,10,10), 1, cv2.LINE_AA)
            _, buf = cv2.imencode(".jpg", draw, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/frame")
def api_frame():
    """Single JPEG frame — avoids browser tab spinner caused by infinite MJPEG stream."""
    with frame_lock:
        f = latest_frame
    if f is None:
        return Response(status=204)
    draw = f.copy()
    for face in _last_faces:
        if not face["matched"]:
            continue
        color = (99, 179, 237)
        cv2.rectangle(draw, (face["x1"], face["y1"]), (face["x2"], face["y2"]), color, 2)
        label = face["name"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(draw, (face["x1"], face["y1"]-th-10), (face["x1"]+tw+8, face["y1"]), color, -1)
        cv2.putText(draw, label, (face["x1"]+4, face["y1"]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10,10,10), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", draw, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})

# ── PERSONS API ───────────────────────────────────────────────────────────────

@app.route("/api/persons", methods=["GET"])
def list_persons():
    con = get_db()
    rows = con.execute("SELECT id, name, role, face_path, blocked, created_at FROM persons ORDER BY name").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/persons/<int:pid>/block", methods=["POST"])
def block_person(pid):
    data = request.json or {}
    blocked = 1 if data.get("blocked") else 0
    con = get_db()
    con.execute("UPDATE persons SET blocked=? WHERE id=?", (blocked, pid))
    con.commit()
    con.close()
    return jsonify({"ok": True, "blocked": blocked})

@app.route("/api/persons", methods=["POST"])
def add_person():
    name = request.form.get("name", "").strip()
    role = request.form.get("role", "").strip()
    if not name:
        return jsonify({"error": "nombre requerido"}), 400

    face_path, embedding = None, None
    if "face" in request.files:
        f = request.files["face"]
        if f.filename:
            fname = secure_filename(f.filename)
            p = FACES_DIR / fname
            f.save(str(p))
            face_path = fname
            # compute embedding
            fa = get_face_app()
            if fa:
                img = cv2.imread(str(p))
                if img is not None:
                    faces = fa.get(img)
                    if faces:
                        # use bbox hint if provided (user selected from multiple faces)
                        chosen = faces[0]
                        bbox_hint = request.form.get("face_bbox")
                        if bbox_hint and len(faces) > 1:
                            try:
                                import json as _json
                                bh = _json.loads(bbox_hint)
                                cx, cy = bh["x"] + bh["w"]/2, bh["y"] + bh["h"]/2
                                best_dist = float("inf")
                                for fc in faces:
                                    fx = (fc.bbox[0]+fc.bbox[2])/2
                                    fy = (fc.bbox[1]+fc.bbox[3])/2
                                    dist = (fx-cx)**2 + (fy-cy)**2
                                    if dist < best_dist:
                                        best_dist, chosen = dist, fc
                            except Exception:
                                pass
                        embedding = chosen.embedding.astype(np.float32).tobytes()

    con = get_db()
    cur = con.execute(
        "INSERT INTO persons(name, role, face_path, embedding) VALUES(?,?,?,?)",
        (name, role, face_path, embedding)
    )
    con.commit()
    pid = cur.lastrowid
    con.close()
    return jsonify({"ok": True, "id": pid, "has_embedding": embedding is not None})

@app.route("/api/persons/<int:pid>", methods=["PUT"])
def edit_person(pid):
    name = request.form.get("name", "").strip()
    role = request.form.get("role", "").strip()
    con = get_db()
    if "face" in request.files:
        f = request.files["face"]
        if f.filename:
            fname = secure_filename(f.filename)
            p = FACES_DIR / fname
            f.save(str(p))
            embedding = None
            fa = get_face_app()
            if fa:
                img = cv2.imread(str(p))
                if img is not None:
                    faces = fa.get(img)
                    if faces:
                        embedding = faces[0].embedding.astype(np.float32).tobytes()
            con.execute(
                "UPDATE persons SET name=?, role=?, face_path=?, embedding=? WHERE id=?",
                (name, role, fname, embedding, pid)
            )
    else:
        con.execute("UPDATE persons SET name=?, role=? WHERE id=?", (name, role, pid))
    con.commit()
    con.close()
    load_embeddings()
    return jsonify({"ok": True})

@app.route("/api/persons/<int:pid>", methods=["DELETE"])
def del_person(pid):
    con = get_db()
    con.execute("DELETE FROM persons WHERE id=?", (pid,))
    con.execute("DELETE FROM attendance WHERE person_id=?", (pid,))
    con.commit()
    con.close()
    return jsonify({"ok": True})

# ── ATTENDANCE API ─────────────────────────────────────────────────────────────

@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    date = request.args.get("date", time.strftime("%Y-%m-%d"))
    con = get_db()
    rows = con.execute("""
        SELECT a.id, a.person_id, p.name, p.role, p.face_path,
               a.type, a.timestamp
        FROM attendance a
        JOIN persons p ON p.id = a.person_id
        WHERE DATE(a.timestamp) = ?
        ORDER BY a.timestamp DESC
    """, (date,)).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/attendance/manual", methods=["POST"])
def manual_attendance():
    data = request.json or {}
    pid  = data.get("person_id")
    typ  = data.get("type", "entrada")
    ts   = data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    if not pid:
        return jsonify({"error": "person_id required"}), 400
    con = get_db()
    con.execute("INSERT INTO attendance(person_id, type, timestamp) VALUES(?,?,?)", (pid, typ, ts))
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/attendance/<int:aid>", methods=["DELETE"])
def del_attendance(aid):
    con = get_db()
    con.execute("DELETE FROM attendance WHERE id=?", (aid,))
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/attendance/by-person", methods=["GET"])
def attendance_by_person():
    """Returns distinct persons present on a date, with their access timestamps."""
    date = request.args.get("date", time.strftime("%Y-%m-%d"))
    con = get_db()
    persons = con.execute("""
        SELECT DISTINCT p.id, p.name, p.role, p.face_path
        FROM attendance a
        JOIN persons p ON p.id = a.person_id
        WHERE DATE(a.timestamp) = ?
        ORDER BY p.name
    """, (date,)).fetchall()
    result = []
    for p in persons:
        accesses = con.execute("""
            SELECT type, timestamp FROM attendance
            WHERE person_id=? AND DATE(timestamp)=?
            ORDER BY timestamp ASC
        """, (p["id"], date)).fetchall()
        result.append({
            "id": p["id"], "name": p["name"],
            "role": p["role"], "face_path": p["face_path"],
            "accesses": [dict(a) for a in accesses]
        })
    con.close()
    return jsonify(result)

@app.route("/api/attendance/currently-inside", methods=["GET"])
def currently_inside():
    """Persons whose last attendance record is 'entrada' (regardless of date)."""
    con = get_db()
    rows = con.execute("""
        SELECT p.id, p.name, p.role, p.face_path,
               a.timestamp AS last_ts, a.type AS last_type
        FROM persons p
        JOIN attendance a ON a.id = (
            SELECT id FROM attendance WHERE person_id = p.id ORDER BY timestamp DESC LIMIT 1
        )
        WHERE a.type = 'entrada'
        ORDER BY a.timestamp DESC
    """).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/attendance/today-summary", methods=["GET"])
def today_summary():
    today = time.strftime("%Y-%m-%d")
    con = get_db()
    total_persons = con.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    present_today = con.execute("""
        SELECT COUNT(DISTINCT person_id) FROM attendance WHERE DATE(timestamp)=?
    """, (today,)).fetchone()[0]
    con.close()
    return jsonify({"total": total_persons, "present": present_today, "date": today})

# ── VIDEO UPLOAD ──────────────────────────────────────────────────────────────

VIDEOS_DIR = BASE_DIR / "uploads" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    fname = secure_filename(f.filename)
    p = VIDEOS_DIR / fname
    f.save(str(p))
    return jsonify({"path": str(p), "name": fname})

# ── FACE BOX PREVIEW ──────────────────────────────────────────────────────────

@app.route("/api/detect-face-box", methods=["POST"])
def detect_face_box():
    if "face" not in request.files:
        return jsonify({"boxes": []})
    f = request.files["face"]
    img_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"boxes": []})
    fa = get_face_app()
    if not fa:
        return jsonify({"boxes": []})
    h, w = img.shape[:2]
    faces = fa.get(img)
    boxes = [{"x": int(fc.bbox[0]), "y": int(fc.bbox[1]),
               "w": int(fc.bbox[2]-fc.bbox[0]), "h": int(fc.bbox[3]-fc.bbox[1]),
               "imgW": w, "imgH": h} for fc in faces]
    return jsonify({"boxes": boxes})

# ── FACES STATIC ──────────────────────────────────────────────────────────────

@app.route("/uploads/faces/<path:fname>")
def serve_face(fname):
    return send_from_directory(str(FACES_DIR), fname)

if __name__ == "__main__":
    print("YaLlegué corriendo en http://localhost:5252")
    app.run(host="0.0.0.0", port=5252, debug=False, threaded=True)
