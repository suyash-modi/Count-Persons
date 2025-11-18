import os
import importlib
from flask import Flask, render_template_string, Response, jsonify
import cv2, time, threading, queue, numpy as np, math, collections
from openvino.runtime import Core
from pymongo import MongoClient

load_dotenv = lambda *args, **kwargs: None  
dotenv_spec = importlib.util.find_spec("dotenv")
if dotenv_spec is not None:
    load_dotenv = importlib.import_module("dotenv").load_dotenv

# ==================== CONFIGURATION ====================
load_dotenv()
CAM_SOURCES = ["assets/video1.mp4", "assets/video7.mp4"]
DETECTION_MODEL = "models/person-detection-retail-0013.xml"
REID_MODEL = "models/person-reidentification-retail-0287.xml"

REID_SIM_THRESHOLD = 0.75
REID_UPDATE_MOMENTUM = 0.6
GROUP_DISTANCE_THRESHOLD = 120
FRAME_THROTTLE = 0.033  
COUNT_ONCE_PER_SESSION = True
FRAME_PROCESS_INTERVAL = max(1, int(os.getenv("FRAME_PROCESS_INTERVAL", "1")))
ALLOW_STREAM_RESTART = os.getenv("ALLOW_STREAM_RESTART", "true").lower() == "true"
RESTART_BACKOFF_SECONDS = float(os.getenv("RESTART_BACKOFF_SECONDS", "5.0"))
REGISTRY_MAX_AGE_SEC = float(os.getenv("REGISTRY_MAX_AGE_SEC", "600"))
REGISTRY_MAX_SIZE = int(os.getenv("REGISTRY_MAX_SIZE", "5000"))

DETECTION_CONFIDENCE_THRESHOLD = 0.6
MIN_PERSON_WIDTH = 30
MIN_PERSON_HEIGHT = 50
MAX_PERSON_WIDTH = 800
MAX_PERSON_HEIGHT = 2000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 0.8

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "person_analytics")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
hourly_col = db["hourly_stats"]
daily_col = db["daily_stats"]

# ==================== INITIALIZATION ====================
app = Flask(__name__)
core = Core()

try:
    det_model = core.read_model(DETECTION_MODEL)
    compiled_det = core.compile_model(det_model, "CPU")
    outputs = compiled_det.outputs
    print(f"[Model] Detection model loaded: {DETECTION_MODEL}")
    print(f"[Model] Outputs: {[str(o) for o in outputs]}")
    det_out = compiled_det.output(0)
    print(f"[Model] Using output: {det_out}")
except Exception as e:
    print(f"[ERROR] Failed to load detection model: {e}")
    raise

reid_model = core.read_model(REID_MODEL)
compiled_reid = core.compile_model(reid_model, "CPU")
reid_out = compiled_reid.output(0)

camera_streams = []
stop_event = threading.Event()

registry = []
registry_lock = threading.Lock()
next_global_id = 0

entries = {}
entries_lock = threading.Lock()
recent_embeddings = []
recent_embeddings_lock = threading.Lock()
EMBEDDING_MEMORY_TIME = 86400.0

metrics = {
    "total_people": 0,
    "individual_people": 0,
    "group_people": 0,
    "group_events": 0,
    "hourly": {},
    "day_snapshot": None,
    "current_date": None
}
metrics_lock = threading.Lock()


# ==================== UTILITY FUNCTIONS ====================
def l2_norm(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def is_live_source(src):
    """Heuristic to detect live streams (camera indices, RTSP/HTTP)."""
    if isinstance(src, int):
        return True
    if isinstance(src, str):
        stripped = src.strip().lower()
        if stripped.isdigit():
            return True
        if stripped.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            return True
    return False

def cosine(a, b):
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def update_registry(embedding):
    """
    Match person embedding to existing ID or create new ID.
    Returns: (person_id, is_new_person)
    """
    global next_global_id
    with registry_lock:
        best_id, best_sim = None, -1.0
        current_time = time.time()
        
        for entry in registry:
            sim = cosine(entry["emb"], embedding)
            if sim > best_sim:
                best_sim, best_id = sim, entry["id"]
        
        if best_sim >= REID_SIM_THRESHOLD:
            for entry in registry:
                if entry["id"] == best_id:
                    entry["emb"] = l2_norm(REID_UPDATE_MOMENTUM * entry["emb"] +
                                           (1 - REID_UPDATE_MOMENTUM) * embedding)
                    entry["last_seen"] = current_time
                    return best_id, False
        
        new_id = next_global_id
        next_global_id += 1
        registry.append({"id": new_id, "emb": embedding.copy(), "last_seen": current_time})
        return new_id, True

def prune_registry(current_time=None):
    """Remove stale or excess registry entries to control memory/CPU."""
    if current_time is None:
        current_time = time.time()
    with registry_lock:
        original_len = len(registry)
        registry[:] = [entry for entry in registry if current_time - entry["last_seen"] <= REGISTRY_MAX_AGE_SEC]
        if len(registry) > REGISTRY_MAX_SIZE:
            registry.sort(key=lambda e: e["last_seen"], reverse=True)
            registry[:] = registry[:REGISTRY_MAX_SIZE]
        if len(registry) != original_len:
            print(f"[Registry] Pruned {original_len - len(registry)} stale entries; active={len(registry)}")


# ==================== DETECTION & RE-IDENTIFICATION ====================
def detect_and_reid(frame):
    """
    Detect people in video frame and extract their re-identification features.
    Returns: (bounding_boxes, centroids, person_ids, is_new_flags, embeddings)
    """
    h, w = frame.shape[:2]
    
    if "0202" in DETECTION_MODEL:
        model_w, model_h = 512, 512
        img = cv2.resize(frame, (model_w, model_h))
        inp = img.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, :]
    else:
        model_w, model_h = 544, 320
        img = cv2.resize(frame, (model_w, model_h))
        inp = img.astype(np.float32).transpose(2, 0, 1)[None, :]

    try:
        result = compiled_det([inp])
        
        if isinstance(result, dict):
            dets = list(result.values())[0]
        elif isinstance(result, list):
            dets = result[0]
        else:
            dets = result[det_out]
        
        original_shape = dets.shape
        if len(dets.shape) == 4:
            dets = dets.reshape(-1, 7)
        elif len(dets.shape) == 3:
            dets = dets.reshape(-1, dets.shape[-1])
        elif len(dets.shape) == 2:
            dets = dets
        
        if not hasattr(detect_and_reid, '_debug_printed'):
            print(f"[DEBUG] Model: {DETECTION_MODEL}")
            print(f"[DEBUG] Input shape: {inp.shape}, dtype: {inp.dtype}, range: [{inp.min():.3f}, {inp.max():.3f}]")
            print(f"[DEBUG] Model inference successful")
            print(f"[DEBUG] Original output shape: {original_shape}")
            print(f"[DEBUG] Reshaped output shape: {dets.shape}")
            print(f"[DEBUG] Total detections: {len(dets)}")
            
            if len(dets) > 0:
                print(f"[DEBUG] All detections with conf > 0.01:")
                shown = 0
                for i, d in enumerate(dets):
                    if len(d) >= 7:
                        conf = float(d[2])
                        label = int(d[1])
                        if conf > 0.01:
                            print(f"  [{i}] image_id={int(d[0])}, label={label}, conf={conf:.4f}, "
                                  f"bbox=({float(d[3]):.1f},{float(d[4]):.1f},{float(d[5]):.1f},{float(d[6]):.1f})")
                            shown += 1
                            if shown >= 20:
                                break
                
                label_counts = {}
                for d in dets:
                    if len(d) >= 7:
                        label = int(d[1])
                        label_counts[label] = label_counts.get(label, 0) + 1
                print(f"[DEBUG] Detections by label: {label_counts}")
                
                valid_count = sum(1 for d in dets if len(d) >= 7 and float(d[2]) > DETECTION_CONFIDENCE_THRESHOLD and int(d[1]) == 0)
                print(f"[DEBUG] Valid person detections (label=0, conf>={DETECTION_CONFIDENCE_THRESHOLD}): {valid_count}")
            else:
                print(f"[DEBUG] No detections returned from model!")
            detect_and_reid._debug_printed = True
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], [], []
    
    boxes, scores = [], []
    debug_stats = {"total": 0, "invalid_len": 0, "invalid_id": 0, "low_conf": 0, "wrong_label": 0, "invalid_coords": 0, "size_filtered": 0, "passed": 0}

    for d in dets:
        debug_stats["total"] += 1
        if len(d) < 7:
            debug_stats["invalid_len"] += 1
            continue
            
        image_id = int(d[0])
        label = int(d[1])
        conf = float(d[2])
        
        if image_id < 0 or conf <= 0:
            debug_stats["invalid_id"] += 1
            continue
        
        if conf < DETECTION_CONFIDENCE_THRESHOLD:
            debug_stats["low_conf"] += 1
            continue
        
        if "0202" in DETECTION_MODEL:
            if label != 0:
                debug_stats["wrong_label"] += 1
                continue
        else:
            if label != 1:
                debug_stats["wrong_label"] += 1
                continue
        
        xmin, ymin, xmax, ymax = float(d[3]), float(d[4]), float(d[5]), float(d[6])
        
        if "0202" in DETECTION_MODEL:
            xmin_scaled = xmin * model_w
            ymin_scaled = ymin * model_h
            xmax_scaled = xmax * model_w
            ymax_scaled = ymax * model_h
            
            if xmin_scaled >= xmax_scaled or ymin_scaled >= ymax_scaled:
                debug_stats["invalid_coords"] += 1
                continue
            
            xmin_scaled = max(0, min(xmin_scaled, model_w))
            ymin_scaled = max(0, min(ymin_scaled, model_h))
            xmax_scaled = max(xmin_scaled + 1, min(xmax_scaled, model_w))
            ymax_scaled = max(ymin_scaled + 1, min(ymax_scaled, model_h))
            
            scale_x, scale_y = w / model_w, h / model_h
            x1 = max(0, int(xmin_scaled * scale_x))
            y1 = max(0, int(ymin_scaled * scale_y))
            x2 = min(w, int(xmax_scaled * scale_x))
            y2 = min(h, int(ymax_scaled * scale_y))
        else:
            if xmax <= 1.0 and ymax <= 1.0:
                xmin, ymin, xmax, ymax = xmin*model_w, ymin*model_h, xmax*model_w, ymax*model_h
            scale_x, scale_y = w / model_w, h / model_h
            x1 = max(0, int(xmin * scale_x))
            y1 = max(0, int(ymin * scale_y))
            x2 = min(w, int(xmax * scale_x))
            y2 = min(h, int(ymax * scale_y))
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        if box_width < MIN_PERSON_WIDTH or box_height < MIN_PERSON_HEIGHT:
            debug_stats["size_filtered"] += 1
            continue
        
        if box_width > MAX_PERSON_WIDTH or box_height > MAX_PERSON_HEIGHT:
            debug_stats["size_filtered"] += 1
            continue
        
        aspect_ratio = box_width / box_height if box_height > 0 else 0
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            debug_stats["size_filtered"] += 1
            continue
        
        debug_stats["passed"] += 1
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
    
    if not hasattr(detect_and_reid, '_filter_debug_printed'):
        print(f"[DEBUG] Filtering stats: {debug_stats}")
        detect_and_reid._filter_debug_printed = True

    if boxes:
        rects = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
        idxs = cv2.dnn.NMSBoxes(rects, scores, 0.3, 0.3)
        if len(idxs) > 0:
            idxs = [i[0] if isinstance(i, (list, np.ndarray)) else i for i in idxs]
            boxes = [boxes[i] for i in idxs]
            scores = [scores[i] for i in idxs]
        
        if len(boxes) > 1:
            filtered_boxes = []
            filtered_scores = []
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                box_area = (x2 - x1) * (y2 - y1)
                
                should_keep = True
                indices_to_remove = []
                for j, (other_box, other_score) in enumerate(zip(filtered_boxes, filtered_scores)):
                    ox1, oy1, ox2, oy2 = other_box
                    other_centroid = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
                    other_area = (ox2 - ox1) * (oy2 - oy1)
                    
                    dist = math.sqrt((centroid[0] - other_centroid[0])**2 + (centroid[1] - other_centroid[1])**2)
                    
                    inter_x1 = max(x1, ox1)
                    inter_y1 = max(y1, oy1)
                    inter_x2 = min(x2, ox2)
                    inter_y2 = min(y2, oy2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    
                    min_dim = min(box_area**0.5, other_area**0.5)
                    if dist < min_dim * 0.5 or inter_area > min(box_area, other_area) * 0.5:
                        if score <= other_score:
                            should_keep = False
                            break
                        else:
                            indices_to_remove.append(j)
                
                for j in reversed(indices_to_remove):
                    filtered_boxes.pop(j)
                    filtered_scores.pop(j)
                
                if should_keep:
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
            
            boxes = filtered_boxes
            scores = filtered_scores

    centroids, ids, new_flags, embeddings = [], [], [], []
    for (x1, y1, x2, y2) in boxes:
        centroids.append(((x1 + x2)//2, (y1 + y2)//2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            ids.append(-1); new_flags.append(False); embeddings.append(None)
            continue
        
        reid_in = cv2.resize(crop, (128, 256)).astype(np.float32).transpose(2, 0, 1)[None, :]
        feat = l2_norm(compiled_reid([reid_in])[reid_out].flatten())
        pid, is_new = update_registry(feat)
        ids.append(pid); new_flags.append(is_new); embeddings.append(feat)
    
    return boxes, centroids, ids, new_flags, embeddings


# ==================== GROUP DETECTION ====================
def cluster_indices(centroids, threshold):
    """
    Group person centroids that are close together.
    Returns list of clusters, where each cluster is a list of indices.
    """
    n = len(centroids)
    clusters, used = [], [False]*n
    for i in range(n):
        if used[i]: continue
        stack = [i]; used[i] = True
        for j in range(i+1, n):
            if used[j]: continue
            if np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j])) < threshold:
                stack.append(j); used[j] = True
        clusters.append(stack)
    return clusters

def check_recent_embedding(embedding, current_time):
    """
    Check if this person was already counted today by comparing embedding similarity.
    Prevents counting the same person multiple times even if REID assigns different IDs.
    """
    with recent_embeddings_lock:
        recent_embeddings[:] = [(emb, ts) for emb, ts in recent_embeddings 
                                if current_time - ts < EMBEDDING_MEMORY_TIME]
        
        for recent_emb, _ in recent_embeddings:
            sim = cosine(embedding, recent_emb)
            if sim >= REID_SIM_THRESHOLD:
                return True
    return False

def process_new_entries(per_cam_results):
    """
    Process detected people across all cameras and count them as individuals or groups.
    Each person is counted only once per day using multiple deduplication layers.
    """
    current_time = time.time()
    newly_seen_ids, index_map = {}, {}
    
    for cam_idx, (boxes, centroids, ids, new_flags, embeddings) in enumerate(per_cam_results):
        for i, (pid, embedding) in enumerate(zip(ids, embeddings)):
            if pid < 0 or embedding is None: continue
            
            with entries_lock:
                already_registered = pid in entries
            
            if already_registered:
                continue
            
            embedding_already_counted = check_recent_embedding(embedding, current_time)
            
            if embedding_already_counted:
                continue
            
            index_map.setdefault(cam_idx, []).append((i, pid, embedding))
            newly_seen_ids[pid] = cam_idx
    
    if not newly_seen_ids: return

    for cam_idx, (boxes, centroids, ids, new_flags, embeddings) in enumerate(per_cam_results):
        if cam_idx not in index_map: continue
        
        clusters = cluster_indices(centroids, GROUP_DISTANCE_THRESHOLD)
        cent_to_cluster = {idx: ci for ci, cluster in enumerate(clusters) for idx in cluster}
        
        cluster_new = collections.defaultdict(list)
        for (i, pid, embedding) in index_map[cam_idx]:
            cidx = cent_to_cluster.get(i, None)
            cluster_new[cidx].append((i, pid, embedding))
        
        for cidx, items in cluster_new.items():
            if len(items) > 1:
                pids = [pid for _, pid, _ in items]
                group_embeddings = [emb for _, _, emb in items]
                register_group(pids, cam_idx, group_embeddings)
            else:
                _, pid, embedding = items[0]
                with entries_lock:
                    if pid in entries and entries[pid].get("is_group_member", False):
                        continue
                register_entry(pid, cam_idx, is_group_member=False, embedding=embedding)

    prune_registry(current_time)

def register_entry(pid, cam_idx, is_group_member=False, embedding=None):
    """
    Register a single person and update metrics.
    Once registered, this person is never counted again in the same day.
    """
    ts = time.time()
    
    with entries_lock:
        if pid in entries:
            existing_entry = entries[pid]
            if existing_entry.get("is_group_member", False):
                return
            return
    
    if is_group_member:
        with entries_lock:
            entries[pid] = {
                "first_seen": ts, 
                "first_cam": cam_idx, 
                "is_group_member": True,
                "counted": True
            }
        with metrics_lock:
            metrics["group_people"] += 1
            metrics["total_people"] = metrics["individual_people"] + metrics["group_people"]
    else:
        with entries_lock:
            entries[pid] = {
                "first_seen": ts, 
                "first_cam": cam_idx, 
                "is_group_member": False,
                "counted": True
            }
        
        if embedding is not None:
            with recent_embeddings_lock:
                recent_embeddings.append((embedding.copy(), ts))
        
        with metrics_lock:
            metrics["individual_people"] += 1
            metrics["total_people"] = metrics["individual_people"] + metrics["group_people"]

def register_group(pids, cam_idx, embeddings=None):
    """
    Register a group of people detected together.
    Each person in the group is counted as group_people (not individual_people).
    group_events counts how many times groups entered (2+ people together).
    """
    ts = time.time()
    new_pids = []
    new_embeddings = []
    converted_from_individual = 0
    
    if embeddings:
        filtered_pids = []
        filtered_embeddings = []
        for pid, emb in zip(pids, embeddings):
            if emb is None:
                continue
            
            if check_recent_embedding(emb, ts):
                continue
            
            with entries_lock:
                if pid in entries:
                    existing_entry = entries[pid]
                    if not existing_entry.get("is_group_member", False):
                        with metrics_lock:
                            metrics["individual_people"] -= 1
                            metrics["group_people"] += 1
                        existing_entry["is_group_member"] = True
                        converted_from_individual += 1
                    continue
            
            filtered_pids.append(pid)
            filtered_embeddings.append(emb)
        pids = filtered_pids
        embeddings = filtered_embeddings
    
    with entries_lock:
        for pid in pids:
            if pid not in entries:
                entries[pid] = {
                    "first_seen": ts, 
                    "first_cam": cam_idx, 
                    "is_group_member": True,
                    "counted": True
                }
                new_pids.append(pid)
    
    total_group_members = len(new_pids) + converted_from_individual
    
    if total_group_members >= 2:
        if embeddings:
            with recent_embeddings_lock:
                for emb in embeddings:
                    if emb is not None:
                        recent_embeddings.append((emb.copy(), ts))
        
        with metrics_lock:
            metrics["group_events"] += 1
            metrics["group_people"] += len(new_pids)
            metrics["total_people"] = metrics["individual_people"] + metrics["group_people"]
    elif len(new_pids) > 0:
        if embeddings:
            with recent_embeddings_lock:
                for emb in embeddings:
                    if emb is not None:
                        recent_embeddings.append((emb.copy(), ts))
        
        with metrics_lock:
            metrics["group_people"] += len(new_pids)
            metrics["total_people"] = metrics["individual_people"] + metrics["group_people"]


# ==================== DATA PERSISTENCE ====================
def hourly_snapshot_worker():
    """
    Save metrics to MongoDB at the end of each hour (at :00 minutes).
    Also resets hourly data when a new day starts.
    """
    with metrics_lock:
        if metrics["current_date"] is None:
            metrics["current_date"] = time.strftime("%Y-%m-%d")
    
    while not stop_event.is_set():
        now = time.localtime()
        next_hour = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour + 1, 0, 0, now.tm_wday, now.tm_yday, now.tm_isdst))
        wait = max(1.0, next_hour - time.time())
        time.sleep(wait)
        
        current_time = time.localtime()
        hour_just_ended = (current_time.tm_hour - 1) % 24
        
        with metrics_lock:
            current_date = time.strftime("%Y-%m-%d")
            if metrics["current_date"] != current_date:
                metrics["hourly"] = {}
                metrics["current_date"] = current_date
                print(f"[System] New day detected: {current_date}, resetting hourly data")
            
            snapshot = {
                "timestamp": int(time.time()),
                "date": current_date,
                "hour": hour_just_ended,
                "total_people": metrics["total_people"],
                "individual_people": metrics["individual_people"],
                "group_people": metrics["group_people"],
                "group_events": metrics["group_events"]
            }
            
            metrics["hourly"][hour_just_ended] = {
                "total_people": metrics["total_people"],
                "individual_people": metrics["individual_people"],
                "group_people": metrics["group_people"],
                "group_events": metrics["group_events"],
                "timestamp": snapshot["timestamp"]
            }
            
            hourly_col.insert_one(snapshot)
            print(f"[Mongo] Hourly snapshot pushed for hour {hour_just_ended} ({hour_just_ended}:00) - Total: {metrics['total_people']}")

def end_of_day_snapshot_worker():
    """
    Save daily summary to MongoDB at midnight and reset all metrics for new day.
    """
    while not stop_event.is_set():
        now = time.localtime()
        midnight = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 23, 59, 59, now.tm_wday, now.tm_yday, now.tm_isdst))
        wait = max(1.0, midnight - time.time() + 1)
        time.sleep(wait)
        
        with metrics_lock:
            ending_date = time.strftime("%Y-%m-%d")
            metrics["day_snapshot"] = {
                "timestamp": int(time.time()),
                "date": ending_date,
                "total_people": metrics["total_people"],
                "individual_people": metrics["individual_people"],
                "group_people": metrics["group_people"],
                "group_events": metrics["group_events"],
                "hourly_data": metrics["hourly"].copy()
            }
            daily_col.insert_one(metrics["day_snapshot"])
            print(f"[Mongo] Daily snapshot pushed for {ending_date}")
        
        time.sleep(2)
        
        with metrics_lock:
            metrics["total_people"] = 0
            metrics["individual_people"] = 0
            metrics["group_people"] = 0
            metrics["group_events"] = 0
            metrics["hourly"] = {}
            metrics["day_snapshot"] = None
            metrics["current_date"] = time.strftime("%Y-%m-%d")
        
        with entries_lock:
            entries.clear()
        with recent_embeddings_lock:
            recent_embeddings.clear()
        
        with registry_lock:
            current_time = time.time()
            registry[:] = [r for r in registry if current_time - r["last_seen"] < 86400]
        
        print(f"[System] Metrics reset for new day: {metrics['current_date']}")


# ==================== VIDEO STREAMING ====================
def camera_reader(stream):
    """Read frames from source and optionally restart when live feeds drop."""
    src = stream["src"]
    q = stream["queue"]
    live_source = stream["is_live"]
    backoff = max(1.0, RESTART_BACKOFF_SECONDS)
    
    while not stop_event.is_set():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[Camera Reader] Cannot open: {src}")
            if live_source and ALLOW_STREAM_RESTART:
                time.sleep(backoff)
                continue
            stream["finished"] = True
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_delay = 1.0 / fps
        print(f"[Camera Reader] Source ready: {src} | FPS: {fps:.2f}")
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[Camera Reader] Source ended or failed: {src}")
                break
            frame = cv2.resize(frame, (640, 360))
            with q.mutex:
                q.queue.clear()
            q.put(frame)
            stream["last_frame_ts"] = time.time()
            time.sleep(frame_delay)
        
        cap.release()
        with q.mutex:
            q.queue.clear()
        
        if live_source and ALLOW_STREAM_RESTART:
            print(f"[Camera Reader] Restarting live source: {src} in {backoff:.1f}s")
            time.sleep(backoff)
            continue
        
        stream["finished"] = True
        print(f"[Camera Reader] Finalized source: {src}")
        return

def create_camera_stream(src):
    stream = {
        "src": src,
        "queue": queue.Queue(maxsize=1),
        "thread": None,
        "finished": False,
        "is_live": is_live_source(src),
        "frame_count": 0,
        "last_result": None,
        "last_frame_ts": 0.0,
    }
    t = threading.Thread(target=camera_reader, args=(stream,), daemon=True)
    stream["thread"] = t
    t.start()
    return stream

def compose_frame(frames, per_cam_results, fps):
    """
    Combine multiple camera frames into one display frame with bounding boxes and labels.
    """
    heights = [f.shape[0] for f in frames]
    widths = [f.shape[1] for f in frames]
    max_h, total_w = max(heights), sum(widths)
    out = np.full((max_h + 40, total_w, 3), 0, dtype=np.uint8)
    x_off, all_visible_ids, total_groups = 0, [], 0
    for idx, f in enumerate(frames):
        h, w = f.shape[:2]
        out[40:40+h, x_off:x_off+w] = f
        boxes, centroids, ids, _, _ = per_cam_results[idx]
        valid_ids = [pid for pid in ids if pid >= 0]
        all_visible_ids.extend(valid_ids)
        clusters = cluster_indices(centroids, GROUP_DISTANCE_THRESHOLD)
        total_groups += sum(1 for c in clusters if len(c) > 1)
        cv2.putText(out, f"Cam {idx} - People: {len(valid_ids)} | Groups: {total_groups}",
                    (x_off + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        for (box, pid) in zip(boxes, ids):
            if pid < 0: continue
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x_off+x1, 40+y1), (x_off+x2, 40+y2), (0,255,0), 2)
            cv2.putText(out, f"ID:{pid}", (x_off+x1, 40+y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        x_off += w
    cv2.putText(out, f"Total: {len(set(all_visible_ids))} | FPS: {fps:.1f}",
                (10, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return out

def gen_multi_stream():
    """
    Main video processing loop: reads frames from all cameras, detects people,
    tracks them across cameras, and streams the result to web interface.
    """
    global camera_streams
    camera_streams.clear()
    for src in CAM_SOURCES:
        camera_streams.append(create_camera_stream(src))
    threading.Thread(target=hourly_snapshot_worker, daemon=True).start()
    threading.Thread(target=end_of_day_snapshot_worker, daemon=True).start()
    prev_time, fps = time.time(), 0.0
    try:
        while True:
            camera_streams = [s for s in camera_streams if not s["finished"]]
            active_streams = camera_streams
            if not active_streams:
                frames = [np.zeros((360, 640, 3), dtype=np.uint8)]
                per_cam_results = [([], [], [], [], [])]
            else:
                frames, per_cam_results = [], []
                for stream in active_streams:
                    q = stream["queue"]
                    try:
                        frame = q.get(timeout=1.0)
                    except queue.Empty:
                        frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    frames.append(frame)
                for stream, frame in zip(active_streams, frames):
                    stream["frame_count"] += 1
                    should_process = (
                        stream["last_result"] is None or
                        FRAME_PROCESS_INTERVAL == 1 or
                        stream["frame_count"] % FRAME_PROCESS_INTERVAL == 0
                    )
                    if should_process:
                        stream["last_result"] = detect_and_reid(frame)
                    per_cam_results.append(stream["last_result"] or ([], [], [], [], []))
                process_new_entries(per_cam_results)
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (curr_time - prev_time)))
            prev_time = curr_time
            out = compose_frame(frames, per_cam_results, fps)
            _, jpg = cv2.imencode(".jpg", out)
            time.sleep(FRAME_THROTTLE)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
    finally:
        stop_event.set()
        time.sleep(0.5)


# ==================== WEB INTERFACE ====================
@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
      <title>Multi-Camera ReID Stream</title>
      <style>
        body { background:#101010; color:white; font-family:Arial, sans-serif; text-align:center; padding:20px; }
        h2 { margin-top:20px; }
        .container { display:flex; flex-direction:column; align-items:center; gap:20px; }
        .metrics {
          background:#1a1a1a;
          display:inline-block;
          padding:15px 30px;
          border-radius:12px;
          margin-top:15px;
          text-align:left;
        }
        .metric { display:flex; justify-content:space-between; margin:4px 0; }
        .label { color:#bbb; }
        .value { font-weight:bold; color:#00ffcc; }
        .widget {
          background:#1a1a1a;
          padding:20px;
          border-radius:12px;
          width:95%;
          max-width:1200px;
          margin-top:20px;
        }
        .widget h3 { margin-top:0; color:#00ffcc; }
        .hourly-chart {
          display:flex;
          align-items:flex-end;
          justify-content:space-around;
          height:200px;
          margin-top:20px;
          border-bottom:2px solid #333;
          padding-bottom:10px;
        }
        .hour-bar {
          display:flex;
          flex-direction:column;
          align-items:center;
          width:40px;
          margin:0 2px;
        }
        .bar {
          width:100%;
          background:linear-gradient(to top, #00ffcc, #0099cc);
          border-radius:4px 4px 0 0;
          min-height:4px;
          margin-bottom:5px;
          transition:height 0.3s;
        }
        .bar-label {
          font-size:10px;
          color:#888;
          margin-top:5px;
        }
        .bar-value {
          font-size:11px;
          color:#00ffcc;
          font-weight:bold;
          margin-top:2px;
        }
        .validation {
          margin-top:15px;
          padding:10px;
          border-radius:8px;
          background:#2a2a2a;
        }
        .validation.ok { border-left:4px solid #00ff00; }
        .validation.error { border-left:4px solid #ff0000; }
      </style>
    </head>
    <body>
      <div class="container">
        <h2>Multi-Camera ReID Stream</h2>
        <img src="/video_feed" width="95%" style="max-width:1400px;">
        <div class="metrics">
          <div class="metric"><div class="label">Total People</div><div class="value" id="total">0</div></div>
          <div class="metric"><div class="label">Individuals</div><div class="value" id="ind">0</div></div>
          <div class="metric"><div class="label">Group People</div><div class="value" id="grp">0</div></div>
          <div class="metric"><div class="label">Group Events</div><div class="value" id="gpe">0</div></div>
        </div>
        <div class="widget">
          <h3>Daily Hourly Statistics</h3>
          <div class="hourly-chart" id="hourlyChart"></div>
          <div class="validation" id="validation">
            <strong>Data Validation:</strong> <span id="validationText">Checking...</span>
          </div>
        </div>
      </div>
      <script>
        async function refreshMetrics() {
          try {
            let res = await fetch('/metrics');
            let data = await res.json();
            document.getElementById('total').innerText = data.total_people;
            document.getElementById('ind').innerText = data.individual_people;
            document.getElementById('grp').innerText = data.group_people;
            document.getElementById('gpe').innerText = data.group_events;
            
            let sum = data.individual_people + data.group_people;
            let isValid = sum <= data.total_people;
            let validationEl = document.getElementById('validation');
            let validationText = document.getElementById('validationText');
            if (isValid) {
              validationEl.className = 'validation ok';
              validationText.innerHTML = `✓ Valid: Individuals (${data.individual_people}) + Group People (${data.group_people}) = ${sum} ≤ Total (${data.total_people})`;
            } else {
              validationEl.className = 'validation error';
              validationText.innerHTML = `✗ Invalid: Individuals (${data.individual_people}) + Group People (${data.group_people}) = ${sum} > Total (${data.total_people})`;
            }
          } catch(e) {
            console.error("Metrics fetch failed", e);
          }
        }
        
        async function refreshHourlyData() {
          try {
            let res = await fetch('/hourly_data');
            let data = await res.json();
            let chart = document.getElementById('hourlyChart');
            chart.innerHTML = '';
            
            if (!data.hourly || data.hourly.length === 0) {
              chart.innerHTML = '<div style="color:#888;">No hourly data yet. Data will appear after the first hour completes.</div>';
              return;
            }
            
            let maxVal = Math.max(...data.hourly.map(h => h.total_people || 0), 1);
            if (maxVal === 0) maxVal = 1;
            
            data.hourly.forEach((hourData) => {
              let hour = hourData.hour;
              let hasData = hourData.timestamp !== null;
              let currentHour = data.current_hour;
              let isCurrentHour = hour === currentHour;
              
              let barDiv = document.createElement('div');
              barDiv.className = 'hour-bar';
              if (isCurrentHour) {
                barDiv.style.border = '2px solid #00ffcc';
                barDiv.style.borderRadius = '4px';
                barDiv.style.padding = '2px';
              }
              
              let bar = document.createElement('div');
              bar.className = 'bar';
              let height = ((hourData.total_people || 0) / maxVal) * 180;
              bar.style.height = height + 'px';
              if (!hasData) {
                bar.style.opacity = '0.3';
                bar.style.background = 'linear-gradient(to top, #666, #444)';
              }
              bar.title = `Hour ${hour}:00 - Total=${hourData.total_people}, Ind=${hourData.individual_people}, Groups=${hourData.group_people}`;
              
              let label = document.createElement('div');
              label.className = 'bar-label';
              label.textContent = hour + 'h';
              if (isCurrentHour) {
                label.style.color = '#00ffcc';
                label.style.fontWeight = 'bold';
              }
              
              let value = document.createElement('div');
              value.className = 'bar-value';
              value.textContent = hourData.total_people || (hasData ? '0' : '-');
              if (!hasData) {
                value.style.color = '#666';
              }
              
              barDiv.appendChild(bar);
              barDiv.appendChild(value);
              barDiv.appendChild(label);
              chart.appendChild(barDiv);
            });
          } catch(e) {
            console.error("Hourly data fetch failed", e);
          }
        }
        
        setInterval(refreshMetrics, 3000);
        setInterval(refreshHourlyData, 10000);
        refreshMetrics();
        refreshHourlyData();
      </script>
    </body>
    </html>
    """)


@app.route('/video_feed')
def video_feed():
    return Response(gen_multi_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics_json():
    with metrics_lock:
        return jsonify(metrics)

@app.route('/hourly_data')
def hourly_data():
    """Return hourly statistics for the current day (hours 0-23)."""
    with metrics_lock:
        hourly_array = []
        for hour in range(24):
            if hour in metrics["hourly"]:
                hourly_array.append({
                    "hour": hour,
                    "total_people": metrics["hourly"][hour]["total_people"],
                    "individual_people": metrics["hourly"][hour]["individual_people"],
                    "group_people": metrics["hourly"][hour]["group_people"],
                    "group_events": metrics["hourly"][hour]["group_events"],
                    "timestamp": metrics["hourly"][hour]["timestamp"]
                })
            else:
                hourly_array.append({
                    "hour": hour,
                    "total_people": 0,
                    "individual_people": 0,
                    "group_people": 0,
                    "group_events": 0,
                    "timestamp": None
                })
        return jsonify({
            "hourly": hourly_array,
            "current_date": metrics.get("current_date", time.strftime("%Y-%m-%d")),
            "current_hour": time.localtime().tm_hour
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
