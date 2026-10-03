# AI Product Detection & Counting Web Application

A full-stack product detection system built with FastAPI, OpenCV, Ultralytics YOLOv8, ByteTrack-inspired tracking, MySQL, and a responsive web frontend.

## Features

- Upload video for AI detection and counting
- Open webcam for real-time detection
- YOLOv8 detection pipeline with configurable device
- Tracking-based unique counting logic
- MySQL persistence for sessions and detection results
- Statistics dashboard with class counts and activity overview
- Export-friendly result schema for CSV/Pandas processing

## Project structure

```text
project/
├── app/
│   ├── api/
│   │   ├── routes_video.py
│   │   ├── routes_camera.py
│   │   ├── routes_detection.py
│   │   └── routes_statistics.py
│   ├── database/
│   │   ├── crud.py
│   │   └── mysql.py
│   ├── models/
│   │   ├── detection.py
│   │   ├── product.py
│   │   └── tracking.py
│   ├── schemas/
│   │   ├── detection_schema.py
│   │   └── product_schema.py
│   ├── services/
│   │   ├── camera_service.py
│   │   ├── counting_service.py
│   │   ├── tracking_service.py
│   │   ├── video_service.py
│   │   └── yolo_service.py
│   ├── utils/
│   │   ├── drawing.py
│   │   ├── helpers.py
│   │   └── video_utils.py
│   ├── config.py
│   ├── main.py
│   └── __init__.py
├── frontend/
│   ├── camera.html
│   ├── css/d
│   │   └── style.css
│   ├── history.html
│   ├── index.html
│   ├── js/
│   │   ├── camera.js
│   │   ├── dashboard.js
│   │   └── upload.js
│   ├── statistics.html
│   └── upload.html
├── logs/
├── models/
│   └── best.pt
├── outputs/
├── uploads/
├── .env.example
├── .env
├── requirements.txt
├── run.py
├── README.md
└── schema.sql
```

## Setup

### 1. Clone the repo

```bash
git clone <your-repository-url>
cd project
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and configure MySQL

Create a database named `product_detection` and a user with access.

Example SQL:

```sql
CREATE DATABASE product_detection;
```

### 5. Configure environment

Copy `.env.example` to `.env` and update values if needed.

```bash
copy .env.example .env
```

Example:

```env
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/product_detection
MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.5
DEVICE=0
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
MAX_UPLOAD_SIZE=52428800
```

### 6. Add the YOLO model

Place your custom YOLOv8 model at:

```text
models/best.pt
```

The system reads the model from the configured `MODEL_PATH` and uses class names from the model itself.

### 7. Initialize database schema

```bash
python -c "from app.database.mysql import init_db; init_db(); print('db ready')"
```

### 8. Run the app

```bash
python run.py
```

Then open:

- http://localhost:8000/
- http://localhost:8000/upload
- http://localhost:8000/camera
- http://localhost:8000/statistics

## Usage

### Upload video

1. Open the upload page.
2. Choose a supported video file (`.mp4`, `.avi`, `.mov`, `.mkv`).
3. Click "Start Detection".
4. Wait for processing to finish.
5. View the processed output and summary metrics.

### Open camera

1. Open the camera page.
2. Click "Start Camera".
3. Allow webcam access.
4. The system streams live detections and tracking metadata.
5. Stop the camera when finished.

## API endpoints

- `GET /api/health`
- `POST /api/video/upload`
- `POST /api/video/process`
- `GET /api/video/result/{id}`
- `POST /api/camera/start`
- `POST /api/camera/stop`
- `GET /api/sessions`
- `GET /api/sessions/{id}`
- `GET /api/statistics`
- `GET /api/products`

## MySQL schema

The ORM models include:

- `products`
- `detection_sessions`
- `detection_results`
- `product_counts`

Database creation is handled by SQLAlchemy metadata when the app starts.

## Notes

- If CUDA is available, the app chooses GPU mode when `DEVICE` is set to a valid GPU index.
- If no GPU is available, it falls back to CPU automatically.
- All logs are written to `logs/app.log`.

## Acceptance summary

The solution is designed to satisfy the required AI workflow:

Detect → Classify → Track → Count → Display → Save Database → Statistics
