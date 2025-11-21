# AgriShield ML Flask API

## 🎯 Simple Setup

### Local Development (Your Computer):
```bash
python app.py
```
✅ Runs on port 5001  
✅ Flask development server  
✅ Good for testing

### Production Webserver:
```bash
gunicorn -c gunicorn_config.py wsgi:application
```
✅ Production-ready  
✅ Handles multiple requests  
✅ Auto-restarts on crash

---

## 📁 Files

- **`app.py`** - Main Flask application (pest detection + forecasting)
- **`wsgi.py`** - WSGI entry point for Gunicorn
- **`gunicorn_config.py`** - Gunicorn production configuration
- **`requirements.txt`** - Python dependencies

---

## 🔧 Configuration

### Port
- Default: **5001**
- Matches PHP backend calls: `http://localhost:5001/detect`

### Model Path
1. Checks database via PHP endpoint
2. Falls back to `datasets/best 2.pt`
3. Falls back to `pest_detection_ml/models/best.pt`

### Classes
- **Dynamically loaded** from model
- No hardcoding - reads from `model.names`

---

## 📡 Endpoints

### Detection:
- `GET /health` - Health check
- `GET /status` - Status check  
- `POST /detect` - Detect pests (multipart form-data)
- `POST /classify` - Classify pests (Android app)

### Forecasting:
- `GET /forecast` - Generate 7-day pest forecast
- `POST /forecast` - Generate forecast with custom days
- `GET /forecast/quick` - Quick single forecast
- `GET /forecast/current` - Get forecast from database
- `POST /forecast/update` - Manually update forecast

---

## ✅ Features

- ✅ Dynamic class loading from model
- ✅ Database model path lookup
- ✅ Fallback to "best 2.pt"
- ✅ Pest detection with YOLO
- ✅ Pest forecasting
- ✅ Production-ready with Gunicorn
- ✅ Auto-restart on crash

---

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run locally:**
   ```bash
   python app.py
   ```

3. **Test:**
   ```bash
   curl http://localhost:5001/health
   ```

---

## 📝 Notes

- Flask framework is used for all endpoints
- Local = Flask dev server (`app.run()`)
- Production = Gunicorn (WSGI server)
- Both use the same Flask code!



