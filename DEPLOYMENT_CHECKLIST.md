# 🚀 Flask ML Server Deployment Checklist

## Files to Upload to Server

### 1. Core Application Files
- ✅ `app.py` - Main Flask application
- ✅ `config.py` - Configuration file
- ✅ `wsgi.py` - WSGI entry point
- ✅ `requirements.txt` - Python dependencies
- ✅ `gunicorn_config.py` - Gunicorn configuration (IMPORTANT: Update bind address!)

### 2. Model File
- ✅ `datasets/best 2.pt` (or your active model file)
- Or ensure model path in database matches uploaded location

### 3. PHP Endpoint (if not already on server)
- ✅ `pest_detection_ml/api/get_active_model_path.php`

### 4. Configuration
- ✅ `config.php` - Database credentials (verify it's on server)

---

## ⚠️ CRITICAL: Fix Connection Timeout Issue

### Problem
Your Android app can't connect because:
1. Flask server might not be running
2. Server might be bound to `127.0.0.1` (localhost only)
3. Port 5001 might be blocked by firewall

### Solution 1: Update Gunicorn Config for External Access

**Edit `gunicorn_config.py` on your server:**

```python
# Change this line:
bind = "127.0.0.1:5001"  # ❌ Only localhost

# To this:
bind = "0.0.0.0:5001"  # ✅ Accept external connections
```

### Solution 2: Start Flask Server Correctly

**Option A: Using Gunicorn (Recommended for Production)**
```bash
cd /path/to/your/flask/app
gunicorn -c gunicorn_config.py wsgi:application
```

**Option B: Direct Flask (For Testing)**
```bash
cd /path/to/your/flask/app
python3 app.py
```

**Option C: Using systemd service (Auto-start)**
Create `/etc/systemd/system/agrishield-flask.service`:
```ini
[Unit]
Description=AgriShield Flask ML API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/your/flask/app
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/local/bin/gunicorn -c gunicorn_config.py wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable agrishield-flask
sudo systemctl start agrishield-flask
sudo systemctl status agrishield-flask
```

### Solution 3: Open Firewall Port

**Ubuntu/Debian:**
```bash
sudo ufw allow 5001/tcp
sudo ufw reload
```

**CentOS/RHEL:**
```bash
sudo firewall-cmd --permanent --add-port=5001/tcp
sudo firewall-cmd --reload
```

**cPanel/Shared Hosting:**
- Contact your hosting provider to open port 5001
- Or use a reverse proxy through Apache/Nginx on port 80/443

### Solution 4: Test Connection

**From your server:**
```bash
curl http://localhost:5001/health
```

**From your computer (replace with your domain):**
```bash
curl http://agrishield.bccbsis.com:5001/health
```

---

## 📋 Server Setup Steps

1. **Upload files** to your server
2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```
3. **Update `gunicorn_config.py`** - Change `bind = "0.0.0.0:5001"`
4. **Verify `config.php`** has correct database credentials
5. **Start Flask server** (see Solution 2 above)
6. **Open firewall port** (see Solution 3 above)
7. **Test connection** (see Solution 4 above)

---

## 🔍 Troubleshooting

### Check if Flask is running:
```bash
ps aux | grep gunicorn
# or
ps aux | grep python
```

### Check if port 5001 is listening:
```bash
netstat -tulpn | grep 5001
# or
ss -tulpn | grep 5001
```

### View Flask logs:
```bash
# If using gunicorn:
tail -f /var/log/agrishield-ml-api-error.log

# If running directly:
# Check console output or log file
```

### Test from Android app:
- Ensure device has internet connection
- Try accessing `http://agrishield.bccbsis.com:5001/health` in browser first
- Check Android logcat for detailed error messages

---

## ✅ Success Indicators

- ✅ `curl http://agrishield.bccbsis.com:5001/health` returns JSON
- ✅ Android app connects without timeout
- ✅ Images upload successfully
- ✅ Database entries are created

