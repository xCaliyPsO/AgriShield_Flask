# Production Deployment Guide - Flask ML API

## 🚀 Production-Ready Configuration

The Flask ML API is now configured for **production deployment** with online database support.

---

## ✅ **What's Configured**

### 1. **Database Configuration**
- ✅ **Automatically loads from `config.php`**
- ✅ **Uses online database credentials:**
  - Host: `localhost` (or your DB host)
  - User: `u520834156_uAShield2025`
  - Database: `u520834156_dbAgriShield`
- ✅ **Environment variable override support**

### 2. **Production Settings**
- ✅ **Debug mode disabled** in production
- ✅ **Gunicorn configured** for production
- ✅ **Proper logging** configured
- ✅ **PHP API calls** use configurable base URL

### 3. **Model Path Resolution**
- ✅ **Database lookup** via PHP endpoint
- ✅ **Configurable PHP base URL** (not hardcoded)
- ✅ **Fallback paths** for reliability

---

## 📋 **Deployment Steps**

### **Step 1: Set Environment Variables**

Create a `.env` file or set environment variables:

```bash
# Production mode
export PRODUCTION=true

# PHP Base URL (for API calls)
export PHP_BASE_URL=https://yourdomain.com/Proto1

# Database (optional - will use config.php if not set)
export DB_HOST=localhost
export DB_USER=u520834156_uAShield2025
export DB_PASSWORD=:JqjB0@0zb6v
export DB_NAME=u520834156_dbAgriShield

# Flask settings
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5001
export FLASK_DEBUG=false
```

### **Step 2: Install Dependencies**

```bash
cd AgriShield_ML_Flask
pip install -r requirements.txt
```

### **Step 3: Test Configuration**

```bash
# Test if config loads correctly
python -c "from config import DB_CONFIG, PHP_BASE_URL; print('DB:', DB_CONFIG); print('PHP URL:', PHP_BASE_URL)"
```

### **Step 4: Run with Gunicorn (Production)**

```bash
cd AgriShield_ML_Flask
gunicorn -c gunicorn_config.py wsgi:application
```

### **Step 5: Auto-Start on Server Reboot**

#### **Option A: Systemd (Linux)**

Create `/etc/systemd/system/agrishield-ml-api.service`:

```ini
[Unit]
Description=AgriShield ML API (Gunicorn)
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/Proto1/AgriShield_ML_Flask
Environment="PATH=/usr/bin:/usr/local/bin"
Environment="PRODUCTION=true"
Environment="PHP_BASE_URL=https://yourdomain.com/Proto1"
ExecStart=/usr/local/bin/gunicorn -c gunicorn_config.py wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable agrishield-ml-api
sudo systemctl start agrishield-ml-api
sudo systemctl status agrishield-ml-api
```

#### **Option B: Supervisor**

Create `/etc/supervisor/conf.d/agrishield-ml-api.conf`:

```ini
[program:agrishield-ml-api]
command=/usr/local/bin/gunicorn -c gunicorn_config.py wsgi:application
directory=/path/to/Proto1/AgriShield_ML_Flask
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/agrishield-ml-api.log
environment=PRODUCTION="true",PHP_BASE_URL="https://yourdomain.com/Proto1"
```

Reload:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start agrishield-ml-api
```

#### **Option C: PM2 (Node.js process manager)**

```bash
pm2 start gunicorn --name agrishield-ml-api -- -c gunicorn_config.py wsgi:application
pm2 save
pm2 startup
```

---

## 🔧 **Configuration Files**

### **config.py**
- ✅ Loads database from `config.php`
- ✅ Supports environment variables
- ✅ Production-ready defaults

### **gunicorn_config.py**
- ✅ Production server settings
- ✅ Multiple workers
- ✅ Proper logging
- ✅ Timeout settings for ML inference

### **wsgi.py**
- ✅ WSGI entry point
- ✅ Production-ready

---

## 🌐 **PHP Base URL Configuration**

The Flask API needs to know where your PHP backend is located.

### **Local Development:**
```bash
export PHP_BASE_URL=http://localhost/Proto1
```

### **Production:**
```bash
export PHP_BASE_URL=https://yourdomain.com/Proto1
```

Or set in `.env` file:
```
PHP_BASE_URL=https://yourdomain.com/Proto1
```

---

## ✅ **Verification**

### **1. Check Database Connection:**
```bash
python -c "from config import DB_CONFIG; import pymysql; conn = pymysql.connect(**DB_CONFIG); print('✅ Database connected!'); conn.close()"
```

### **2. Test API:**
```bash
curl http://localhost:5001/health
```

### **3. Check Logs:**
```bash
tail -f /var/log/agrishield-ml-api-access.log
tail -f /var/log/agrishield-ml-api-error.log
```

---

## 🔒 **Security Notes**

1. **Bind to 127.0.0.1** - Only accessible from localhost (PHP calls it)
2. **Firewall** - Ensure port 5001 is not exposed externally
3. **Database credentials** - Stored in `config.php` (not in code)
4. **Environment variables** - Use for sensitive data

---

## 📝 **Troubleshooting**

### **Database Connection Failed:**
- Check `config.php` exists and has correct credentials
- Verify database is accessible from server
- Check firewall rules

### **PHP API Calls Failing:**
- Set `PHP_BASE_URL` environment variable
- Verify PHP endpoint is accessible
- Check network connectivity

### **Model Not Found:**
- Verify model file exists at fallback path
- Check file permissions
- Review logs for model path resolution

---

## 🎯 **Summary**

✅ **Database:** Configured from `config.php` (online DB)  
✅ **Production:** Gunicorn ready  
✅ **Auto-start:** Systemd/Supervisor/PM2 ready  
✅ **Configurable:** Environment variables supported  
✅ **Secure:** Localhost binding, proper logging  

**Status: PRODUCTION READY! 🚀**

