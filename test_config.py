#!/usr/bin/env python3
"""
Test script to verify production configuration
"""

from config import DB_CONFIG, PHP_BASE_URL, FLASK_HOST, FLASK_PORT, FLASK_DEBUG

print("=" * 60)
print("Production Configuration Test")
print("=" * 60)
print()
print("Database Configuration:")
print(f"  Host: {DB_CONFIG['host']}")
print(f"  User: {DB_CONFIG['user']}")
print(f"  Database: {DB_CONFIG['database']}")
print(f"  Charset: {DB_CONFIG['charset']}")
print()
print("PHP Base URL:")
print(f"  {PHP_BASE_URL}")
print()
print("Flask Settings:")
print(f"  Host: {FLASK_HOST}")
print(f"  Port: {FLASK_PORT}")
print(f"  Debug: {FLASK_DEBUG}")
print()
print("=" * 60)
print("✅ Configuration loaded successfully!")
print("=" * 60)

