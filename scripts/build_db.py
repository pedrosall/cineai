"""
build_db.py — Migra director_map.pkl, company_map.pkl y cast_map.pkl
(diccionarios "nombre -> tasa de éxito histórica") a una base de datos
SQLite en models/cineai.db.

Se ejecuta UNA VEZ (o cada vez que reentrenes y regeneres los .pkl).
La API en producción solo lee de cineai.db, nunca de los .pkl originales.

Uso:
    python scripts/build_db.py
"""

import sqlite3
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DB_PATH = os.path.join(MODELS_DIR, "cineai.db")


def build():
    director_map = joblib.load(os.path.join(MODELS_DIR, "director_map.pkl"))
    company_map = joblib.load(os.path.join(MODELS_DIR, "company_map.pkl"))
    cast_map = joblib.load(os.path.join(MODELS_DIR, "cast_map.pkl"))
    global_rate = float(joblib.load(os.path.join(MODELS_DIR, "global_rate.pkl")))

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # regenerar limpio cada vez, evita datos huérfanos

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("CREATE TABLE directors (name TEXT PRIMARY KEY, success_rate REAL NOT NULL)")
    cur.execute("CREATE TABLE companies (name TEXT PRIMARY KEY, success_rate REAL NOT NULL)")
    cur.execute("CREATE TABLE cast_members (name TEXT PRIMARY KEY, success_rate REAL NOT NULL)")
    cur.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value REAL NOT NULL)")

    cur.executemany("INSERT INTO directors VALUES (?, ?)", director_map.items())
    cur.executemany("INSERT INTO companies VALUES (?, ?)", company_map.items())
    cur.executemany("INSERT INTO cast_members VALUES (?, ?)", cast_map.items())
    cur.execute("INSERT INTO meta VALUES ('global_rate', ?)", (global_rate,))

    conn.commit()

    # Verificación rápida antes de cerrar
    for table in ["directors", "companies", "cast_members"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count} filas")
    print(f"global_rate: {global_rate:.4f}")
    print(f"\nBase de datos creada en: {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    build()