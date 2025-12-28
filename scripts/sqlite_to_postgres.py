#!/usr/bin/env python3
"""
Simple migration script:
 - reads rows from sqlite (bot_cache.db -> files table)
 - inserts into postgres (same schema assumed)
Usage:
  pip install psycopg2-binary
  python sqlite_to_postgres.py --sqlite ./bot_cache.db --pg "postgresql://botuser:botpass@localhost:5432/botdb"
"""
import sqlite3
import psycopg2
import argparse
from psycopg2.extras import execute_values

parser = argparse.ArgumentParser()
parser.add_argument("--sqlite", required=True)
parser.add_argument("--pg", required=True, help="Postgres DSN")
args = parser.parse_args()

src = sqlite3.connect(args.sqlite)
sc = src.cursor()

dst = psycopg2.connect(args.pg)
dc = dst.cursor()

# Read rows from sqlite: adapt columns as your schema
sc.execute("SELECT user_id, file_id, artist, title, url, thumbnail, duration, popularity, genre, cache_path, tg_file_id, last_accessed FROM files")
rows = sc.fetchall()

# Create table in postgres if not exists (adjust types)
dc.execute("""
CREATE TABLE IF NOT EXISTS files (
  user_id BIGINT,
  file_id TEXT PRIMARY KEY,
  artist TEXT,
  title TEXT,
  url TEXT,
  thumbnail TEXT,
  duration INTEGER,
  popularity INTEGER,
  genre TEXT,
  cache_path TEXT,
  tg_file_id TEXT,
  last_accessed TIMESTAMP
)
""")
dst.commit()

# Insert rows in batches
batch = []
for r in rows:
    batch.append(r)
    if len(batch) >= 200:
        execute_values(dc,
                       "INSERT INTO files (user_id, file_id, artist, title, url, thumbnail, duration, popularity, genre, cache_path, tg_file_id, last_accessed) VALUES %s ON CONFLICT (file_id) DO UPDATE SET user_id=EXCLUDED.user_id, cache_path=EXCLUDED.cache_path, last_accessed=EXCLUDED.last_accessed",
                       batch)
        dst.commit()
        batch.clear()

if batch:
    execute_values(dc,
                   "INSERT INTO files (user_id, file_id, artist, title, url, thumbnail, duration, popularity, genre, cache_path, tg_file_id, last_accessed) VALUES %s ON CONFLICT (file_id) DO UPDATE SET user_id=EXCLUDED.user_id, cache_path=EXCLUDED.cache_path, last_accessed=EXCLUDED.last_accessed",
                   batch)
    dst.commit()

src.close()
dst.close()
print("Migration finished: {} rows".format(len(rows)))
