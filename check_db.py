"""Quick script to check the database state."""
from pathlib import Path
from sentence_mixer.database.database import Database

db = Database(Path("data/sentence_mixer.db"))
db.initialize()
conn = db.connection

rows = conn.execute("SELECT * FROM videos").fetchall()
print(f"Videos: {len(rows)}")
for r in rows:
    print(f"  {dict(r)}")

words = conn.execute("SELECT COUNT(*) as cnt FROM words").fetchone()
cnt = words["cnt"]
print(f"Words: {cnt}")

if cnt > 0:
    sample = conn.execute("SELECT * FROM words LIMIT 10").fetchall()
    for w in sample:
        print(f"  {dict(w)}")

db.close()
