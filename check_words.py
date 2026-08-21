from pathlib import Path
from sentence_mixer.database.database import Database

db = Database(Path('data/sentence_mixer.db'))
db.initialize()
conn = db.connection
rows = conn.execute('''
    SELECT normalized_word, COUNT(*) as cnt 
    FROM words 
    GROUP BY normalized_word 
    HAVING cnt > 2 
    ORDER BY cnt DESC 
    LIMIT 50
''').fetchall()
for row in rows:
    print(f'{row["normalized_word"]:15s} ({row["cnt"]} occurrences)')
db.close()
