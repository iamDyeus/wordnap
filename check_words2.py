from pathlib import Path
from sentence_mixer.database.database import Database

db = Database(Path('data/sentence_mixer.db'))
db.initialize()
conn = db.connection

# Check for specific words we want to use
target_words = ['you', 'have', 'to', 'trust', 'in', 'something', 'your', 'gut', 'destiny', 
                'life', 'karma', 'whatever', 'because', 'believing', 'that', 'the', 'dots',
                'will', 'connect', 'down', 'road', 'give', 'confidence', 'follow', 'heart',
                'find', 'what', 'love', 'only', 'way', 'do', 'great', 'work', 'is',
                'if', 'not', 'found', 'it', 'yet', 'keep', 'looking',
                'time', 'limited', 'waste', 'living', 'someone', 'else', 'courage']

for word in sorted(set(target_words)):
    rows = conn.execute('SELECT COUNT(*) as cnt FROM words WHERE normalized_word = ?', (word,)).fetchone()
    count = rows['cnt'] if rows else 0
    status = "OK" if count > 0 else "MISSING"
    print(f'{word:15s} {count:3d} {status}')

db.close()
