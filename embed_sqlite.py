# modified version of ./embed.py

import sqlite3
import torch
from transformers import AutoTokenizer, AutoModel
import time

device = "cuda"
batch_size = 128

print('Loading model and tokenizer...', flush=True, end=' ')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en')
model = AutoModel.from_pretrained('BAAI/bge-base-en')
model = model.to(device)
model.eval()
torch.set_grad_enabled(False)
print('done')

def initialize_db(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY,
        content TEXT,
        embedding BLOB
    );
    """)
    conn.commit()

def fetch_null_embeddings(conn):
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, content FROM messages WHERE embedding IS NULL OR length(embedding) = 0 LIMIT {batch_size}")
    return cursor.fetchall()

def update_embeddings(conn, embeddings):
    cursor = conn.cursor()
    cursor.executemany(
        "UPDATE messages SET embedding = ? WHERE id = ?",
        embeddings
    )
    conn.commit()

def embed_messages():
    initialize_db(conn)

    throughput_avg = None
    while True:
        records = fetch_null_embeddings(conn)
        if not records:
            print('No records found. waiting...')
            time.sleep(60)
            continue

        start = time.monotonic()
        sentences = [record[1] for record in records]

        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}  # Move to cuda:1

        print(f'Embedding {len(records)} records...', flush=True, end=' ')
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.cpu()  # Move back to CPU
        print('done')

        print(f"Updating {len(sentence_embeddings)} records in the db...", flush=True, end=' ')
        embeddings = [(embedding.numpy().tobytes(), record[0]) for record, embedding in zip(records, sentence_embeddings)]
        update_embeddings(conn, embeddings)
        took = time.monotonic() - start
        throughput = len(sentence_embeddings) / took
        throughput_avg = throughput if throughput_avg is None else 0.99 * throughput_avg + 0.01 * throughput

        print(f'done in {took:.2f}s throughput: {len(sentence_embeddings) / took:.2f} records/s, avg: {throughput_avg:.2f} records/s')


if __name__ == "__main__":
    conn = sqlite3.connect('/workspace/eleuther.db')
    try:
        embed_messages()
    finally:
        conn.close()

