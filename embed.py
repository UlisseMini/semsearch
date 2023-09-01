import asyncpg
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import time
import os

load_dotenv()

db_url = os.environ['DATABASE_URL']

print('Loading model and tokenizer...', flush=True, end=' ')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en')
model = AutoModel.from_pretrained('BAAI/bge-base-en')
model.eval(); torch.set_grad_enabled(False)
print('done')

async def fetch_null_embeddings(conn):
    return await conn.fetch("SELECT id, content FROM discord WHERE embedding IS NULL LIMIT 10")

async def update_embeddings(conn, embeddings):
    await conn.executemany(
        "UPDATE discord SET embedding = $1::float4[] WHERE id = $2",
        embeddings
    )


async def embed_messages():
    conn = await asyncpg.connect(db_url)

    throughput_avg = 0
    while True:
        records = await fetch_null_embeddings(conn)
        if not records:
            print('No records found. waiting...')
            time.sleep(60)
            continue

        sentences = [record['content'] for record in records]

        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        print(f'Embedding {len(records)} records...', flush=True, end=' ')
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        print('done')

        print(f"Updating {len(sentence_embeddings)} records in the db...", flush=True, end=' ')
        start = time.monotonic()
        embeddings = [(embedding.numpy().tolist(), record['id']) for record, embedding in zip(records, sentence_embeddings)]
        await update_embeddings(conn, embeddings)
        took = time.monotonic() - start
        print(f'done in {took:.2f}s throughput: {len(sentence_embeddings) / took:.2f} records/s, avg: {throughput_avg:.2f} records/s')

        # update throughput average with exponential moving average
        throughput_avg = 0.9 * throughput_avg + 0.1 * (len(sentence_embeddings) / took)

    await conn.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(embed_messages())

