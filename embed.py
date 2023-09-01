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
    return await conn.fetch("SELECT id, content FROM discord WHERE embedding IS NULL LIMIT 100")  # Fetching 100 at a time, you can adjust this number

async def update_embeddings(conn, embeddings):
    await conn.executemany(
        "UPDATE discord SET embedding = $1::float4[] WHERE id = $2",
        embeddings
    )


async def embed_messages():
    conn = await asyncpg.connect(db_url)

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
        embeddings = [(embedding.numpy().tolist(), record['id']) for record, embedding in zip(records, sentence_embeddings)]
        await update_embeddings(conn, embeddings)
        print('done')

    await conn.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(embed_messages())

