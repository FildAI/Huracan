import os
import pandas as pd
import time
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.embeddings import OpenAIEmbedder, SentenceTransformerShort

def get_embedder(embedder_type):
    if embedder_type == 'openai':
        return OpenAIEmbedder()
    elif embedder_type == 'sentence-transformer':
        return SentenceTransformerShort()
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")

def process_chunk(chunk, text_column, embedder_type, embedding_column, pbar, batch_size, temp_path):
    embedder = get_embedder(embedder_type)
    if embedder_type == 'openai':
        idxs = list(chunk.index)
        texts = [row[text_column] if text_column else " ".join(str(row[col]) for col in chunk.columns if col != embedding_column) for _, row in chunk.iterrows()]
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_idxs = idxs[start:end]
            embeddings = embedder.create_embeddings_batch(batch_texts)
            for idx, embedding in zip(batch_idxs, embeddings):
                chunk.at[idx, embedding_column] = embedding
                pbar.update(1)
    else:
        for idx, row in chunk.iterrows():
            text = row[text_column] if text_column else " ".join(str(row[col]) for col in chunk.columns if col != embedding_column)
            embedding = embedder.create_embedding(str(text))
            chunk.at[idx, embedding_column] = embedding
            pbar.update(1)
    # Save chunk to temp file
    chunk.to_csv(temp_path, index=False)
    return temp_path

def process_csv(
    input_csv,
    output_csv,
    text_column=None,
    embedder_type='sentence-transformer',
    batch_size=32,
    autosave_interval=100,
    embedding_column='embedding',
    threads=1,
):
    if os.path.exists(output_csv):
        print(f"Resuming from {output_csv}")
        df = pd.read_csv(output_csv)
        processed_rows = df[df[embedding_column].notnull()].shape[0]
    else:
        df = pd.read_csv(input_csv)
        df[embedding_column] = None
        processed_rows = 0

    total_rows = df.shape[0]
    print(f"Total rows: {total_rows}, Already processed: {processed_rows}")

    MAX_CHUNK_SIZE = 1000

    if threads > 1:
        # Split DataFrame into chunks of max 10k rows
        num_chunks = (total_rows + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
        chunks = [df.iloc[i*MAX_CHUNK_SIZE : min((i+1)*MAX_CHUNK_SIZE, total_rows)] for i in range(num_chunks)]
        temp_files = []
        actual_threads = min(threads, num_chunks)
        with tqdm(total=total_rows, initial=processed_rows, desc="Embedding rows (parallel)") as pbar:
            with ThreadPoolExecutor(max_workers=actual_threads) as executor:
                for i, chunk in enumerate(chunks):
                    temp_path = f"{output_csv}.part{i}"
                    temp_files.append(temp_path)
                futures = [executor.submit(process_chunk, chunk, text_column, embedder_type, embedding_column, pbar, batch_size, temp_path) for chunk, temp_path in zip(chunks, temp_files)]
                for future in as_completed(futures):
                    future.result()  # Ensure all threads complete
        # Concatenate all temp files into final output
        dfs = [pd.read_csv(temp_path) for temp_path in temp_files]
        df_final = pd.concat(dfs).sort_index()
        print("Processing complete. Final save.")
        df_final.to_csv(output_csv, index=False)
        # Remove temp files
        for temp_path in temp_files:
            os.remove(temp_path)
    else:
        rows_since_save = 0
        embedder = get_embedder(embedder_type)
        with tqdm(total=total_rows, initial=processed_rows, desc="Embedding rows") as pbar:
            for start in range(processed_rows, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                batch = df.iloc[start:end]
                batch_idxs = batch.index.tolist()
                if embedder_type == 'openai':
                    texts = [row[text_column] if text_column else " ".join(str(row[col]) for col in df.columns if col != embedding_column) for _, row in batch.iterrows()]
                    embeddings = embedder.create_embeddings_batch(texts)
                    for idx, embedding in zip(batch_idxs, embeddings):
                        df.at[idx, embedding_column] = embedding
                        rows_since_save += 1
                        pbar.update(1)
                else:
                    for idx, row in batch.iterrows():
                        text = row[text_column] if text_column else " ".join(str(row[col]) for col in df.columns if col != embedding_column)
                        embedding = embedder.create_embedding(str(text))
                        df.at[idx, embedding_column] = embedding
                        rows_since_save += 1
                        pbar.update(1)
                if rows_since_save >= autosave_interval or end == total_rows:
                    print(f"Autosaving at row {end}")
                    df.to_csv(output_csv, index=False)
                    time.sleep(0.1)
                    rows_since_save = 0
            print("Processing complete. Final save.")
            df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch add embeddings to CSV with autosave and recovery.")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path (for autosave/recovery)')
    parser.add_argument('--text-column', default=None, help='Column name containing text to embed. If not specified, all columns will be concatenated.')
    parser.add_argument('--embedder', default='sentence-transformer', choices=['openai', 'sentence-transformer'], help='Embedder type')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--autosave-interval', type=int, default=100, help='Autosave every N rows')
    parser.add_argument('--embedding-column', default='embedding', help='Name of the embedding column')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads (map-reduce style)')
    args = parser.parse_args()

    process_csv(
        input_csv=args.input,
        output_csv=args.output,
        text_column=args.text_column,
        embedder_type=args.embedder,
        batch_size=args.batch_size,
        autosave_interval=args.autosave_interval,
        embedding_column=args.embedding_column,
        threads=args.threads,
    )
