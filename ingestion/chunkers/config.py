# ingestion/chunkers/config.py
CHUNK_CONFIGS = {
    '256T':  {'chunk_size': 256,  'chunk_overlap': 40},
    '512T':  {'chunk_size': 512,  'chunk_overlap': 80},
    '1024T': {'chunk_size': 1024, 'chunk_overlap': 150},
}
DEFAULT_CHUNK_CONFIG = '512T'
