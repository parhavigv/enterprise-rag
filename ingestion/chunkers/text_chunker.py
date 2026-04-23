from typing import List, Dict, Any


def chunk_text(
    text: str,
    source: str,
    doc_type: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []
    words = text.split()
    chunks = []
    current_words = []
    current_len = 0
    chunk_index = 0
    char_start = 0
    for word in words:
        word_len = len(word) + 1
        if current_len + word_len > chunk_size and current_words:
            chunks.append({"text": " ".join(current_words), "source": source, "doc_type": doc_type, "chunk_index": chunk_index, "char_start": char_start})
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_words):
                if overlap_len + len(w) + 1 <= chunk_overlap:
                    overlap_words.insert(0, w)
                    overlap_len += len(w) + 1
                else:
                    break
            char_start += current_len - overlap_len
            current_words = overlap_words
            current_len = overlap_len
            chunk_index += 1
        current_words.append(word)
        current_len += word_len
    if current_words:
        chunks.append({"text": " ".join(current_words), "source": source, "doc_type": doc_type, "chunk_index": chunk_index, "char_start": char_start})
    return chunks