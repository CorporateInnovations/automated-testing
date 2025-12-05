def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if size <= 0:
        return [text]

    chunks, start = [], 0
    overlap = max(0, min(overlap, size - 1))

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
