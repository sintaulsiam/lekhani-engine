#!/usr/bin/env python3
"""
Lekhani Multi-Domain Balanced Bengali Corpus Builder

Downloads and cleans real, high-quality Bengali data across:
1. Conversational & Social comments (YouTube BN, Coke Studio BN)
2. Daily dialogues, advice, and instructions (Alpaca BN)
3. Encyclopedic, scientific, and literary Bengali (Wikipedia BN)

Applies:
- NFC Unicode normalization
- Cleaning of URLs, English-only noise, @handles, HTML tags
- Minimum Bengali script ratio filter (>= 60%)
- Deduplication via MD5 sentence hashes
"""
import hashlib
import os
import re
import sys
import time
import unicodedata

OUT_DIR = "/mnt/data/lekhani-engine/data/corpus"
OUT_FILE = os.path.join(OUT_DIR, "bengali_corpus.txt")
os.makedirs(OUT_DIR, exist_ok=True)

URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
HTML_REGEX = re.compile(r'<[^>]+>')
HANDLE_REGEX = re.compile(r'@\w+')

BN_RANGE_START = ord('\u0980')
BN_RANGE_END = ord('\u09FF')

def is_bengali_char(c):
    return BN_RANGE_START <= ord(c) <= BN_RANGE_END

def clean_sentence(text):
    text = URL_REGEX.sub('', text)
    text = HTML_REGEX.sub('', text)
    text = HANDLE_REGEX.sub('', text)
    text = unicodedata.normalize('NFC', text.strip())
    return text

def is_valid_bengali_sentence(s):
    if len(s) < 8:
        return False
    words = s.split()
    if len(words) < 2:
        return False
    # Check Bengali character ratio
    letters = [c for c in s if c.isalpha() or is_bengali_char(c)]
    if not letters:
        return False
    bn_chars = sum(1 for c in letters if is_bengali_char(c))
    return (bn_chars / len(letters)) >= 0.60

def split_sentences(text):
    # Split on newline, dari, question mark, exclamation mark
    chunks = re.split(r'[\r\n।?!]+', text)
    return [c.strip() for c in chunks if c.strip()]

def main():
    from datasets import load_dataset
    
    seen = set()
    total_tokens = 0
    total_lines = 0
    start_time = time.time()
    
    print("=" * 65)
    print("🚀 Lekhani Balanced Multi-Domain Bengali Corpus Ingestion")
    print(f"Output: {OUT_FILE}")
    print("=" * 65)

    with open(OUT_FILE, 'w', encoding='utf-8') as out_f:
        
        # ── 1. Conversational YouTube Comments (215k items) ──
        print("\n[1/4] Processing Conversational Comments (irfanhossainsust)...", flush=True)
        try:
            ds = load_dataset("irfanhossainsust/bangla-youtube-comments-sentiment", split="train")
            conv_lines = 0
            for item in ds:
                raw = item.get("text", "") or ""
                raw = clean_sentence(raw)
                for s in split_sentences(raw):
                    if is_valid_bengali_sentence(s):
                        h = hashlib.md5(s.encode('utf-8')).digest()
                        if h not in seen:
                            seen.add(h)
                            out_f.write(s + "\n")
                            total_tokens += len(s.split())
                            total_lines += 1
                            conv_lines += 1
            print(f"   ✓ Wrote {conv_lines:,} conversational comment sentences.", flush=True)
        except Exception as e:
            print(f"   ⚠ Warning: {e}", flush=True)

        # ── 2. Coke Studio Cultural & Casual Comments (57k items) ──
        print("\n[2/4] Processing Cultural/Casual Dialogue (coke_studio_bangla)...", flush=True)
        try:
            ds = load_dataset("ragibhasan/coke_studio_bangla_sentiment", split="train")
            coke_lines = 0
            for item in ds:
                raw = item.get("text_clean", "") or item.get("text_raw", "") or ""
                raw = clean_sentence(raw)
                for s in split_sentences(raw):
                    if is_valid_bengali_sentence(s):
                        h = hashlib.md5(s.encode('utf-8')).digest()
                        if h not in seen:
                            seen.add(h)
                            out_f.write(s + "\n")
                            total_tokens += len(s.split())
                            total_lines += 1
                            coke_lines += 1
            print(f"   ✓ Wrote {coke_lines:,} dialogue sentences.", flush=True)
        except Exception as e:
            print(f"   ⚠ Warning: {e}", flush=True)

        # ── 3. Daily QA, Advice & Instructions (Alpaca BN, 18k pairs) ──
        print("\n[3/4] Processing Daily Q&A, Advice & Instructions (nihalbaig/alpaca_bangla)...", flush=True)
        try:
            ds = load_dataset("nihalbaig/alpaca_bangla", split="train")
            qa_lines = 0
            for item in ds:
                text_parts = [
                    item.get("instruction", ""),
                    item.get("input", ""),
                    item.get("output", "")
                ]
                for raw in text_parts:
                    if not raw:
                        continue
                    raw = clean_sentence(raw)
                    for s in split_sentences(raw):
                        if is_valid_bengali_sentence(s):
                            h = hashlib.md5(s.encode('utf-8')).digest()
                            if h not in seen:
                                seen.add(h)
                                out_f.write(s + "\n")
                                total_tokens += len(s.split())
                                total_lines += 1
                                qa_lines += 1
            print(f"   ✓ Wrote {qa_lines:,} instruction & daily Q&A sentences.", flush=True)
        except Exception as e:
            print(f"   ⚠ Warning: {e}", flush=True)

        # ── 4. Encyclopedic & Literary Foundation (Wikipedia BN, streaming) ──
        print("\n[4/4] Streaming Bengali Wikipedia (wikimedia/wikipedia)...", flush=True)
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train", streaming=True)
            wiki_lines = 0
            for i, item in enumerate(ds):
                raw = clean_sentence(item.get("text", ""))
                for s in split_sentences(raw):
                    if is_valid_bengali_sentence(s):
                        h = hashlib.md5(s.encode('utf-8')).digest()
                        if h not in seen:
                            seen.add(h)
                            out_f.write(s + "\n")
                            total_tokens += len(s.split())
                            total_lines += 1
                            wiki_lines += 1
                if i % 1000 == 0 and i > 0:
                    print(f"   ... streamed {i:,} articles ({total_tokens:,} total tokens)", flush=True)
                if i >= 12000:
                    break
            print(f"   ✓ Wrote {wiki_lines:,} Wikipedia sentences.", flush=True)
        except Exception as e:
            print(f"   ⚠ Warning: {e}", flush=True)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print("\n" + "=" * 65)
    print("✅ Balanced Multi-Domain Corpus Preparation Complete!")
    print(f"   Total Unique Sentences: {total_lines:,}")
    print(f"   Total Tokens:           {total_tokens:,} (~{total_tokens/1_000_000:.1f}M words)")
    print(f"   Corpus File Size:       {file_size_mb:.2f} MB")
    print(f"   Total Ingestion Time:   {elapsed:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()
