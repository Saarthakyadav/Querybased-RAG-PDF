

import fitz
import re
from pathlib import Path

fitz.TOOLS.mupdf_display_errors(False)

PDF_DIR = "data/pdf_files"

def debug_pdf(pdf_path):
    print(f"\n{'='*60}")
    print(f"FILE: {pdf_path.name}")
    print('='*60)

    doc = fitz.open(str(pdf_path))

    # Show raw first-page text so we can see exactly what PyMuPDF reads
    first_page = doc[0].get_text()
    second_page = doc[1].get_text() if len(doc) > 1 else ""
    author_text = (first_page + "\n" + second_page)[:4000]

    print("\n--- RAW FIRST 1500 CHARS FROM PDF ---")
    print(repr(author_text[:1500]))   # repr shows hidden chars/newlines

    print("\n--- LINES (first 30) ---")
    lines = [ln.strip() for ln in author_text.split("\n") if ln.strip()]
    for i, ln in enumerate(lines[:30]):
        print(f"  [{i:02d}] {repr(ln)}")

    doc.close()

    # Now run the extractor and show which strategy fired
    print("\n--- AUTHOR EXTRACTOR TRACE ---")
    extract_authors_debug(author_text[:4000])


def extract_authors_debug(text: str):
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    header = lines[:30]

    name_re = re.compile(
        r"\b"
        r"(?:[A-Z][a-z]+|[A-Z]\.)"
        r"(?:\s+[A-Z]\.)*"
        r"\s+[A-Z][a-z]+"
        r"(?:-[A-Z][a-z]+)?"
        r"\b"
    )

    print("\nStrategy 1 — line-by-line name regex:")
    for i, line in enumerate(header):
        skip = re.search(
            r'\b(abstract|introduction|keywords|received|department|university|institute|college|school)\b',
            line, re.IGNORECASE
        )
        if skip:
            print(f"  [{i:02d}] SKIPPED (bad keyword: {skip.group()}) → {repr(line[:80])}")
            continue

        cleaned = re.sub(r'[\d¹²³⁴⁵⁶⁷⁸⁹⁰†‡∗*]', '', line)
        cleaned = re.sub(r'\S+@\S+', '', cleaned)
        cleaned = re.sub(r'\(.*?\)', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if cleaned.isupper() and len(cleaned) > 4:
            print(f"  [{i:02d}] ALL-CAPS → title-cased: {repr(cleaned[:80])}")
            cleaned = cleaned.title()

        names = name_re.findall(cleaned)
        names = [
            n.strip() for n in names
            if len(n.split()) >= 2
            and not any(w in n.lower() for w in [
                'university','institute','college','department',
                'laboratory','center','school','faculty'
            ])
        ]

        if names:
            print(f"  [{i:02d}] ✅ FOUND: {names}  ← from line: {repr(line[:80])}")
            return ", ".join(names)
        else:
            raw_matches = name_re.findall(cleaned)
            print(f"  [{i:02d}] no match | cleaned={repr(cleaned[:80])} | raw_re={raw_matches}")

    print("\nStrategy 3 — heuristic fallback:")
    best_line, best_score = None, 0
    token_re = re.compile(r'\b[A-Z][a-z]{1,14}\b')
    bad_words = {'university','institute','college','department','laboratory',
                 'center','school','faculty','abstract','introduction',
                 'keywords','received','published','journal','conference','proceedings'}

    for i, line in enumerate(header):
        low = line.lower()
        if any(w in low for w in bad_words): continue
        if '@' in line or 'http' in line: continue
        tokens = line.split()
        cap_tokens = token_re.findall(line)
        if not tokens: continue
        score = len(cap_tokens) / len(tokens)
        print(f"  [{i:02d}] score={score:.2f} cap_tokens={cap_tokens} → {repr(line[:80])}")
        if score > best_score and len(cap_tokens) >= 2:
            best_score = score
            best_line = line

    if best_line:
        names = name_re.findall(best_line)
        names = [n.strip() for n in names if len(n.split()) >= 2]
        if names:
            print(f"\n  ✅ FALLBACK FOUND: {names}")
            return ", ".join(names)

    print("\n  ❌ RESULT: Unknown Author")
    return "Unknown Author"


if __name__ == "__main__":
    pdf_files = list(Path(PDF_DIR).glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
    else:
        for pdf in pdf_files:
            debug_pdf(pdf)