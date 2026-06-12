import fitz
import re
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Disable MuPDF noisy logs
fitz.TOOLS.mupdf_display_errors(False)


class PDFProcessor:
    def __init__(self, llm=None, vision_func= None, process_images=True):
        self.llm = llm
        self.vision_func = vision_func
        self.process_images = process_images

        # Optimized chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self.MIN_IMAGE_BYTES = 5000
       
        self.MIN_IMAGE_DIM   = 80

    # Improved author extraction
    def extract_authors(self, text: str) -> str:
        

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        header = lines[:30]

        false_positive_re = re.compile(
            r'^(digital object|index terms?|senior member|associate member|'
            r'corresponding author|date of|received|accepted|published|'
            r'doi|ieee|arxiv|preprint)',
            re.IGNORECASE
        )

        all_caps_name_re = re.compile(
            r'^(?:[A-Z][A-Z\s\-\.]{2,})$'   
        )
        initial_caps_re  = re.compile(
            r'^[A-Z]\.\s+[A-Z]{2,}$'         
        )
 
        skip_re = re.compile(
            r'\b(abstract|introduction|keywords|received|department|'
            r'university|institute|college|school|ieee|journal|'
            r'corresponding|digital|object|identifier)\b',
            re.IGNORECASE
        )
 
        author_lines: List[str] = []
        found_title  = False
 
        for line in header:
            # Skip metadata / DOI lines before title
            if re.match(r'^(Received|Digital Object|DOI|10\.\d{4})', line, re.IGNORECASE):
                continue
 
            # Treat the first non-metadata, non-caps long line as the title
            if not found_title and len(line) > 20 and not line.isupper():
                found_title = True
                continue
 
            if not found_title:
                continue
 
            # Once we hit affiliations / abstract, stop collecting
            if skip_re.search(line):
                if author_lines:        
                    break
                continue
 
            # Strip footnote markers and check if line is an author name
            stripped = re.sub(r'[\d¹²³⁴⁵⁶⁷⁸⁹⁰†‡∗*,;]', '', line).strip()
            stripped = re.sub(r'\(.*?\)', '', stripped).strip()
 
            if all_caps_name_re.match(stripped) or initial_caps_re.match(stripped):
                # Convert ALL CAPS → Title Case
                name = stripped.title()
                # Filter obvious non-names
                if not any(w in name.lower() for w in
                           ['and', 'ieee', 'senior', 'member', 'associate']):
                    author_lines.append(name)
 
        if author_lines:
            return ", ".join(author_lines)

        name_re = re.compile(
            r"\b"
            r"(?:[A-Z][a-z]+|[A-Z]\.)"   # first name or initial
            r"(?:\s+[A-Z]\.)*"            # optional middle initials
            r"\s+[A-Z][a-z]+"             # last name
            r"(?:-[A-Z][a-z]+)?"          # optional hyphenated last name
            r"\b"
        )

        bad_two_words = {
            'digital object', 'index terms', 'senior member', 'associate member',
            'corresponding author', 'et al', 'pp', 'vol'
        }
 

        for line in header:
            if false_positive_re.match(line):
                continue
            if skip_re.search(line):
                continue
 
            # Remove superscripts, footnote markers, emails, numbers
            cleaned = re.sub(r'[\d¹²³⁴⁵⁶⁷⁸⁹⁰†‡∗*]', '', line)
            cleaned = re.sub(r'\S+@\S+', '', cleaned)
            cleaned = re.sub(r'\(.*?\)', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
 
            #ALL CAPS author line 
            # e.g. "JOHN DOE, ALICE SMITH" → title-case then extract
            if cleaned.isupper() and len(cleaned) > 4:
                cleaned = cleaned.title()

        
            names = name_re.findall(cleaned)

           # Filter out false positives (single tokens, place names, etc.)
            names = [
                n.strip() for n in names
                if len(n.split()) >= 2
                and n.lower() not in bad_two_words

                and not any(w in n.lower() for w in [
                    'university', 'institute', 'college', 'department',
                    'laboratory', 'center', 'school', 'faculty',
                    'digital', 'object', 'index', 'terms'  
                ])
            ]

            if names:
                return ", ".join(names)
            

        for line in header:
            if re.match(r'corresponding author', line, re.IGNORECASE):
                # Extract name between ":" and "("
                match = re.search(r':\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', line)
                if match:
                    return match.group(1).strip() 
            
        
        # heuristic fallback 
        # Score each header line by how many tokens look like proper names.
        # Pick the line with the highest name-density that isn't an
        # institution / affiliation line.
        best_line   = None
        best_score  = 0
        token_re    = re.compile(r'\b[A-Z][a-z]{1,14}\b')
        bad_words   = {'university', 'institute', 'college', 'department',
                       'laboratory', 'center', 'school', 'faculty',
                       'abstract', 'introduction', 'keywords', 'received',
                       'published', 'journal', 'conference', 'proceedings'}
 
        for line in header:
            low = line.lower()
            if any(w in low for w in bad_words):
                continue
            if '@' in line or 'http' in line:
                continue
            tokens       = line.split()
            cap_tokens   = token_re.findall(line)
            if not tokens:
                continue
            score = len(cap_tokens) / len(tokens)  # ratio of name-like tokens
            if score > best_score and len(cap_tokens) >= 2:
                best_score = score
                best_line  = line
 
        if best_line:
            names = name_re.findall(best_line)
            names = [n.strip() for n in names if len(n.split()) >= 2]
            if names:
                return ", ".join(names)

        return "Unknown Author"
    # Fixed year extraction
    def extract_year(self, text: str) -> str:
        lines = text.lower().split("\n")

        # Priority lines
        for line in lines[:40]:
            if "received" in line or "published" in line or "accepted" in line:
                match = re.search(r'(20\d{2})', line)
                if match:
                    return match.group(1)

        # fallback
        years = re.findall(r'(20\d{2})', text)
        if years:
            return years[0]

        return "N/A"

    # Clean text instead of skipping
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove DOI
        text = re.sub(r'doi:\S+', '', text)

        # Remove reference section ONLY
        text = re.sub(r'References.*', '', text, flags=re.IGNORECASE)

        return text.strip()

    def process_pdfs(self, pdf_directory: str) -> List[Document]:
        all_docs = []
        pdf_files = list(Path(pdf_directory).glob("**/*.pdf"))

        for pdf_file in pdf_files:
            try:
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as e:
                print(f" Skipping file {pdf_file}: {e}")
                continue

            # Extract metadata
            try:
                first_page_text = doc[0].get_text()
            except:
                first_page_text = ""
            


            author_text = first_page_text
            if len(doc)>1:
                try:
                    author_text +="\n"+doc[1].get_text()
                except Exception:
                    pass

            author = self.extract_authors(author_text[:4000])
            year = self.extract_year(first_page_text)

            # Process pages
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                except:
                    continue

                base_metadata = {
                    "source_file": pdf_file.name,
                    "page": page_num + 1,
                    "author": author,
                    "date": year,
                    "type": "text"
                }

                try:
                    text_blocks = page.get_text("blocks", sort=True)
                    full_text = "\n\n".join(
                        b[4].strip() for b in text_blocks if b[4].strip()
                    )
                except:
                    continue

                if not full_text or len(full_text.strip()) < 80:
                    continue

                # CLEAN instead of SKIP
                full_text = self.clean_text(full_text)

                all_docs.append(Document(
                    page_content=full_text,
                    metadata=base_metadata
                ))

            doc.close()

        return all_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []

        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            final_chunks.extend(chunks)

        return final_chunks