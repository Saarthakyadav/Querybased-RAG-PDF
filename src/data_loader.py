import fitz
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

fitz.TOOLS.mupdf_display_errors(False)
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, llm=None, vision_func=None, process_images=True):
        self.llm = llm
        self.vision_func = vision_func
        self.process_images = process_images

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self.MIN_IMAGE_BYTES = 5000
        self.MIN_IMAGE_DIM   = 80

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def extract_authors(self, text: str) -> str:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        header = lines[:30]

        false_positive_re = re.compile(
            r'^(digital object|index terms?|senior member|associate member|'
            r'corresponding author|date of|received|accepted|published|'
            r'doi|ieee|arxiv|preprint)',
            re.IGNORECASE
        )
        all_caps_name_re = re.compile(r'^(?:[A-Z][A-Z\s\-\.]{2,})$')
        initial_caps_re  = re.compile(r'^[A-Z]\.\s+[A-Z]{2,}$')
        skip_re = re.compile(
            r'\b(abstract|introduction|keywords|received|department|'
            r'university|institute|college|school|ieee|journal|'
            r'corresponding|digital|object|identifier)\b',
            re.IGNORECASE
        )

        author_lines: List[str] = []
        found_title  = False

        for line in header:
            if re.match(r'^(Received|Digital Object|DOI|10\.\d{4})', line, re.IGNORECASE):
                continue
            if not found_title and len(line) > 20 and not line.isupper():
                found_title = True
                continue
            if not found_title:
                continue
            if skip_re.search(line):
                if author_lines:
                    break
                continue
            stripped = re.sub(r'[\d¹²³⁴⁵⁶⁷⁸⁹⁰†‡∗*,;]', '', line).strip()
            stripped = re.sub(r'\(.*?\)', '', stripped).strip()
            if all_caps_name_re.match(stripped) or initial_caps_re.match(stripped):
                name = stripped.title()
                if not any(w in name.lower() for w in
                           ['and', 'ieee', 'senior', 'member', 'associate']):
                    author_lines.append(name)

        if author_lines:
            return ", ".join(author_lines)

        name_re = re.compile(
            r"\b"
            r"(?:[A-Z][a-z]+|[A-Z]\.)"
            r"(?:\s+[A-Z]\.)*"
            r"\s+[A-Z][a-z]+"
            r"(?:-[A-Z][a-z]+)?"
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
            cleaned = re.sub(r'[\d¹²³⁴⁵⁶⁷⁸⁹⁰†‡∗*]', '', line)
            cleaned = re.sub(r'\S+@\S+', '', cleaned)
            cleaned = re.sub(r'\(.*?\)', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned.isupper() and len(cleaned) > 4:
                cleaned = cleaned.title()
            names = name_re.findall(cleaned)
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
                match = re.search(r':\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', line)
                if match:
                    return match.group(1).strip()

        best_line   = None
        best_score  = 0
        token_re    = re.compile(r'\b[A-Z][a-z]{1,14}\b')
        bad_words   = {
            'university', 'institute', 'college', 'department',
            'laboratory', 'center', 'school', 'faculty',
            'abstract', 'introduction', 'keywords', 'received',
            'published', 'journal', 'conference', 'proceedings'
        }

        for line in header:
            low = line.lower()
            if any(w in low for w in bad_words):
                continue
            if '@' in line or 'http' in line:
                continue
            tokens     = line.split()
            cap_tokens = token_re.findall(line)
            if not tokens:
                continue
            score = len(cap_tokens) / len(tokens)
            if score > best_score and len(cap_tokens) >= 2:
                best_score = score
                best_line  = line

        if best_line:
            names = name_re.findall(best_line)
            names = [n.strip() for n in names if len(n.split()) >= 2]
            if names:
                return ", ".join(names)

        return "Unknown Author"

    def extract_year(self, text: str) -> str:
        lines = text.lower().split("\n")
        for line in lines[:40]:
            if "received" in line or "published" in line or "accepted" in line:
                match = re.search(r'(20\d{2})', line)
                if match:
                    return match.group(1)
        years = re.findall(r'(20\d{2})', text)
        return years[0] if years else "N/A"

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'doi:\S+', '', text)
        # FIX: added re.DOTALL so .* matches across newlines, removing the
        # entire References section rather than just the header line.
        text = re.sub(r'References.*', '', text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    # ------------------------------------------------------------------
    # Image processing (parallel)
    # ------------------------------------------------------------------

    def _describe_images_for_page(
        self, page: fitz.Page, base_metadata: dict
    ) -> List[Document]:
        """
        Extract images from a page and describe them.
        FIX: image API calls are now run in parallel via ThreadPoolExecutor
        in process_pdfs(), not sequentially here.  This method returns raw
        (image_bytes, metadata) tuples for the caller to dispatch.
        """
        if not self.process_images or not self.vision_func:
            return []

        image_docs = []
        try:
            image_list = page.get_images(full=True)
        except Exception as exc:
            logger.warning("Failed to list images on page %s: %s", base_metadata.get("page"), exc)
            return []

        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image.get("image", b"")
                if (
                    not image_bytes
                    or len(image_bytes) < self.MIN_IMAGE_BYTES
                    or base_image.get("width", 0) < self.MIN_IMAGE_DIM
                    or base_image.get("height", 0) < self.MIN_IMAGE_DIM
                ):
                    continue
                image_docs.append((image_bytes, dict(base_metadata)))
            except Exception as exc:
                logger.debug("Skipping image xref=%s: %s", xref, exc)

        return image_docs

    # ------------------------------------------------------------------
    # PDF processing
    # ------------------------------------------------------------------

    def process_pdfs(self, pdf_directory: str) -> List[Document]:
        all_docs: List[Document] = []
        pending_images: List[tuple] = []   # (image_bytes, metadata) for parallel description

        pdf_files = list(Path(pdf_directory).glob("**/*.pdf"))
        logger.info("Processing %d PDF file(s) in %s", len(pdf_files), pdf_directory)

        for pdf_file in pdf_files:
            try:
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as e:
                # FIX: was bare `except:` — now logs the actual error.
                logger.error("Skipping file %s: %s", pdf_file, e)
                continue

            # Extract metadata from first (and optionally second) page.
            try:
                first_page_text = doc[0].get_text()
            except Exception as exc:
                logger.warning("Could not read first page of %s: %s", pdf_file.name, exc)
                first_page_text = ""

            author_text = first_page_text
            if len(doc) > 1:
                try:
                    author_text += "\n" + doc[1].get_text()
                except Exception:
                    pass

            author = self.extract_authors(author_text[:4000])
            year   = self.extract_year(first_page_text)

            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                except Exception as exc:
                    # FIX: was bare `except:` — now logs.
                    logger.warning("Could not load page %d of %s: %s", page_num, pdf_file.name, exc)
                    continue

                base_metadata = {
                    "source_file": pdf_file.name,
                    "page":        page_num + 1,
                    "author":      author,
                    "date":        year,
                    "type":        "text",
                }

                try:
                    text_blocks = page.get_text("blocks", sort=True)
                    full_text   = "\n\n".join(
                        b[4].strip() for b in text_blocks if b[4].strip()
                    )
                except Exception as exc:
                    logger.warning("Text extraction failed on page %d of %s: %s",
                                   page_num, pdf_file.name, exc)
                    continue

                if not full_text or len(full_text.strip()) < 80:
                    continue

                full_text = self.clean_text(full_text)
                all_docs.append(Document(page_content=full_text, metadata=base_metadata))

                # Collect images to process in parallel after the text loop.
                if self.process_images and self.vision_func:
                    pending_images.extend(
                        self._describe_images_for_page(page, base_metadata)
                    )

            doc.close()

        # FIX: describe all images in parallel instead of serially.
        # This drastically reduces ingestion time for image-heavy PDFs.
        if pending_images:
            logger.info("Describing %d images (parallel) …", len(pending_images))
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(self.vision_func, img_bytes): meta
                    for img_bytes, meta in pending_images
                }
                for future in as_completed(futures):
                    meta = futures[future]
                    try:
                        description = future.result()
                    except Exception as exc:
                        logger.warning("Image description failed: %s", exc)
                        continue
                    if description:
                        img_meta = dict(meta)
                        img_meta["type"] = "image"
                        all_docs.append(Document(
                            page_content=description,
                            metadata=img_meta,
                        ))

        logger.info("Extracted %d documents from %d PDF(s).", len(all_docs), len(pdf_files))
        return all_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks: List[Document] = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            final_chunks.extend(chunks)
        return final_chunks