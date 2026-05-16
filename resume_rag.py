"""Load, parse, and index resumes into ChromaDB with section-aware chunking."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


RESUMES_DIR = "./data/resumes"
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


SECTION_HEADERS = {
    "SUMMARY",
    "PROFESSIONAL SUMMARY",
    "PROFILE",
    "OBJECTIVE",
    "SKILLS",
    "TECHNICAL SKILLS",
    "CORE COMPETENCIES",
    "EXPERIENCE",
    "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE",
    "EMPLOYMENT HISTORY",
    "EDUCATION",
    "ACADEMIC BACKGROUND",
    "PROJECTS",
    "CERTIFICATIONS",
    "ACHIEVEMENTS",
}


class ResumeRAGPipeline:
    """Manages the complete document processing pipeline for resumes."""

    def __init__(
        self,
        resumes_dir: str = RESUMES_DIR,
        chroma_db_path: str = CHROMA_DB_PATH,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
    ):
        self.resumes_dir = resumes_dir
        self.chroma_db_path = chroma_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = OpenAIEmbeddings(model=embedding_model) if OPENAI_API_KEY else None
        self.llm = ChatOpenAI(model=llm_model, temperature=0) if OPENAI_API_KEY else None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
        )

        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = None

    def _require_embeddings(self) -> OpenAIEmbeddings:
        if self.embeddings is None:
            raise ValueError("OPENAI_API_KEY is required to generate embeddings")
        return self.embeddings

    def _get_or_create_collection(self) -> chromadb.Collection:
        try:
            self.collection = self.chroma_client.get_collection("resumes")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name="resumes",
                metadata={"hnsw:space": "cosine"},
            )
        return self.collection

    @staticmethod
    def _clean_whitespace(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _unique_preserve_order(items: Sequence[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized:
                continue
            lower_value = normalized.lower()
            if lower_value in seen:
                continue
            seen.add(lower_value)
            result.append(normalized)
        return result

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        cleaned = re.sub(r"[:\-]+$", "", line.strip())
        normalized = re.sub(r"[^A-Z0-9 &/().,-]", "", cleaned.upper()).strip()
        return normalized in SECTION_HEADERS

    def _split_into_sections(self, resume_text: str) -> List[Tuple[str, str]]:
        lines = resume_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        sections: List[Tuple[str, List[str]]] = []
        current_section = "FULL_TEXT"
        current_lines: List[str] = []
        heading_found = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                current_lines.append("")
                continue

            if self._looks_like_heading(line):
                heading_found = True
                if any(part.strip() for part in current_lines):
                    sections.append((current_section, current_lines))
                current_section = re.sub(r"[:\-]+$", "", line.strip()).upper()
                current_lines = []
                continue

            current_lines.append(raw_line.rstrip())

        if any(part.strip() for part in current_lines):
            sections.append((current_section, current_lines))

        if not heading_found:
            return [("FULL_TEXT", resume_text.strip())]

        return [
            (section_name, "\n".join(lines).strip())
            for section_name, lines in sections
            if "\n".join(lines).strip()
        ]

    def _chunk_section(self, section_name: str, section_text: str) -> List[str]:
        if not section_text.strip():
            return []

        chunks = self.text_splitter.split_text(section_text.strip())
        if not chunks:
            chunks = [section_text.strip()]

        if section_name == "FULL_TEXT":
            return [self._clean_whitespace(chunk) for chunk in chunks if chunk.strip()]

        return [
            f"{section_name}\n{self._clean_whitespace(chunk)}"
            for chunk in chunks
            if chunk.strip()
        ]

    def _build_section_chunks(self, resume_text: str) -> List[Dict[str, str]]:
        section_chunks: List[Dict[str, str]] = []

        for section_name, section_text in self._split_into_sections(resume_text):
            for chunk_index, chunk_text in enumerate(self._chunk_section(section_name, section_text)):
                section_chunks.append(
                    {
                        "section": section_name,
                        "chunk_index": str(chunk_index),
                        "text": chunk_text,
                    }
                )

        return section_chunks

    @staticmethod
    def _extract_first_matching_section(
        sections: List[Tuple[str, str]],
        target_names: Sequence[str],
    ) -> str:
        target_lookup = {name.upper() for name in target_names}
        for section_name, section_text in sections:
            if section_name.upper() in target_lookup:
                return section_text.strip()
        return ""

    @staticmethod
    def _extract_experience_years(resume_text: str) -> int:
        patterns = [
            r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience)?",
            r"experience\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, resume_text, flags=re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (TypeError, ValueError):
                    continue
        return 0

    @staticmethod
    def _extract_candidate_name(resume_text: str, filename: str) -> str:
        lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
        for line in lines[:10]:
            if "@" in line or re.search(r"\b(?:\+?\d[\d\-\s]{7,})\b", line):
                continue
            if len(line.split()) > 6:
                continue
            if line.upper() in SECTION_HEADERS:
                continue
            return line

        stem = Path(filename).stem.replace("_", " ")
        stem = re.sub(r"^resume\s*\d+\s*", "", stem, flags=re.IGNORECASE)
        return stem.strip().title() or Path(filename).stem

    def _extract_skills(self, sections: List[Tuple[str, str]], resume_text: str) -> List[str]:
        skills_section = self._extract_first_matching_section(
            sections,
            ["SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES"],
        )
        if not skills_section:
            skills_section = resume_text

        raw_items = re.split(r"[\n,;/•|]+", skills_section)
        cleaned_skills = []
        for item in raw_items:
            skill = self._clean_whitespace(item)
            if not skill:
                continue
            if len(skill) < 2:
                continue
            if skill.upper() in SECTION_HEADERS:
                continue
            cleaned_skills.append(skill)

        return self._unique_preserve_order(cleaned_skills)

    def _fallback_metadata(self, resume_text: str, filename: str) -> Dict[str, object]:
        sections = self._split_into_sections(resume_text)
        candidate_name = self._extract_candidate_name(resume_text, filename)
        skills = self._extract_skills(sections, resume_text)
        education = self._extract_first_matching_section(
            sections,
            ["EDUCATION", "ACADEMIC BACKGROUND"],
        )

        return {
            "candidate_name": candidate_name,
            "skills": skills,
            "years_of_experience": self._extract_experience_years(resume_text),
            "education": self._clean_whitespace(education),
        }

    def _normalize_metadata(self, metadata: Dict[str, object], fallback: Dict[str, object]) -> Dict[str, object]:
        normalized = dict(fallback)

        candidate_name = str(metadata.get("candidate_name", "")).strip()
        if candidate_name:
            normalized["candidate_name"] = candidate_name

        raw_skills = metadata.get("skills", [])
        if isinstance(raw_skills, str):
            skills = [skill.strip() for skill in raw_skills.split(",") if skill.strip()]
        elif isinstance(raw_skills, list):
            skills = [str(skill).strip() for skill in raw_skills if str(skill).strip()]
        else:
            skills = []
        if skills:
            normalized["skills"] = self._unique_preserve_order(skills)

        years_of_experience = metadata.get("years_of_experience", normalized.get("years_of_experience", 0))
        try:
            normalized["years_of_experience"] = int(float(years_of_experience))
        except (TypeError, ValueError):
            normalized["years_of_experience"] = int(normalized.get("years_of_experience", 0) or 0)

        education = str(metadata.get("education", "")).strip()
        if education:
            normalized["education"] = education

        return normalized

    def _extract_metadata(self, resume_text: str, filename: str) -> Dict[str, object]:
        """Extract metadata from resume text with an LLM-backed primary path and a heuristic fallback."""
        fallback = self._fallback_metadata(resume_text, filename)
        if self.llm is None:
            return fallback

        extraction_prompt = f"""
Extract the following information from this resume and return ONLY a valid JSON object:
{{
  "candidate_name": "Full name of the candidate",
  "skills": ["skill1", "skill2", "skill3"],
  "years_of_experience": 0,
  "education": "Highest degree and field"
}}

Resume text:
{resume_text[:3000]}

Return ONLY the JSON object, no additional text.
""".strip()

        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            response_text = str(response.content).strip()

            if response_text.startswith("```"):
                response_text = response_text.strip("`")

            parsed_metadata = json.loads(response_text)
            if not isinstance(parsed_metadata, dict):
                raise ValueError("LLM metadata response was not a JSON object")

            return self._normalize_metadata(parsed_metadata, fallback)

        except Exception as exc:
            logger.warning("Metadata extraction fallback used for %s: %s", filename, exc)
            return fallback

    def _load_documents(self) -> List[object]:
        """Load all PDF documents from the resumes directory."""
        if not Path(self.resumes_dir).exists():
            raise FileNotFoundError(f"Resumes directory not found: {self.resumes_dir}")

        logger.info("Loading PDFs from %s", self.resumes_dir)
        loader = PyPDFDirectoryLoader(self.resumes_dir)

        documents = loader.load()
        logger.info("Successfully loaded %d documents", len(documents))
        return documents

    def process_resumes(self) -> None:
        """Process all resumes, chunk them by section, embed, and persist them in ChromaDB."""
        logger.info("Starting resume processing pipeline...")

        self._get_or_create_collection()
        documents = self._load_documents()

        if not documents:
            logger.warning("No documents found to process")
            return

        processed_count = 0
        failed_count = 0

        for idx, doc in enumerate(documents, 1):
            filename = str(doc.metadata.get("source", "unknown"))

            try:
                logger.info("[%d/%d] Processing %s", idx, len(documents), filename)

                resume_text = str(doc.page_content or "").strip()
                if not resume_text:
                    logger.warning("Skipping %s because the resume text is empty", filename)
                    failed_count += 1
                    continue

                metadata = self._extract_metadata(resume_text, filename)
                chunk_records = self._build_section_chunks(resume_text)

                if not chunk_records:
                    logger.warning("No chunks generated for %s", filename)
                    failed_count += 1
                    continue

                logger.info("  Generated %d section-aware chunks", len(chunk_records))

                chunk_texts = [record["text"] for record in chunk_records]
                embeddings = self._require_embeddings()
                chunk_embeddings = embeddings.embed_documents(chunk_texts)

                payloads = []
                for chunk_idx, (record, chunk_embedding) in enumerate(zip(chunk_records, chunk_embeddings)):
                    doc_id = f"{Path(filename).stem}_{record['section'].replace(' ', '_').lower()}_{chunk_idx}"
                    chunk_metadata = {
                        "source": filename,
                        "chunk_index": chunk_idx,
                        "section": record["section"],
                        "candidate_name": metadata["candidate_name"],
                        "years_of_experience": int(metadata["years_of_experience"]),
                        "education": metadata["education"],
                        "skills": ", ".join(metadata["skills"]),
                    }

                    payloads.append(
                        {
                            "id": doc_id,
                            "document": record["text"],
                            "embedding": chunk_embedding,
                            "metadata": chunk_metadata,
                        }
                    )

                self.collection.add(
                    ids=[item["id"] for item in payloads],
                    documents=[item["document"] for item in payloads],
                    embeddings=[item["embedding"] for item in payloads],
                    metadatas=[item["metadata"] for item in payloads],
                )

                processed_count += 1

            except Exception as exc:
                logger.error("Unexpected error processing %s: %s", filename, exc)
                failed_count += 1

        logger.info("=" * 60)
        logger.info("Processing Pipeline Complete")
        logger.info("Successfully processed: %d resumes", processed_count)
        logger.info("Failed: %d resumes", failed_count)
        logger.info("ChromaDB collection: %s", self.collection.name)
        logger.info(f"Persistent storage: {os.path.abspath(self.chroma_db_path)}")
        logger.info("=" * 60)

    def query_resumes(self, query_text: str, n_results: int = 5) -> list:
        """
        Search for similar resumes by query text.
        """
        if self.collection is None:
            self._get_or_create_collection()

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

        return results


def main():
    """Main execution function."""
    pipeline = ResumeRAGPipeline()
    pipeline.process_resumes()


if __name__ == "__main__":
    main()