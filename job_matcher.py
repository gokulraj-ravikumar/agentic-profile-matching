"""Score and rank resumes against job descriptions using semantic search and keyword matching."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_CHUNKS = 30
DEFAULT_TOP_MATCHES = 10

STOPWORDS = {
	"and",
	"the",
	"for",
	"with",
	"that",
	"from",
	"this",
	"will",
	"you",
	"your",
	"are",
	"have",
	"has",
	"our",
	"role",
	"team",
	"work",
	"experience",
	"years",
	"year",
	"responsibilities",
	"requirements",
	"required",
	"skills",
	"skill",
	"ability",
	"knowledge",
	"preferred",
	"must",
	"have",
	"in",
	"on",
	"to",
	"of",
	"a",
	"an",
	"be",
	"as",
	"is",
	"by",
	"or",
	"at",
	"into",
}


class JobMatcherEngine:
	"""Matches resumes against job descriptions."""

	def __init__(
		self,
		chroma_db_path: str = CHROMA_DB_PATH,
		embedding_model: str = EMBEDDING_MODEL,
	):
		self.chroma_db_path = chroma_db_path
		self.embeddings = OpenAIEmbeddings(model=embedding_model) if OPENAI_API_KEY else None
		self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
		self.collection = self.chroma_client.get_or_create_collection(
			name="resumes",
			metadata={"hnsw:space": "cosine"},
		)

	def _require_embeddings(self) -> OpenAIEmbeddings:
		if self.embeddings is None:
			raise ValueError("OPENAI_API_KEY is required to generate query embeddings")
		return self.embeddings

	def _rebuild_resume_index(self) -> None:
		logger.warning("Rebuilding the resume index from source PDFs in %s", Path("./data/resumes").resolve())

		try:
			self.chroma_client.delete_collection("resumes")
		except Exception:
			pass

		from resume_rag import ResumeRAGPipeline

		pipeline = ResumeRAGPipeline(chroma_db_path=self.chroma_db_path)
		pipeline.process_resumes()
		self.collection = self.chroma_client.get_or_create_collection(
			name="resumes",
			metadata={"hnsw:space": "cosine"},
		)

	@staticmethod
	def _normalize_text(value: str) -> str:
		return re.sub(r"\s+", " ", value).strip().lower()

	@staticmethod
	def _normalize_skill(value: str) -> str:
		return re.sub(r"\s+", " ", value).strip().lower()

	@staticmethod
	def _parse_skills(metadata: Dict[str, Any]) -> List[str]:
		raw_skills = metadata.get("skills", "")
		if isinstance(raw_skills, list):
			return [str(skill).strip() for skill in raw_skills if str(skill).strip()]
		if isinstance(raw_skills, str):
			return [skill.strip() for skill in raw_skills.split(",") if skill.strip()]
		return []

	@staticmethod
	def _parse_experience(metadata: Dict[str, Any]) -> int:
		value = metadata.get("years_of_experience", 0)
		try:
			return int(float(value))
		except (TypeError, ValueError):
			return 0

	@staticmethod
	def _split_lines(text: str) -> List[str]:
		return [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]

	def _extract_requirements_from_job_description(
		self,
		job_description: str,
		must_have_skills: Optional[Sequence[str]] = None,
		minimum_experience: Optional[int] = None,
	) -> Tuple[List[str], int]:
		skills: List[str] = [skill.strip() for skill in must_have_skills or [] if str(skill).strip()]

		if not skills:
			for line in self._split_lines(job_description):
				match = re.search(
					r"(?:required skills?|must-have skills?|must have skills?|skills required|required|must have)\s*[:\-]\s*(.+)",
					line,
					flags=re.IGNORECASE,
				)
				if match:
					raw_skills = re.split(r"[,/;•|]+", match.group(1))
					skills = [self._normalize_text(skill) for skill in raw_skills if self._normalize_text(skill)]
					break

		inferred_experience = minimum_experience if minimum_experience is not None else 0
		if inferred_experience == 0:
			experience_matches = [
				int(match.group(1))
				for match in re.finditer(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience)?", job_description, flags=re.IGNORECASE)
			]
			if experience_matches:
				inferred_experience = max(experience_matches)

		return self._dedupe_normalized_skills(skills), inferred_experience

	@staticmethod
	def _dedupe_normalized_skills(skills: Sequence[str]) -> List[str]:
		seen = set()
		cleaned: List[str] = []
		for skill in skills:
			normalized = re.sub(r"\s+", " ", str(skill)).strip()
			if not normalized:
				continue
			key = normalized.lower()
			if key in seen:
				continue
			seen.add(key)
			cleaned.append(normalized)
		return cleaned

	@staticmethod
	def _tokenize(text: str) -> List[str]:
		tokens = []
		for raw_token in re.findall(r"[A-Za-z0-9+#./-]+", text):
			token = raw_token.lower().strip(".-/")
			if len(token) < 3:
				continue
			if token in STOPWORDS:
				continue
			tokens.append(token)
		return tokens

	def _search_top_chunks(self, job_description: str, k: int = TOP_K_CHUNKS) -> Dict[str, List[Any]]:
		embeddings = self._require_embeddings()
		query_embedding = embeddings.embed_query(job_description)
		try:
			return self.collection.query(
				query_embeddings=[query_embedding],
				n_results=k,
				include=["documents", "metadatas", "distances"],
			)
		except Exception as exc:
			message = str(exc)
			if "Nothing found on disk" in message or "HNSW segment reader" in message:
				logger.warning("Detected a broken Chroma index: %s", exc)
				self._rebuild_resume_index()
				return self.collection.query(
					query_embeddings=[query_embedding],
					n_results=k,
					include=["documents", "metadatas", "distances"],
				)
			raise

	@staticmethod
	def _group_chunks_by_candidate(query_results: Dict[str, List[Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
		grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

		documents = query_results.get("documents", [[]])[0]
		metadatas = query_results.get("metadatas", [[]])[0]
		distances = query_results.get("distances", [[]])[0]
		ids = query_results.get("ids", [[]])[0] if query_results.get("ids") else []

		for index, document in enumerate(documents):
			metadata = metadatas[index] if index < len(metadatas) else {}
			chunk_id = ids[index] if index < len(ids) else f"chunk_{index}"
			distance = distances[index] if index < len(distances) else 1.0

			candidate_name = str(metadata.get("candidate_name", "Unknown Candidate")).strip() or "Unknown Candidate"
			resume_path = str(metadata.get("source", "unknown")).strip() or "unknown"
			candidate_key = (candidate_name, resume_path)

			if candidate_key not in grouped:
				grouped[candidate_key] = {
					"candidate_name": candidate_name,
					"resume_path": resume_path,
					"metadata": metadata,
					"chunks": [],
				}

			grouped[candidate_key]["chunks"].append(
				{
					"chunk_id": chunk_id,
					"text": document,
					"metadata": metadata,
					"distance": float(distance),
				}
			)

		return grouped

	def _candidate_meets_requirements(
		self,
		metadata: Dict[str, Any],
		must_have_skills: Sequence[str],
		minimum_experience: int,
		candidate_text: str,
	) -> bool:
		candidate_experience = self._parse_experience(metadata)
		candidate_skills = {
			self._normalize_skill(skill)
			for skill in self._parse_skills(metadata)
		}
		required_skills = {self._normalize_skill(skill) for skill in must_have_skills}

		if candidate_experience < minimum_experience:
			return False

		for required_skill in required_skills:
			if required_skill in candidate_skills:
				continue
			if required_skill and required_skill not in candidate_text.lower():
				return False

		return True

	@staticmethod
	def _semantic_similarity_score(distance: float) -> float:
		similarity = 1.0 - (distance / 2.0)
		"""Convert distance to similarity score (0-100)."""
		return max(0.0, min(1.0, similarity)) * 100.0

	def _keyword_match_score(
		self,
		job_description: str,
		candidate_record: Dict[str, Any],
		must_have_skills: Sequence[str],
	) -> Tuple[float, List[str], List[str]]:
		candidate_chunks = candidate_record["chunks"]
		candidate_text = " \n".join(str(chunk["text"]) for chunk in candidate_chunks).lower()
		candidate_skills = self._dedupe_normalized_skills(self._parse_skills(candidate_record["metadata"]))
		candidate_skill_lookup = {skill.lower(): skill for skill in candidate_skills}

		matched_skills: List[str] = []
		must_have_hits = 0
		for required_skill in must_have_skills:
			normalized_skill = self._normalize_skill(required_skill)
			if not normalized_skill:
				continue
			if normalized_skill in candidate_skill_lookup:
				matched_skills.append(candidate_skill_lookup[normalized_skill])
				must_have_hits += 1
			elif normalized_skill in candidate_text:
				matched_skills.append(required_skill)
				must_have_hits += 1

		job_tokens = self._tokenize(job_description)
		if job_tokens:
			overlap = sum(1 for token in set(job_tokens) if token in candidate_text)
			token_score = (overlap / len(set(job_tokens))) * 100.0
		else:
			token_score = 0.0

		if must_have_skills:
			must_have_score = (must_have_hits / len(must_have_skills)) * 100.0
		else:
			must_have_score = 0.0

		keyword_score = (0.7 * must_have_score) + (0.3 * token_score)

		return keyword_score, self._dedupe_normalized_skills(matched_skills), candidate_skills

	@staticmethod
	def _extract_relevant_excerpts(candidate_record: Dict[str, Any], matched_skills: Sequence[str]) -> List[str]:
		excerpts: List[str] = []
		for chunk in candidate_record["chunks"]:
			chunk_text = re.sub(r"\s+", " ", str(chunk["text"]).strip())
			chunk_section = str(chunk["metadata"].get("section", "resume text")).strip()
			lowered_chunk = chunk_text.lower()

			if matched_skills and not any(skill.lower() in lowered_chunk for skill in matched_skills):
				continue

			snippet = chunk_text[:220]
			excerpts.append(f"[{chunk_section}] {snippet}")
			if len(excerpts) >= 3:
				break

		if not excerpts:
			for chunk in candidate_record["chunks"][:3]:
				chunk_text = re.sub(r"\s+", " ", str(chunk["text"]).strip())
				chunk_section = str(chunk["metadata"].get("section", "resume text")).strip()
				excerpts.append(f"[{chunk_section}] {chunk_text[:220]}")

		return excerpts

	def _score_candidate(
		self,
		job_description: str,
		must_have_skills: Sequence[str],
		minimum_experience: int,
		candidate_record: Dict[str, Any],
	) -> Dict[str, Any]:
		candidate_text = " \n".join(str(chunk["text"]) for chunk in candidate_record["chunks"])
		semantic_score = 0.0
		if candidate_record["chunks"]:
			semantic_score = max(
				self._semantic_similarity_score(chunk["distance"])
				for chunk in candidate_record["chunks"]
			)

		keyword_score, matched_skills, candidate_skills = self._keyword_match_score(
			job_description=job_description,
			candidate_record=candidate_record,
			must_have_skills=must_have_skills,
		)

		candidate_experience = self._parse_experience(candidate_record["metadata"])
		if minimum_experience > 0:
			experience_score = min(100.0, (candidate_experience / minimum_experience) * 100.0) if candidate_experience else 0.0
		else:
			experience_score = min(100.0, candidate_experience * 10.0)

		section_names = []
		if matched_skills:
			for chunk in candidate_record["chunks"]:
				chunk_text = str(chunk["text"]).lower()
				if any(skill.lower() in chunk_text for skill in matched_skills):
					section_names.append(str(chunk["metadata"].get("section", "resume text")).strip())
		if not section_names:
			section_names = [
				str(chunk["metadata"].get("section", "resume text")).strip()
				for chunk in candidate_record["chunks"][:3]
			]
		matched_sections = self._dedupe_normalized_skills(section_names)

		match_score = round(
			max(
				0.0,
				min(
					100.0,
					(0.55 * semantic_score) + (0.35 * keyword_score) + (0.10 * experience_score),
				),
			)
		)

		relevant_excerpts = self._extract_relevant_excerpts(candidate_record, matched_skills)
		if matched_skills:
			reasoning_skills = ", ".join(matched_skills[:4])
		elif candidate_skills:
			reasoning_skills = ", ".join(candidate_skills[:4])
		else:
			reasoning_skills = "relevant resume language"

		reasoning = (
			f"Matched {reasoning_skills} across {', '.join(matched_sections[:3]) or 'resume sections'}. "
			f"Semantic similarity and keyword overlap both support the ranking; "
			f"candidate experience is {candidate_experience} years versus the {minimum_experience}-year requirement."
		)

		return {
			"candidate_name": candidate_record["candidate_name"],
			"resume_path": candidate_record["resume_path"],
			"match_score": int(match_score),
			"matched_skills": matched_skills,
			"relevant_excerpts": relevant_excerpts,
			"reasoning": reasoning,
			"years_of_experience": candidate_experience,
			"all_skills": list(candidate_skills),
		}

	def match_jobs(
		self,
		job_description: str,
		must_have_skills: Optional[Sequence[str]] = None,
		minimum_experience: Optional[int] = None,
		top_n: int = DEFAULT_TOP_MATCHES,
	) -> Dict[str, Any]:
		"""Return top candidates ranked by match score."""
		inferred_skills, inferred_experience = self._extract_requirements_from_job_description(
			job_description=job_description,
			must_have_skills=must_have_skills,
			minimum_experience=minimum_experience,
		)

		try:
			query_results = self._search_top_chunks(job_description, k=TOP_K_CHUNKS)
		except Exception as e:
			if "Nothing found on disk" in str(e) or "HNSW" in str(e):
				logger.error(f"Chroma index corrupted: {e}. Rebuilding from source resumes...")
				self._rebuild_resume_index()
				query_results = self._search_top_chunks(job_description, k=TOP_K_CHUNKS)
			else:
				raise
		grouped_candidates = self._group_chunks_by_candidate(query_results)

		candidate_matches: List[Dict[str, Any]] = []

		for candidate_key, candidate_record in grouped_candidates.items():
			metadata = candidate_record["metadata"]
			candidate_text = " \n".join(str(chunk["text"]) for chunk in candidate_record["chunks"])

			if not self._candidate_meets_requirements(metadata, inferred_skills, inferred_experience, candidate_text):
				logger.info("Filtered out %s due to hard requirements", candidate_key[0])
				continue

			try:
				scored_candidate = self._score_candidate(
					job_description=job_description,
					must_have_skills=inferred_skills,
					minimum_experience=inferred_experience,
					candidate_record=candidate_record,
				)
				candidate_matches.append(scored_candidate)
			except Exception as exc:
				logger.error("Failed to score %s: %s", candidate_key[0], exc)

		candidate_matches.sort(key=lambda item: item.get("match_score", 0), reverse=True)

		return {
			"job_description": job_description,
			"top_matches": candidate_matches[:top_n],
		}


def match_job_description(
	job_description: str,
	must_have_skills: Optional[Sequence[str]] = None,
	minimum_experience: Optional[int] = None,
	top_n: int = DEFAULT_TOP_MATCHES,
) -> Dict[str, Any]:
	"""Convenience function for matching a single job description."""
	engine = JobMatcherEngine()
	return engine.match_jobs(
		job_description=job_description,
		must_have_skills=must_have_skills,
		minimum_experience=minimum_experience,
		top_n=top_n,
	)


if __name__ == "__main__":
	raise SystemExit(
		"Import job_matcher.py and call match_job_description(job_description, must_have_skills=None, minimum_experience=None)"
	)
