from __future__ import annotations

import operator
import re
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from fs_tools import list_files, read_file as fs_read_file, search_in_file, write_file
from job_matcher import match_job_description

load_dotenv()

FILE_SYSTEM_TOOLS = {
    "list_files": list_files,
    "read_file": fs_read_file,
    "search_in_file": search_in_file,
    "write_file": write_file,
}

SKILL_STOPWORDS = {
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
    "find",
    "me",
    "candidates",
    "candidate",
    "top",
    "matches",
    "match",
    "side",
    "compare",
    "comparison",
    "why",
    "rank",
    "higher",
    "than",
    "please",
    "show",
    "give",
    "need",
    "looking",
    "want",
    "adjust",
    "change",
    "instead",
    "more",
    "less",
    "focus",
    "on",
}


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    job_description: str
    job_requirements: Dict[str, Any]
    candidate_shortlist: List[Dict[str, Any]]
    previous_candidate_shortlist: List[Dict[str, Any]]
    ranking_reasoning: str
    comparison_report: str
    interview_questions: Dict[str, List[str]]
    final_recommendation: str
    screening_round: int
    feedback: str
    intent: str
    target_candidates: List[str]
    follow_up_prompt: str
    final_report: str


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        value = re.sub(r"\s+", " ", str(item)).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _split_candidate_text(text: str) -> List[str]:
    cleaned = re.sub(r"\b(?:and|with|plus|including|including)\b", ",", text, flags=re.IGNORECASE)
    parts = re.split(r"[,/;•|&]+", cleaned)
    skills: List[str] = []
    for part in parts:
        candidate = re.sub(r"\b\d+\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience)?\b", "", part, flags=re.IGNORECASE)
        candidate = re.sub(r"\b(?:candidates?|candidate|jobs?|job|role|experience|years?|yrs?)\b", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .,:-\t")
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in SKILL_STOPWORDS:
            continue
        if len(candidate) < 2:
            continue
        skills.append(candidate)
    return _dedupe_preserve_order(skills)


def _extract_years(text: str) -> int:
    matches = [int(match.group(1)) for match in re.finditer(r"(\d+)\+?\s*(?:years?|yrs?)", text, flags=re.IGNORECASE)]
    return max(matches) if matches else 0


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _load_text_if_path(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return value

    path = Path(candidate.strip('"'))
    if path.exists() and path.is_file():
        loaded = fs_read_file(str(path))
        if loaded.get("status") == "success" and loaded.get("content"):
            return str(loaded["content"])
    return value


def _classify_intent(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["compare", "side by side", "versus", "vs"]):
        return "compare"
    if any(keyword in lowered for keyword in ["why did", "why is", "rank higher", "ranked higher", "explain"]):
        return "explain"
    if any(keyword in lowered for keyword in ["question", "screening", "interview"]):
        return "questions"
    if any(keyword in lowered for keyword in ["adjust", "change", "instead", "remove", "add", "focus more", "focus less", "refine"]):
        return "refine"
    return "search"


def _extract_candidate_targets(text: str) -> List[str]:
    lowered = text.lower()
    targets: List[str] = []

    compare_match = re.search(r"why did\s+(.+?)\s+rank higher than\s+(.+?)(?:[?.!]|$)", text, flags=re.IGNORECASE)
    if compare_match:
        targets.extend([compare_match.group(1).strip(), compare_match.group(2).strip()])
        return _dedupe_preserve_order(targets)

    pair_match = re.search(r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:[?.!]|$)", text, flags=re.IGNORECASE)
    if pair_match:
        targets.extend([pair_match.group(1).strip(), pair_match.group(2).strip()])
        return _dedupe_preserve_order(targets)

    if "top 3" in lowered or "top three" in lowered:
        return []

    return []


def _parse_requirement_text(text: str) -> Dict[str, Any]:
    normalized_text = re.sub(r"\s+", " ", text).strip()
    minimum_experience = _extract_years(normalized_text)

    raw_must_have = ""
    raw_nice_to_have = ""
    for pattern in [r"(?:must have|required|need|looking for|with)\s*[:\-]?\s*(.+)", r"skills?\s*[:\-]?\s*(.+)"]:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
        if match:
            raw_must_have = match.group(1)
            break

    for pattern in [r"(?:nice to have|preferred|bonus)\s*[:\-]?\s*(.+)"]:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
        if match:
            raw_nice_to_have = match.group(1)
            break

    if not raw_must_have:
        after_with = re.search(r"\bwith\b\s*(.+)", normalized_text, flags=re.IGNORECASE)
        if after_with:
            raw_must_have = after_with.group(1)

    cleaned_must_have = re.sub(r"\b\d+\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience)?\b", "", raw_must_have, flags=re.IGNORECASE)
    cleaned_nice_to_have = re.sub(r"\b\d+\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience)?\b", "", raw_nice_to_have, flags=re.IGNORECASE)

    must_have = _split_candidate_text(cleaned_must_have)
    nice_to_have = _split_candidate_text(cleaned_nice_to_have)

    # Sanitize extracted skill strings - remove labels or accidental fragments
    def _sanitize_list(items: List[str]) -> List[str]:
        cleaned: List[str] = []
        # map common tokens to canonical forms
        canonical = {
            "python": "Python",
            "react": "React",
            "airflow": "Airflow",
            "aws": "AWS",
            "docker": "Docker",
            "kubernetes": "Kubernetes",
            "java": "Java",
            "scala": "Scala",
            "spark": "Spark",
        }

        for item in items:
            s = re.sub(r"(?i)must[- ]?have:?|nice[- ]?to[- ]?have:?|minimum[: ]*|original request:?|request:?", "", item)
            s = re.sub(r"\bminimum experience\b|\boriginal request\b", "", s, flags=re.IGNORECASE)
            s = s.strip(" .,:-\t")
            if not s:
                continue
            lowered = s.lower()
            # if the phrase mentions a known token, normalize to canonical
            for token, pretty in canonical.items():
                if token in lowered:
                    s = pretty
                    break
            # remove filler phrases
            s = re.sub(r"\b(the person who also have|the person who also has|who also have|who also has|also have|also has)\b", "", s, flags=re.IGNORECASE)
            s = s.strip(" .,:-\t")
            if not s:
                continue
            if len(s) < 2:
                continue
            cleaned.append(s)
        return _dedupe_preserve_order(cleaned)

    must_have = _sanitize_list(must_have)
    nice_to_have = _sanitize_list(nice_to_have)

    if not must_have:
        fallback_skills = _split_candidate_text(normalized_text)
        # filter out obvious questions, compare requests or long conversational fragments
        filtered = []
        bad_tokens = {"why", "how", "what", "who", "which", "compare", "top", "rank", "adjust"}
        for item in fallback_skills:
            lowered = item.lower()
            if any(tok in lowered for tok in bad_tokens):
                continue
            if "?" in item:
                continue
            if len(item) > 40:
                continue
            filtered.append(item)

        must_have = filtered[:4]

    role = ""
    if normalized_text:
        first_line = normalized_text.split("\n", 1)[0].strip()
        # avoid capturing full questions or conversational prompts as the role
        interrogatives = {"why", "how", "what", "who", "which", "when", "where", "compare", "top", "rank"}
        lowered_first = first_line.lower()
        if len(first_line) <= 120 and not any(word in lowered_first for word in interrogatives) and "?" not in first_line:
            role = first_line

    return {
        "role": role,
        "must_have": must_have,
        "nice_to_have": nice_to_have,
        "minimum_experience": minimum_experience,
        "raw_text": text,
    }


def _merge_requirements(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})
    merged_must_have = _dedupe_preserve_order(list(merged.get("must_have", [])) + list(incoming.get("must_have", [])))
    merged_nice_to_have = _dedupe_preserve_order(list(merged.get("nice_to_have", [])) + list(incoming.get("nice_to_have", [])))

    merged["must_have"] = merged_must_have
    merged["nice_to_have"] = merged_nice_to_have
    merged["role"] = incoming.get("role") or merged.get("role", "")
    merged["raw_text"] = incoming.get("raw_text") or merged.get("raw_text", "")

    incoming_experience = int(incoming.get("minimum_experience", 0) or 0)
    existing_experience = int(merged.get("minimum_experience", 0) or 0)
    merged["minimum_experience"] = incoming_experience or existing_experience

    return merged


def _compose_job_description(requirements: Dict[str, Any], fallback_text: str) -> str:
    parts = []
    role = str(requirements.get("role", "")).strip()
    must_have = requirements.get("must_have", []) or []
    nice_to_have = requirements.get("nice_to_have", []) or []
    minimum_experience = int(requirements.get("minimum_experience", 0) or 0)

    if role:
        parts.append(f"Role: {role}")
    if must_have:
        parts.append(f"Must-have: {', '.join(must_have)}")
    if nice_to_have:
        parts.append(f"Nice-to-have: {', '.join(nice_to_have)}")
    if minimum_experience:
        parts.append(f"Minimum experience: {minimum_experience}+ years")
    if fallback_text and fallback_text not in parts:
        parts.append(f"Original request: {fallback_text.strip()}")

    return "\n".join(parts).strip() or fallback_text.strip()


def extract_requirements(jd: str) -> Dict[str, Any]:
    return _parse_requirement_text(jd)


def _candidate_lookup(candidate_pool: Sequence[Dict[str, Any]], candidate_id: str) -> Optional[Dict[str, Any]]:
    normalized_target = _normalize_name(candidate_id)
    for candidate in candidate_pool:
        candidate_name = str(candidate.get("candidate_name", "")).strip()
        resume_path = str(candidate.get("resume_path", "")).strip()
        if normalized_target == _normalize_name(candidate_name):
            return candidate
        if normalized_target and normalized_target in _normalize_name(candidate_name):
            return candidate
        if normalized_target and normalized_target in _normalize_name(resume_path):
            return candidate
    return None


def _candidate_gaps(candidate: Dict[str, Any], requirements: Dict[str, Any]) -> List[str]:
    required_skills = [skill for skill in requirements.get("must_have", []) if skill]
    matched_skills = {skill.lower() for skill in candidate.get("matched_skills", [])}
    gaps = [skill for skill in required_skills if skill.lower() not in matched_skills]
    minimum_experience = int(requirements.get("minimum_experience", 0) or 0)
    candidate_experience = int(candidate.get("years_of_experience", candidate.get("experience_years", 0)) or 0)
    if minimum_experience and candidate_experience < minimum_experience:
        gaps.append(f"Needs {minimum_experience - candidate_experience} more years of experience")
    return gaps


def _candidate_experience(candidate: Dict[str, Any]) -> int:
    return int(candidate.get("years_of_experience", candidate.get("experience_years", 0)) or 0)


def _candidate_table_row(candidate: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    matched_skills = candidate.get("matched_skills", []) or []
    gaps = _candidate_gaps(candidate, requirements)
    return {
        "candidate_name": candidate.get("candidate_name", "Unknown Candidate"),
        "match_score": int(candidate.get("match_score", 0) or 0),
        "experience": _candidate_experience(candidate),
        "matched_skills": matched_skills,
        "gaps": gaps,
        "reasoning": candidate.get("reasoning", ""),
    }


def compare_candidates(
    candidate_ids: List[str],
    candidate_pool: Optional[List[Dict[str, Any]]] = None,
    requirements: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pool = candidate_pool or []
    selected: List[Dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate = _candidate_lookup(pool, candidate_id)
        if candidate:
            selected.append(candidate)

    if not selected and pool:
        selected = list(pool[: len(candidate_ids) or 2])

    if not selected:
        return {
            "summary": "No matching candidates were available to compare.",
            "rows": [],
            "winner": "",
        }

    requirements = requirements or {}
    rows = [_candidate_table_row(candidate, requirements) for candidate in selected]
    winner = max(rows, key=lambda row: row.get("match_score", 0))
    summary_lines = [
        "Side-by-side comparison:",
        *[
            f"- {row['candidate_name']}: {row['match_score']} score, {row['experience']} years, matched {', '.join(row['matched_skills'][:4]) or 'no explicit must-haves'}"
            for row in rows
        ],
        f"Top fit right now: {winner['candidate_name']}.",
    ]

    return {
        "summary": "\n".join(summary_lines),
        "rows": rows,
        "winner": winner["candidate_name"],
    }


def generate_interview_questions(
    candidate_id: str,
    candidate_record: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
) -> List[str]:
    candidate = candidate_record or {}
    requirements = requirements or {}
    gaps = _candidate_gaps(candidate, requirements)
    candidate_name = candidate.get("candidate_name", candidate_id)

    questions: List[str] = []
    if gaps:
        for gap in gaps[:3]:
            if gap.lower().startswith("needs"):
                questions.append(f"{candidate_name}: Can you walk us through a recent project that demonstrates the experience needed here?")
            else:
                questions.append(f"{candidate_name}: How have you used {gap} in production, and what trade-offs did you manage?")
    else:
        questions.extend(
            [
                f"{candidate_name}: Which part of your background is most relevant to this role?",
                f"{candidate_name}: What would you improve in your last project if you had one more week?",
                f"{candidate_name}: How do you stay current with changes in the tools or stack used here?",
            ]
        )

    if len(questions) < 3:
        questions.extend(
            [
                f"{candidate_name}: Describe a difficult problem you solved end-to-end.",
                f"{candidate_name}: Tell us about a time you had to learn a new tool quickly.",
            ]
        )

    return _dedupe_preserve_order(questions)[:5]


def _parse_names_from_query(query: str, shortlist: Sequence[Dict[str, Any]]) -> List[str]:
    lowered = query.lower()
    names: List[str] = []
    if "top 3" in lowered or "top three" in lowered:
        return []

    for candidate in shortlist:
        candidate_name = str(candidate.get("candidate_name", "")).strip()
        if not candidate_name:
            continue
        if _normalize_name(candidate_name) in lowered or any(part.lower() in lowered for part in candidate_name.split()[:2]):
            names.append(candidate_name)

    return _dedupe_preserve_order(names)


def _build_reasoning(shortlist: Sequence[Dict[str, Any]], requirements: Dict[str, Any]) -> str:
    if not shortlist:
        return "No candidates matched the current requirements."

    top = shortlist[0]
    matched_skills = top.get("matched_skills", []) or []
    gaps = _candidate_gaps(top, requirements)
    top_name = top.get("candidate_name", "Unknown Candidate")
    score = int(top.get("match_score", 0) or 0)

    summary = [
        f"Top-ranked candidate: {top_name} ({score}/100).",
        f"Matched skills: {', '.join(matched_skills[:5]) or 'none explicitly detected'}.",
    ]
    if gaps:
        summary.append(f"Gaps: {', '.join(gaps[:3])}.")
    else:
        summary.append("No critical gaps were detected against the current must-have list.")

    return " ".join(summary)


def _build_comparison_report(
    shortlist: Sequence[Dict[str, Any]],
    requirements: Dict[str, Any],
    target_candidates: Sequence[str],
) -> str:
    if not shortlist:
        return "No candidates were available for comparison."

    if target_candidates:
        selected = [_candidate_lookup(shortlist, candidate_id) for candidate_id in target_candidates]
        selected = [candidate for candidate in selected if candidate]
    else:
        selected = list(shortlist[:3])

    if not selected:
        selected = list(shortlist[:3])

    comparison = compare_candidates(
        candidate_ids=[candidate.get("candidate_name", "") for candidate in selected],
        candidate_pool=list(shortlist),
        requirements=requirements,
    )

    lines = [comparison["summary"], "", "Details:"]
    for row in comparison.get("rows", []):
        lines.append(
            f"- {row['candidate_name']}: score {row['match_score']}, {row['experience']} years, gaps: {', '.join(row['gaps']) or 'none'}"
        )

    return "\n".join(lines).strip()


def _build_interview_plan(
    shortlist: Sequence[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[str, List[str]]:
    plan: Dict[str, List[str]] = {}
    for candidate in shortlist[:3]:
        candidate_name = str(candidate.get("candidate_name", "Unknown Candidate"))
        plan[candidate_name] = generate_interview_questions(candidate_name, candidate, requirements)
    return plan


def _build_enhanced_interview_plan(
    shortlist: Sequence[Dict[str, Any]],
    requirements: Dict[str, Any],
    user_input: str = "",
) -> Dict[str, List[str]]:
    """Generate more comprehensive and targeted questions for the 'questions' intent."""
    plan: Dict[str, List[str]] = {}
    
    # Check if user specified specific candidates to ask about
    target_names = _parse_names_from_query(user_input, shortlist)
    
    # If specific candidates mentioned, generate enhanced questions for them
    if target_names:
        for candidate in shortlist:
            candidate_name = str(candidate.get("candidate_name", "Unknown Candidate"))
            if candidate_name in target_names:
                plan[candidate_name] = _generate_enhanced_questions(candidate_name, candidate, requirements)
    else:
        # Generate enhanced questions for top 3 (or all if fewer than 3)
        for candidate in shortlist[:3]:
            candidate_name = str(candidate.get("candidate_name", "Unknown Candidate"))
            plan[candidate_name] = _generate_enhanced_questions(candidate_name, candidate, requirements)
    
    return plan


def _generate_enhanced_questions(
    candidate_name: str,
    candidate: Dict[str, Any],
    requirements: Dict[str, Any],
) -> List[str]:
    """Generate a diverse set of interview questions targeting gaps, strengths, and growth areas."""
    gaps = _candidate_gaps(candidate, requirements)
    matched_skills = candidate.get("matched_skills", []) or []
    all_skills = candidate.get("all_skills", []) or []
    experience = int(candidate.get("years_of_experience", candidate.get("experience_years", 0)) or 0)
    
    questions: List[str] = []
    
    # 1. Gap-focused questions
    if gaps:
        for gap in gaps[:2]:
            if gap.lower().startswith("needs"):
                questions.append(f"{candidate_name}: Can you walk us through a recent project that demonstrates the experience level we need?")
                questions.append(f"{candidate_name}: How would you approach quickly gaining the experience you're missing?")
            else:
                questions.append(f"{candidate_name}: Describe your practical experience with {gap} - include a specific production use case.")
                questions.append(f"{candidate_name}: How have you used {gap} to solve a real business problem?")
    
    # 2. Strength-focused questions (on matched skills)
    if matched_skills:
        top_skill = matched_skills[0]
        questions.append(f"{candidate_name}: Tell us about your most impressive work with {top_skill}. What was the biggest challenge?")
    
    # 3. Growth & learning questions
    questions.extend([
        f"{candidate_name}: Describe a time you had to learn a completely new technology to solve a problem.",
        f"{candidate_name}: How do you stay updated with latest developments in your field?",
    ])
    
    # 4. Behavioral / soft skills
    questions.extend([
        f"{candidate_name}: Describe a difficult problem you solved end-to-end. What was your approach?",
        f"{candidate_name}: Tell us about a time you disagreed with a team decision. How did you handle it?",
    ])
    
    # 5. Role-specific questions
    if experience >= 5:
        questions.append(f"{candidate_name}: As someone with {experience}+ years, how do you mentor junior developers?")
    
    # Deduplicate and limit to 6-7 for follow-up questions
    questions = _dedupe_preserve_order(questions)
    return questions[:7]


def _build_interview_plan(
    shortlist: Sequence[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[str, List[str]]:
    plan: Dict[str, List[str]] = {}
    for candidate in shortlist[:3]:
        candidate_name = str(candidate.get("candidate_name", "Unknown Candidate"))
        plan[candidate_name] = generate_interview_questions(candidate_name, candidate, requirements)
    return plan


def _recommendation(shortlist: Sequence[Dict[str, Any]], requirements: Dict[str, Any]) -> str:
    if not shortlist:
        return "No-hire until a better match appears."

    top = shortlist[0]
    top_score = int(top.get("match_score", 0) or 0)
    gaps = _candidate_gaps(top, requirements)
    if top_score >= 80 and not any("Needs" in gap for gap in gaps):
        return f"Hire / move forward with {top.get('candidate_name', 'the top candidate')}"
    if top_score >= 65:
        return f"Shortlist / proceed with a second-round interview for {top.get('candidate_name', 'the top candidate')}"
    return f"No-hire / continue searching beyond the current shortlist"


def parse_jd_node(state: AgentState) -> Dict[str, Any]:
    latest_msg = str(state["messages"][-1].content)
    normalized_input = _load_text_if_path(latest_msg)
    intent = _classify_intent(normalized_input)
    # Only extract requirements for search/refine intents; avoid parsing explain/compare queries
    if intent in {"search", "refine"}:
        extracted_requirements = extract_requirements(normalized_input)
        merged_requirements = _merge_requirements(state.get("job_requirements", {}), extracted_requirements)
    else:
        merged_requirements = dict(state.get("job_requirements", {}))
    composed_jd = _compose_job_description(merged_requirements, normalized_input)
    target_candidates = _extract_candidate_targets(normalized_input)

    return {
        "user_input": normalized_input,
        "job_description": composed_jd,
        "job_requirements": merged_requirements,
        "intent": intent,
        "feedback": normalized_input if intent == "refine" else "",
        "target_candidates": target_candidates,
        "screening_round": int(state.get("screening_round", 0) or 0) + 1,
        "previous_candidate_shortlist": list(state.get("candidate_shortlist", [])),
    }


def extract_reqs_node(state: AgentState) -> Dict[str, Any]:
    # Only extract requirements when the intent indicates a search/refine action.
    intent = state.get("intent", "search")
    if intent not in {"search", "refine"}:
        # Preserve existing requirements
        existing = state.get("job_requirements", {})
        return {
            "job_requirements": existing,
            "job_description": _compose_job_description(existing, state.get("user_input", state.get("job_description", ""))),
        }

    # Prefer extracting from the user's original input to avoid re-parsing the composed JD
    source_text = state.get("user_input") or state.get("job_description", "")
    requirements = extract_requirements(source_text)
    merged_requirements = _merge_requirements(state.get("job_requirements", {}), requirements)
    return {
        "job_requirements": merged_requirements,
        "job_description": _compose_job_description(merged_requirements, state.get("user_input", state.get("job_description", ""))),
    }


def search_resumes_node(state: AgentState) -> Dict[str, Any]:
    # Avoid re-running the semantic search for non-search intents (explain/compare/questions)
    intent = state.get("intent", "search")
    requirements = state.get("job_requirements", {})

    # If this turn is just an explain/compare/questions action, reuse previous shortlist
    if intent not in {"search", "refine"} and state.get("candidate_shortlist"):
        return {"candidate_shortlist": state.get("candidate_shortlist", [])}

    match_results = match_job_description(
        job_description=state.get("job_description", ""),
        must_have_skills=requirements.get("must_have", []),
        minimum_experience=requirements.get("minimum_experience", 0),
        top_n=10,
    )

    shortlist = match_results.get("top_matches", [])
    # Fallback: if the semantic index returns nothing (or index not built),
    # perform a lightweight scan over resume text files to find keyword hits.
    if not shortlist:
        def local_resume_scan(must_have: Sequence[str], minimum_experience: int) -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []
            resume_dir = Path("data/resumes")
            if not resume_dir.exists():
                return results

            files = [p for p in resume_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".pdf", ".docx"}]
            for f in files:
                try:
                    loaded = fs_read_file(str(f))
                    if loaded.get("status") != "success":
                        continue
                    text = str(loaded.get("content", "") or "").lower()
                    if not text:
                        continue

                    # simple skill match count
                    matched = []
                    for skill in must_have:
                        if not skill:
                            continue
                        if skill.lower() in text:
                            matched.append(skill)

                    # estimate experience from text
                    exp_matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)", text)
                    years = int(max(exp_matches, default=0)) if exp_matches else 0

                    if minimum_experience and years < minimum_experience:
                        # skip if clearly under the required experience
                        continue

                    score = int(min(100, 50 + (10 * len(matched)) + min(30, years * 3)))
                    candidate_name = f.name.replace("_", " ").replace("resume", "").strip().title()
                    results.append({
                        "candidate_name": candidate_name,
                        "resume_path": str(f),
                        "match_score": score,
                        "matched_skills": matched,
                        "years_of_experience": years,
                        "relevant_excerpts": [text[:220]],
                        "reasoning": f"Found {len(matched)} must-have hits; estimated {years} years experience.",
                    })
                except Exception:
                    continue

            results.sort(key=lambda r: r.get("match_score", 0), reverse=True)
            return results[:10]

        shortlist = local_resume_scan(requirements.get("must_have", []), int(requirements.get("minimum_experience", 0) or 0))

    return {"candidate_shortlist": shortlist}


def rank_candidates_node(state: AgentState) -> Dict[str, Any]:
    shortlist = state.get("candidate_shortlist", [])
    requirements = state.get("job_requirements", {})
    intent = state.get("intent", "search")
    user_input = state.get("user_input", "")
    
    reasoning = _build_reasoning(shortlist, requirements)
    comparison_report = _build_comparison_report(shortlist, requirements, state.get("target_candidates", []))
    
    # For "questions" intent, generate MORE comprehensive questions
    if intent == "questions":
        interview_questions = _build_enhanced_interview_plan(shortlist, requirements, user_input)
    else:
        interview_questions = _build_interview_plan(shortlist, requirements)
    
    recommendation = _recommendation(shortlist, requirements)

    return {
        "ranking_reasoning": reasoning,
        "comparison_report": comparison_report,
        "interview_questions": interview_questions,
        "final_recommendation": recommendation,
    }


def deep_analyze_node(state: AgentState) -> Dict[str, Any]:
    """Perform a second-round deep analysis on the current shortlist.

    - If the shortlist is large (initial pass returned many), reduce to top-10 for deeper scoring.
    - Add `improvement_suggestions` for all candidates (gaps + nice-to-haves + general upskilling).
    - Preserve evidence and matched sections for explainability.
    """
    shortlist = list(state.get("candidate_shortlist", []))
    requirements = state.get("job_requirements", {})

    # If initial pass returned more than 10, trim to top-100 then top-10
    if len(shortlist) > 10:
        shortlist = sorted(shortlist, key=lambda r: int(r.get("match_score", 0)), reverse=True)[:10]

    detailed: List[Dict[str, Any]] = []
    for candidate in shortlist:
        cand = dict(candidate)
        # ensure experience field is available
        cand_ex = int(cand.get("years_of_experience", cand.get("experience_years", 0)) or 0)
        cand_skills = set(s.lower() for s in (cand.get("matched_skills") or []))
        gaps = _candidate_gaps(cand, requirements)

        suggestions: List[str] = []
        
        # Add suggestions for gaps
        for gap in gaps:
            if gap.lower().startswith("needs"):
                # extract number of years needed
                m = re.search(r"needs\s+(\d+)", gap, flags=re.IGNORECASE)
                if m:
                    years = int(m.group(1))
                    suggestions.append(f"Gain ~{years} years of relevant experience or highlight equivalent project work demonstrating that experience.")
                else:
                    suggestions.append("Provide stronger experience examples or longer tenure in similar roles.")
            else:
                # skill gap
                skill = gap
                suggestions.append(f"Showcase practical {skill} work: add a GitHub project or describe production usage and measurable results.")

        # Add suggestions for nice-to-have skills if not already present
        nice_to_have = requirements.get("nice_to_have", []) or []
        for skill in nice_to_have[:2]:  # up to 2 nice-to-have skills
            if skill and skill.lower() not in cand_skills:
                suggestions.append(f"Learn {skill} to become an even stronger fit for the role.")

        # If no gaps and no nice-to-haves missing, add general upskilling suggestion
        if not suggestions:
            score = int(cand.get("match_score", 0) or 0)
            if score < 85:
                suggestions.append("Consider contributing to open-source projects in your core skills to strengthen real-world project examples.")
            suggestions.append("Stay updated with the latest industry trends and best practices in your specialization.")

        cand["gaps"] = gaps
        cand["improvement_suggestions"] = suggestions
        # keep matched evidence short
        cand["evidence_snippet"] = (cand.get("relevant_excerpts") or [])[:2]
        detailed.append(cand)

    # Replace shortlist with the refined top-10 detailed list
    detailed_sorted = sorted(detailed, key=lambda r: int(r.get("match_score", 0)), reverse=True)
    return {"candidate_shortlist": detailed_sorted, "detailed_shortlist": detailed_sorted}


def final_decision_node(state: AgentState) -> Dict[str, Any]:
    """Produce a final recommendation using multi-round thresholds and improvement guidance."""
    shortlist = state.get("candidate_shortlist", []) or []
    requirements = state.get("job_requirements", {})

    if not shortlist:
        return {"final_recommendation": "No-hire until a better match appears."}

    top = shortlist[0]
    top_score = int(top.get("match_score", 0) or 0)
    gaps = _candidate_gaps(top, requirements)

    # stricter thresholds for final decision
    if top_score >= 85 and not gaps:
        decision = f"Hire / move forward with {top.get('candidate_name', 'the top candidate')}"
    elif top_score >= 70:
        decision = f"Shortlist / proceed with final interview for {top.get('candidate_name', 'the top candidate')}. Provide improvement suggestions to borderline candidates."
    else:
        decision = "No-hire / continue searching beyond the current shortlist"

    return {"final_recommendation": decision}


def _build_change_summary(previous: Sequence[Dict[str, Any]], current: Sequence[Dict[str, Any]]) -> str:
    if not previous:
        return ""

    previous_top = [str(item.get("candidate_name", "")).strip() for item in previous[:3]]
    current_top = [str(item.get("candidate_name", "")).strip() for item in current[:3]]
    previous_top = [name for name in previous_top if name]
    current_top = [name for name in current_top if name]

    if previous_top == current_top:
        return "Ranking stayed stable after the latest refinement."

    dropped = [name for name in previous_top if name not in current_top]
    added = [name for name in current_top if name not in previous_top]
    parts = []
    if added:
        parts.append(f"Newly promoted: {', '.join(added)}.")
    if dropped:
        parts.append(f"Moved down or out of the top group: {', '.join(dropped)}.")
    return " ".join(parts)


def generate_report_node(state: AgentState) -> Dict[str, Any]:
    shortlist = state.get("candidate_shortlist", [])
    requirements = state.get("job_requirements", {})
    previous_shortlist = state.get("previous_candidate_shortlist", [])

    role = str(requirements.get("role", "")).strip() or "Candidate Match"
    must_have = requirements.get("must_have", []) or []
    nice_to_have = requirements.get("nice_to_have", []) or []
    minimum_experience = int(requirements.get("minimum_experience", 0) or 0)

    report_lines = [f"### {role} Report", ""]
    if must_have:
        report_lines.append(f"**Must-have skills:** {', '.join(must_have)}")
    if nice_to_have:
        report_lines.append(f"**Nice-to-have skills:** {', '.join(nice_to_have)}")
    if minimum_experience:
        report_lines.append(f"**Minimum experience:** {minimum_experience}+ years")
    report_lines.append("")
    report_lines.append(f"**Recommendation:** {state.get('final_recommendation', 'Review the shortlist')}")
    report_lines.append("")
    report_lines.append(f"**Ranking reasoning:** {state.get('ranking_reasoning', 'No ranking available.')}")
    report_lines.append("")

    if previous_shortlist:
        change_summary = _build_change_summary(previous_shortlist, shortlist)
        if change_summary:
            report_lines.append(f"**What changed after refinement:** {change_summary}")
            report_lines.append("")

    if shortlist:
        report_lines.append("**Top matches:**")
        for index, candidate in enumerate(shortlist[:5], start=1):
            gaps = _candidate_gaps(candidate, requirements)
            report_lines.append(
                f"{index}. {candidate.get('candidate_name', 'Unknown Candidate')} - {candidate.get('match_score', 0)} / 100"
            )
            strengths = ', '.join(candidate.get('matched_skills', [])[:4]) or 'broad profile overlap'
            experience = int(candidate.get('years_of_experience', candidate.get('experience_years', 0)) or 0)
            all_skills = ', '.join(candidate.get('all_skills', [])[:6])
            report_lines.append(f"   - Strengths: {strengths}")
            report_lines.append(f"   - Experience: {experience} years")
            if all_skills:
                report_lines.append(f"   - All skills (sample): {all_skills}")
            report_lines.append(f"   - Gaps: {', '.join(gaps) or 'none'}")
            if candidate.get("relevant_excerpts"):
                report_lines.append(f"   - Evidence: {candidate['relevant_excerpts'][0]}")
            # improvement suggestions from deep analysis
            if candidate.get('improvement_suggestions'):
                for sug in candidate.get('improvement_suggestions')[:2]:
                    report_lines.append(f"   - Improvement: {sug}")
        report_lines.append("")

    if state.get("comparison_report"):
        report_lines.append("**Comparison view:**")
        report_lines.append(state["comparison_report"])
        report_lines.append("")

    if state.get("interview_questions"):
        report_lines.append("**Screening questions:**")
        for candidate_name, questions in state["interview_questions"].items():
            report_lines.append(f"- {candidate_name}")
            for question in questions[:3]:
                report_lines.append(f"  - {question}")
        report_lines.append("")

    follow_up = state.get("follow_up_prompt") or (
        "You can refine this search by adding or removing skills, ask for a side-by-side comparison, or request interview questions for any candidate."
    )
    report_lines.append(f"**Next step:** {follow_up}")

    final_report = "\n".join(report_lines).strip()
    return {"messages": [AIMessage(content=final_report)], "final_report": final_report}


def human_feedback_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("intent", "search")
    interview_questions = state.get("interview_questions", {})
    candidates_with_questions = list(interview_questions.keys()) if interview_questions else []
    
    if intent == "refine":
        prompt = "Requirements updated. You can now refine further, compare top matches, ask for interview questions, or request a deeper explanation."
    elif intent == "compare":
        prompt = "Comparison complete. You can ask for a different comparison set, request interview questions, or refine requirements."
    elif intent == "explain":
        prompt = "Explanation provided. You can ask for more details about other candidates, request interview questions, or refine your search."
    elif intent == "questions":
        if len(candidates_with_questions) == 1:
            prompt = "Interview questions generated for {}. Ask for questions about another candidate or refine requirements.".format(candidates_with_questions[0])
        else:
            prompt = "Interview questions generated for {} candidates. Ask for questions about a specific candidate or refine requirements.".format(len(candidates_with_questions))
    else:
        prompt = "Search complete. You can refine requirements, ask for a comparison, request interview questions, or seek explanations."

    return {"follow_up_prompt": prompt}


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_jd", parse_jd_node)
    workflow.add_node("extract_reqs", extract_reqs_node)
    workflow.add_node("search_resumes", search_resumes_node)
    workflow.add_node("rank_candidates", rank_candidates_node)
    workflow.add_node("deep_analyze", deep_analyze_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("final_decision", final_decision_node)
    workflow.add_node("human_feedback", human_feedback_node)

    workflow.add_edge(START, "parse_jd")
    workflow.add_edge("parse_jd", "extract_reqs")
    workflow.add_edge("extract_reqs", "search_resumes")
    workflow.add_edge("search_resumes", "rank_candidates")
    workflow.add_edge("rank_candidates", "deep_analyze")
    workflow.add_edge("deep_analyze", "final_decision")
    workflow.add_edge("final_decision", "generate_report")
    workflow.add_edge("generate_report", "human_feedback")
    workflow.add_edge("human_feedback", END)

    return workflow


workflow = build_graph()
app = workflow.compile()


def run_agent_turn(user_input: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    current_state = dict(state or {})
    messages = list(current_state.get("messages", []))
    messages.append(HumanMessage(content=user_input))
    current_state["messages"] = messages
    updated_state = app.invoke(current_state)
    return updated_state


def get_last_assistant_message(state: Dict[str, Any]) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return str(state.get("final_report", ""))


if __name__ == "__main__":
    raise SystemExit("Import matching_agent.py and call run_agent_turn(user_input, state) or use app.py.")