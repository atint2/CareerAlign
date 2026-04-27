from spacy.lang.en import English
from spacy.matcher import PhraseMatcher

SKILLS_CACHE = None
_nlp = None
_matcher = None

def get_skills_map(db_session, models):
    global SKILLS_CACHE
    if SKILLS_CACHE is None:
        skills = db_session.query(models.Skill).all()
        SKILLS_CACHE = {
            s.skill.lower(): {
                "hot": (s.hot_technology or "").lower() == "yes",
                "in_demand": (s.in_demand or "").lower() == "yes"
            }
            for s in skills
        }
    return SKILLS_CACHE

def get_phrase_matcher(skills_map: dict) -> tuple:
    """Build a spaCy PhraseMatcher from your DB skills. Cached after first build."""
    global _nlp, _matcher
    if _matcher is None:
        _nlp = English()
        _matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
        patterns = [_nlp.make_doc(skill) for skill in skills_map.keys()]
        _matcher.add("SKILLS", patterns)
    return _nlp, _matcher

def _extract_phrase_matcher(text: str, skills_map: dict) -> set[str]:
    """Use spaCy PhraseMatcher for efficient, token-aware skill extraction."""
    nlp, matcher = get_phrase_matcher(skills_map)
    doc = nlp(text.lower())
    found = set()
    for _, start, end in matcher(doc):
        span = doc[start:end].text.lower()
        if span in skills_map:
            found.add(span)
    return found

def extract_skills(text: str, skills_map: dict) -> set[str]:
    return _extract_phrase_matcher(text, skills_map)

def build_missing_skills(missing: set[str], skills_map: dict):
    enriched = []
    for skill in missing:
        meta = skills_map.get(skill, {})
        enriched.append({
            "skill": skill,
            "hot": meta.get("hot", False),
            "in_demand": meta.get("in_demand", False)
        })
    return enriched