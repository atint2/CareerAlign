import re

SKILLS_CACHE = None

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

def extract_skills(text: str, skills_map: dict) -> set[str]:
    text = text.lower()
    found = set()

    for skill in skills_map.keys():
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, text):
            found.add(skill)

    return found

def score_skill(skill: str, meta: dict) -> int:
    score = 0

    if meta.get("hot"):
        score += 2
    if meta.get("in_demand"):
        score += 3

    return score

def build_missing_skills(missing: set[str], skills_map: dict):
    enriched = []

    for skill in missing:
        meta = skills_map.get(skill, {})

        score = score_skill(skill, meta)

        if score >= 4:
            priority = "high"
        elif score >= 2:
            priority = "medium"
        else:
            priority = "low"

        enriched.append({
            "skill": skill,
            "priority": priority,
            "hot": meta.get("hot", False),
            "in_demand": meta.get("in_demand", False)
        })

    return sorted(enriched, key=lambda x: x["priority"] != "high")