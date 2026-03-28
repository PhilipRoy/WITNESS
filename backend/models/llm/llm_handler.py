# W.I.T.N.E.S.S. - Web-based Interrogation and Testimony via a Neural Engaged Speech System
# Copyright (C) 2026 Philip Roy <https://www.bluengrey.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import re
import time
from calendar import month_name
import requests
import json
from datetime import datetime
from dateutil.parser import parse as parse_date

def _detect_ollama_base_url() -> str:
    """Return the Ollama API base URL.

    Checks the environment variable first. If running inside a Docker container
    (detected via /.dockerenv), uses the Docker host bridge address so the
    container can reach Ollama running on the host machine.
    """
    env_url = os.getenv("OLLAMA_BASE_URL")
    if env_url:
        return env_url
    try:
        if os.path.exists("/.dockerenv"):
            return "http://host.docker.internal:11434"
    except Exception:
        pass
    return "http://localhost:11434"

OLLAMA_BASE_URL = _detect_ollama_base_url()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")

def _is_open_incident_question(text: str) -> bool:
    """Return True if the interviewer is asking the witness to describe what they saw or experienced.

    This determines whether to serve the constrained 3-sentence opener (Tier 2) rather than
    delegating to the general LLM. The patterns are intentionally broad because interviewers
    phrase open questions in many ways; false positives are preferable to missing the opener.
    """
    t = (text or "").lower()
    if not t:
        return False
    keys = [
        "tell me what you saw", "what happened", "what did you see",
        "describe what you saw", "walk me through what you saw",
        "talk me through what you saw", "describe the incident",
        "tell me about the incident", "tell me what you witnessed",
        "what did you witness", "what did you observe"
    ]
    if any(k in t for k in keys):
        return True
    if re.search(r"\btell me (about )?(what you (saw|witnessed)|what happened)\b", t):
        return True
    if re.search(r"\b(describe|what\s+did\s+(he|the\s+(man|boy|person|suspect))\s+look\s+like)\b", t):
        return True
    if re.search(r"\b(where\s+did\s+(he|they)\s+(go|run)\b)", t):
        return True
    if re.search(r"\b(what\s+time|when\s+was\s+this)\b", t):
        return True
    if re.search(r"\b(did|could)\s+you\s+see\b.*\b(weapon|gun|knife)\b", t):
        return True
    if re.search(r"\b(did|could)\s+you\s+(see|get\s+a\s+look\s+at)\s+(his|the)\s+face\b", t):
        return True
    if re.search(r"\b(did|could)\s+you\s+see\s+inside\s+(the\s+)?(shop|store)\b", t):
        return True
    if re.search(r"\bhow\s+far\b", t):
        return True
    # Require a question mark to avoid triggering on statements that mention "incident"
    # as a passing reference (e.g., "I understand there was an incident").
    if "?" not in t:
        return False
    # Must be asking ABOUT what happened/was witnessed, not just mentioning those words
    if re.search(r"\bwhat\s+(?:did\s+you\s+)?(see|saw|witness|observe)\b", t):
        return True
    if re.search(r"\bwhat\s+happened\b", t) and not re.search(r"\bdon'?t\s+feel\s+bad\b", t):
        return True
    return False

def _is_persona_assertion(text: str) -> bool:
    """Return True if the interviewer is asserting a fact about the witness rather than asking a question.

    Used to prevent the unsolicited-narrative suppressor from cutting a legitimate correction response.
    Example: "I understand you used to work at the dairy" — the witness should confirm or deny.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    patterns = [
        r"\bi\s+(understand|believe|gather|was\s+told|heard)\b.*\b(you|you're|you were|you used to be)\b",
        r"\bit\s+says\s+here\b.*\b(you|you're|you were|you used to be)\b",
        r"\byou\s+(are|were|used\s+to\s+be)\b.*[\.!]?$",
    ]
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)

def _is_confirmation_question(text: str) -> bool:
    """Detects tag questions/confirmation requests like 'correct?', 'right?', 'is that right?'"""
    t = (text or "").strip().lower()
    if not t:
        return False
    # Tag questions: "..., correct?", "..., right?", "..., yes?"
    if re.search(r",\s*(correct|right|yes|true|yeah)\s*\??$", t):
        return True
    # End-of-sentence confirmation: "...correct?", "...right?"
    if re.search(r"\b(correct|right|yes|true)\s*\??$", t):
        return True
    # Explicit confirmation: "is that correct?", "is that right?"
    if re.search(r"\bis\s+that\s+(correct|right|true|accurate)\b", t):
        return True
    return False

def _is_smalltalk_question(text: str) -> bool:
    """Return True for opening pleasantries that should get a short polite response, not an incident description."""
    t = (text or "").lower().strip()
    if not t:
        return False
    patterns = [
        r"\bhow\s+are\s+you\b\s*(\?|$)",
        r"\bhow\s+is\s+(it\s+going|your\s+day(?:\s+going)?)\b\s*(\?|$)",
        r"\bhow'?s\s+(it\s+going|your\s+day(?:\s+going)?)\b\s*(\?|$)",
        r"\bare\s+you\s+(ready|there|ok|okay)\b\s*(\?|$)",
        r"\bcan\s+you\s+hear\s+me\b\s*(\?|$)",
        r"\btest(ing)?\b\s*(\?|$)",
    ]
    return any(re.search(p, t) for p in patterns)

def _is_how_are_you_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(re.search(r"\bhow\s+are\s+you\b", t))

def _is_hows_day_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Match any "how" + "day" question or "having a good/bad day" question
    return bool(
        re.search(r"\bhow\s+(is|'?s|has|was)\s+(your\s+day|it)\b", t)
        or re.search(r"\b(having|had)\s+a\s+(good|bad|nice|great)\s+day\b", t)
        or re.search(r"\bare\s+you\s+having\s+a\s+(good|bad|nice|great)\s+day\b", t)
    )

def _is_trivial_pref_question(text: str) -> bool:
    """Return True for off-topic preference questions ("Do you like football?") that should be deflected.

    Witnesses stay in-character and politely redirect rather than engaging with irrelevant topics.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    patterns = [
        r"\bdo\s+you\s+like\b",
        r"\bwhat('?| i)s\s+your\s+favo?urite\b",
        r"\bwhat\s+is\s+your\s+favorite\b",
        r"\bwhat\s+do\s+you\s+like\b",
        r"\bwhat\s+food\s+do\s+you\s+like\b",
    ]
    return any(re.search(p, t) for p in patterns)

def _is_consent_prompt(text: str) -> bool:
    """Return True if the interviewer is asking the witness to confirm they are ready to proceed.

    These prompts get a canned "Yes, that's okay." reply rather than going through the LLM,
    since the witness should always agree to begin — no persona-specific variation is needed.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    patterns = [
        r"\bis\s+that\s+(ok|okay|alright|all\s*right)\b",
        r"\bare\s+you\s+okay\s+with\s+that\b",
        r"\bis\s+that\s+acceptable\b",
        r"\bshall\s+we\s+(begin|get\s+started|start)\b",
        r"\b(if\s+you'?re\s+feeling\s+up\s+to\s+it|when\s+you'?re\s+ready).*(we'?ll\s+)?get\s+started\b",
    ]
    return any(re.search(p, t) for p in patterns)

def _is_job_enjoyment_prompt(text: str) -> bool:
    """Return True if the interviewer is asking whether the witness enjoys their job.

    These questions need a nuanced answer grounded in persona background, so they are routed
    to a constrained micro-LLM call (_answer_job_enjoyment_via_llm) rather than the generic handler.
    """
    t = (text or "").lower().strip()
    if not t:
        return False
    if re.search(r"\b(do|did|would)\s+you\s+(enjoy|like|love)\b", t) and re.search(r"\b(job|work|role|working|being)\b", t):
        return True
    if re.search(r"\b(must\s+be|sounds\s+(?:like\s+)?)\b.*\b(rewarding|great|good|enjoyable|fulfilling|tough|hard|challenging)\b.*\b(job|role|work)\b", t):
        return True
    if re.search(r"\b(must\s+be|sounds\s+(?:like\s+)?)\b.*\b(job|role|work)\b.*\b(rewarding|great|good|enjoyable|fulfilling|tough|hard|challenging)\b", t):
        return True
    return False


def _answer_job_enjoyment_via_llm(base_msgs, persona_data, last_user_text):
    """Send a constrained micro-prompt to the LLM to answer job-enjoyment questions.

    Uses a separate micro-call with strict rules (1–2 sentences, no invention) rather than
    the full conversation prompt, because the full prompt's broad persona context would
    encourage the model to elaborate with invented employment history.
    Falls back to neutral canned responses if the LLM is unavailable or returns nothing useful.
    """
    occupation = (persona_data.get("occupation") or "").strip()
    employed_by = (persona_data.get("employed_by") or "").strip()
    hidden = persona_data.get("hidden_motivations") or []

    # Tight guardrails: 1–2 sentences, only what the persona background supports.
    sys_rules = {
        "role": "system",
        "content": (
            "When asked whether you enjoy your job, respond in 1–2 short sentences in plain NZ English. "
            "Base your answer only on the persona background below; do not invent employers, history, or extra facts. "
            "Be brief and natural, and avoid small talk or meta comments. "
            "If the background gives no clue, answer neutrally without guessing."
        )
    }

    # Only pass occupation-relevant fields — hidden_motivations can hint at job dissatisfaction.
    facts_lines = []
    if occupation:
        facts_lines.append(f"Occupation: {occupation}")
    if employed_by:
        facts_lines.append(f"Employed by: {employed_by}")
    relevant_hidden = [h for h in hidden if isinstance(h, str) and h.strip()]
    if relevant_hidden:
        facts_lines.append("Possible factors: " + "; ".join(relevant_hidden))

    user_blob = {
        "role": "user",
        "content": (
            "PERSONA BACKGROUND:\n" +
            ("\n".join(facts_lines) if facts_lines else "(no specific job background)") +
            "\n\nQUESTION: " + (last_user_text or "Do you enjoy your job?")
        )
    }

    payload = {"model": OLLAMA_MODEL, "stream": True, "messages": base_msgs + [sys_rules, user_blob]}
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=45, stream=True)
        if r.status_code != 200:
            # Neutral fallback — avoids evasive responses like "I'd rather not say".
            if occupation:
                return f"Mostly, yes. Working as a {occupation} can be challenging at times."
            else:
                return "It's alright, I guess."
        text = ""
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            content = chunk.get("message", {}).get("content", "")
            if content:
                text += content
            if chunk.get("done", False):
                break
        out = (text or "").strip()
    except Exception:
        if occupation:
            return f"Mostly, yes. Working as a {occupation} can be challenging at times."
        else:
            return "It's just a job, nothing more."

    # Strip any role labels the model may have prefixed, then cap at 2 sentences.
    out = re.sub(r"(?im)^(interviewer|detective|officer)\s*:\s*", "", out).strip()
    parts = re.split(r'(?<=[\.!\?])\s+', out)
    trimmed = " ".join(parts[:2]).strip()

    if trimmed:
        return trimmed

    # Last-resort neutral fallbacks
    if occupation:
        return f"Mostly, yes. Working as a {occupation} can be challenging at times."
    return "It's alright, I guess."


def _answer_occupation_via_llm(base_msgs, persona_data, last_user_text):
    """Send a constrained micro-prompt to produce a one-sentence answer about the witness's occupation.

    Occupation questions are routed here rather than to the full LLM because the model tends
    to invent plausible-sounding but fictional job descriptions when given broad context.
    Enforcing a single sentence with only persona background data prevents hallucination.
    """
    occupation = (persona_data.get("occupation") or "").strip()
    employed_by = (persona_data.get("employed_by") or "").strip()

    sys_rules = {
        "role": "system",
        "content": (
            "When asked about your job or occupation, answer in one short sentence in plain NZ English. "
            "Base your answer only on the persona background below. Do not invent employers, roles, or history. "
            "If the background does not state an occupation or employer, reply neutrally that you are not currently working."
        ),
    }

    facts_lines = []
    if occupation:
        facts_lines.append(f"Occupation: {occupation}")
    if employed_by:
        facts_lines.append(f"Employed by: {employed_by}")

    user_blob = {
        "role": "user",
        "content": (
            "PERSONA BACKGROUND:\n" +
            ("\n".join(facts_lines) if facts_lines else "(no job details)") +
            "\n\nQUESTION: " + (last_user_text or "What job do you do?")
        ),
    }

    payload = {"model": OLLAMA_MODEL, "stream": True, "messages": base_msgs + [sys_rules, user_blob]}

    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=45, stream=True)
        if r.status_code != 200:
            return "I'm not currently working."
        text = ""
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            content = chunk.get("message", {}).get("content", "")
            if content:
                text += content
            if chunk.get("done", False):
                break
        out = (text or "").strip()
    except Exception:
        return "I'm not currently working."

    out = re.sub(r"(?im)^(interviewer|detective|officer)\s*:\s*", "", out).strip()
    parts = re.split(r'(?<=[\.!\?])\s+', out)
    # Keep only the first sentence — occupation answers must be brief.
    trimmed = " ".join(parts[:1]).strip()
    return trimmed or ("I'm not currently working." if not occupation or occupation.lower() in ("unemployed", "not employed", "not working") else f"I'm a {occupation}.")


def _contains_unsolicited_narrative(text: str) -> bool:
    """Return True if a response appears to contain an unprompted incident retelling.

    The LLM sometimes volunteers the full incident story even when answering a simple yes/no
    question. This heuristic detects first-person past-tense narrative so the caller can
    suppress it or replace it with a generic "I'm not sure." / "No." response.
    Two separate signal tiers are checked: strong (first-person + action verb) and medium
    (two of: time expression, place term, vehicle+movement).
    """
    if not text:
        return False
    t = text.lower()
    first_person = re.search(
        r"\b(i|we)\b", t
    )
    past_verbs = re.search(
        r"\b(saw|heard|went|ran|walked|drove|parked|stopped|entered|exited|left|called|phoned|noticed|approached|took|pulled|looked)\b",
        t
    )
    if first_person and past_verbs:
        return True
    time_expr = re.search(
        r"\b(yesterday|last\s+(night|evening|week|friday|saturday|sunday|monday|tuesday|wednesday|thursday))\b"
        r"|(\b\d{1,2}[:\.]\d{2}\s*(am|pm)?\b)"
        r"|(\b\d{1,2}\s*(am|pm)\b)"
        r"|(\b\d{1,2}\s+\b(january|february|march|april|may|june|july|august|september|october|november|december)\b)",
        t
    )
    place_term = re.search(
        r"\b(street|road|ave|avenue|lane|alley|intersection|roundabout|shop|store|house|building|carpark|parking|corner)\b", t
    )
    vehicle_term = re.search(r"\b(car|vehicle|van|ute|truck|motorbike|bike|bicycle|scooter)\b", t)
    movement = re.search(r"\b(ran|drove|sped|turned|pulled\s+over|took\s+off|got\s+into|emerged|fled)\b", t)
    medium_hits = sum(bool(x) for x in [time_expr, place_term, (vehicle_term and movement)])
    return medium_hits >= 2

def _movement_mode_from_facts(facts_low: str) -> str:
    """Return 'driving along' or 'walking along' based on what the facts say about the witness's mode of travel.

    Used in the first-overview sentence builder to set the opening clause
    ("I was driving along…" vs "I was walking along…").
    Defaults to walking if no driving cues are found.
    """
    if not facts_low:
        return "walking along"
    if re.search(r"\b(driving|drove|as\s+you\s+drove|in\s+(my|a)\s+car|dashboard|steering\s*wheel)\b", facts_low):
        return "driving along"
    return "walking along"

def _place_category_from_facts(facts_low: str) -> str:
    """Classify the incident location into a broad category used to choose natural phrasing.

    The category drives _place_anchor_from_category(), which produces phrases like
    "at the property" or "near the shop" for the first-overview sentence.
    Returns "unknown" when no location keywords are found rather than guessing.
    """
    if re.search(r"\b(house|home|residence|dwelling|property|address|garage|shed|carport|outbuilding)\b", facts_low):
        return "residential"
    if re.search(r"\b(store|shop|business|premises)\b", facts_low):
        return "commercial"
    if re.search(r"\b(street|road|avenue|ave|lane|alley|alleyway|highway|footpath|sidewalk|carpark|parking|intersection|corner)\b", facts_low):
        return "public_way"
    return "unknown"

def _subarea_from_facts(facts_low: str) -> str | None:
    if re.search(r"\b(lawn|yard|garden|grass)\b", facts_low):
        return "on the lawn"
    if re.search(r"\b(driveway)\b", facts_low):
        return "by the driveway"
    if re.search(r"\b(porch|deck|veranda|verandah)\b", facts_low):
        return "on the porch"
    if re.search(r"\b(carpark|parking)\b", facts_low):
        return "in the carpark"
    if re.search(r"\b(corner|intersection)\b", facts_low):
        return "at the corner"
    return None

def _place_anchor_from_category(cat: str) -> str | None:
    if cat == "residential":
        return "at the property"
    if cat == "commercial":
        return "near the shop"
    if cat == "public_way":
        return "on the street"
    return None

def _event_from_facts(facts_low: str) -> str:
    """
    Infer a high-level event label from facts. Keep generic, no scenario hard-coding.
    """
    if re.search(r"\b(fire|arson)\b", facts_low):
        return "a fire"
    if re.search(r"\b(robbery|robbed|held\s*up|hold-?up|theft|stole|steal|shop\s*lift|shop-?lift)\b", facts_low):
        return "a robbery"
    if re.search(r"\b(glass\s+breaking|broke\s+glass|smashed\s+glass|shouting|screaming|commotion|disturbance)\b", facts_low):
        return "a disturbance"
    if re.search(r"\b(ran\s+into|went\s+into|entered|inside)\b", facts_low):
        return "someone go inside"
    if re.search(r"\b(came\s+out|ran\s+out|exited)\b", facts_low):
        return "someone come out"
    return "an incident"

def _pronoun_forms(facts_low: str):
    """Return (He/She/They, he/she/they, his/her/their, was/were) inferred from the facts.

    Used to produce grammatically correct sentences in the first-overview builder without
    hardcoding a gender. Returns they/their/were as the gender-neutral default.
    """
    if re.search(r"\b(boy|man|male|he)\b", facts_low):
        return ("He", "he", "his", "was")
    if re.search(r"\b(girl|woman|female|she)\b", facts_low):
        return ("She", "she", "her", "was")
    return ("They", "they", "their", "were")

def _extract_incident_timeframe(persona_prompt: str) -> str | None:
    """Extract the incident timeframe from persona_prompt (e.g., 'three days ago', 'yesterday', 'last Tuesday').

    Returns a phrase like 'three days ago' or None if not found.
    """
    if not persona_prompt:
        return None
    prompt_low = persona_prompt.lower()
    # Match patterns like "three days ago", "two weeks ago", "a week ago", "yesterday", "last Tuesday"
    # Pattern 1: X days/weeks/months ago
    m = re.search(r"\b(a|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(day|days|week|weeks|month|months)\s+ago\b", prompt_low)
    if m:
        return m.group(0)
    # Pattern 2: yesterday, last night, last week, last Tuesday, etc.
    m = re.search(r"\b(yesterday|last\s+(?:night|evening|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", prompt_low)
    if m:
        return m.group(0)
    # Pattern 3: "on [day]" like "on Tuesday", "on the 15th"
    m = re.search(r"\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|the\s+\d{1,2}(?:st|nd|rd|th)?)\b", prompt_low)
    if m:
        return m.group(0)
    return None

def _extract_time_phrase(facts_low: str) -> str | None:
    """Extract the most incident-relevant time expression from the facts.

    When multiple times appear (e.g., "I left work at 5pm. The fire was at 5:30pm."),
    a proximity-and-specificity score is used to prefer the time closest to an incident
    keyword and with a non-zero minute component (e.g., 5:30 beats 5:00).
    Returns an "around HH:MM" phrase or None if no time is found.
    """
    times = [m for m in re.finditer(r"\b(\d{1,2}[:\.]\d{2})\s*(am|pm)?\b", facts_low)]
    if not times:
        if re.search(r"\b(yesterday|last\s+(night|evening|week))\b", facts_low):
            return "around that time"
        return None
    incident_anchor = re.compile(r"\b(fire|arson|robbery|incident|saw|witnessed|smoke|ran|entered|went\s+inside)\b")
    best = None
    best_score = -1
    for m in times:
        hhmm = m.group(1)
        minutes = int(hhmm.replace('.', ':').split(':')[1])
        start = max(0, m.start() - 60)
        end = min(len(facts_low), m.end() + 60)
        window = facts_low[start:end]
        proximity = 1 if incident_anchor.search(window) else 0
        non_zero_min = 1 if minutes != 0 else 0
        score = proximity * 2 + non_zero_min
        if score > best_score:
            best = m
            best_score = score
    t = best.group(1).replace('.', ':') if best else times[0].group(1).replace('.', ':')
    ampm = (best.group(2) if best else times[0].group(2)) or ''
    return f"around {t}{ampm}".strip()

def _street_anchor_phrase(facts_text: str) -> str | None:
    if not facts_text:
        return None
    m = re.search(r"\b([A-Z][A-Za-z'’\-]+)\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln)\b", facts_text, re.IGNORECASE)
    if not m:
        return None
    name = m.group(1)
    road = m.group(2)
    anchor = f"{name.title()} {road.title()}"
    return f"on {anchor}"

def _extract_named_route(facts_text: str) -> str | None:
    if not facts_text:
        return None
    m1 = re.search(r"\b([A-Z][A-Za-z'’\-]+)'s\s+(lane|alley|alleyway|path|track)\b", facts_text, re.IGNORECASE)
    if m1:
        name = m1.group(1)
        kind = m1.group(2)
        return f"{name.title()}'s {kind.title()}"
    m2 = re.search(r"\b([A-Z][A-Za-z'’\-]+)\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Way)\b", facts_text, re.IGNORECASE)
    if m2:
        return f"{m2.group(1).title()} {m2.group(2).title()}"
    return None

def _extract_distance_phrase(facts_low: str) -> str | None:
    """Extract the first numeric metre distance from the facts and normalise its unit spelling.

    "meters" is converted to "metres" for NZ English consistency. Returns None if no
    metric distance is present — the caller should then return "I'm not sure." rather
    than guessing a distance.
    """
    m = re.search(r"\b(\d{1,4})\s*(metres|meters|m)\b", facts_low)
    if not m:
        return None
    num, unit = m.group(1), m.group(2)
    unit = "metres" if unit.lower().startswith(("m", "meter")) else unit
    return f"{num} {unit}"

def _vehicle_details_phrase(facts_low: str) -> str | None:
    """Extract a natural vehicle description (colour + make/model + plates) from the facts.

    Processes facts line by line to avoid colour bleed across sentences
    (e.g., "light grey Mazda2" on one line, "black jeans" on the next). Tries named
    make/model first; falls back to generic body type (sedan, hatchback, etc.) if no
    make is found. Returns None if no vehicle is mentioned at all.
    """
    colours = r"(light\s+grey|light\s+gray|dark\s+grey|dark\s+gray|silver|grey|gray|black|white|red|blue|green|brown)"
    model_patterns = [
        r"\bmazda\s*3\b", r"\bmazda3\b", r"\bmazda\s+three\b",
        r"\bmazda\s*2\b", r"\bmazda2\b", r"\bmazda\s+two\b",
        r"\btoyota\s+corolla\b", r"\bsubaru\b", r"\bholden\b", r"\bford\b",
        r"\bnissan\b", r"\bhonda\b", r"\bhyundai\b"
    ]
    lines = [ln.strip() for ln in (facts_low or "").splitlines() if ln.strip()]
    chosen_colour = None
    chosen_model = None
    chosen_generic = None
    plates = None
    def _norm_model(raw: str) -> str:
        raw = re.sub(r"\bmazda\b", "Mazda", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b(Mazda)\s+(\d)\b", r"\1\2", raw)
        return re.sub(r"\s+", " ", raw).strip()
    for ln in lines:
        m_model = None
        for pat in model_patterns:
            m = re.search(pat, ln)
            if m:
                m_model = m
                break
        if not m_model:
            continue
        m_col = re.search(rf"\b{colours}\b", ln)
        if m_col:
            chosen_colour = m_col.group(0)
        chosen_model = _norm_model(m_model.group(0))
        if re.search(r"\b(no|without|missing)\s+(number\s+)?plates?\b|\bno\s+licen[cs]e\s+plates?\b", ln):
            plates = "without plates"
        break
    if not chosen_model:
        for ln in lines:
            m_gen = re.search(r"\b(sedan|hatchback|hatch|ute|van|truck|motorbike|scooter|car|vehicle)\b", ln)
            if not m_gen:
                continue
            chosen_generic = m_gen.group(0)
            m_col = re.search(rf"\b{colours}\b", ln)
            if m_col:
                chosen_colour = m_col.group(0)
            if re.search(r"\b(no|without|missing)\s+(number\s+)?plates?\b|\bno\s+licen[cs]e\s+plates?\b", ln):
                plates = "without plates"
            break
    if not (chosen_model or chosen_generic):
        return None
    bits = []
    if chosen_colour:
        colour = chosen_colour.replace("gray", "grey")
        colour = re.sub(r"\s+", " ", colour).strip()
        bits.append(colour)
    if chosen_model:
        bits.append(chosen_model)
    else:
        bits.append(chosen_generic)
    phrase = " ".join(bits).strip()
    if plates:
        phrase += f" {plates}"
    phrase = re.sub(r"\s+", " ", phrase)
    return ("a " + phrase).strip()

def _vehicle_colour_from_facts(facts_low: str) -> str | None:
    colours = r"(light\s+grey|light\s+gray|dark\s+grey|dark\s+gray|silver|grey|gray|black|white|red|blue|green|brown)"
    model_patterns = [
        r"\bmazda\s*3\b", r"\bmazda3\b", r"\bmazda\s+three\b",
        r"\bmazda\s*2\b", r"\bmazda2\b", r"\bmazda\s+two\b",
        r"\btoyota\s+corolla\b", r"\bsubaru\b", r"\bholden\b", r"\bford\b",
        r"\bnissan\b", r"\bhonda\b", r"\bhyundai\b"
    ]
    lines = [ln.strip() for ln in (facts_low or "").splitlines() if ln.strip()]
    for ln in lines:
        if any(re.search(pat, ln) for pat in model_patterns) or re.search(r"\b(car|vehicle|van|truck|ute|motorbike|scooter|hatch|hatchback|sedan)\b", ln):
            m_col = re.search(rf"\b{colours}\b", ln)
            if m_col:
                col = m_col.group(0).replace("gray", "grey")
                return re.sub(r"\s+", " ", col).strip()
    return None

def _mask_description_from_facts(facts_low: str) -> str | None:
    if "mask" not in (facts_low or ""):
        return None
    colours = r"(black|white|dark|light|grey|gray|navy|blue)"
    types = r"(ski|balaclava|surgical|cloth|hood|full-?face|face|medical|disposable)"
    for ln in (facts_low or "").splitlines():
        if "mask" not in ln:
            continue
        m_typ = re.search(rf"\b{types}\b\s+mask|mask\s+\b{types}\b", ln, re.IGNORECASE)
        if m_typ:
            typ = re.sub(r"\s+", " ", m_typ.group(0)).strip()
            return f"a {typ}"
        m_col = re.search(rf"\b{colours}\b\s+mask|mask\s+\b{colours}\b", ln, re.IGNORECASE)
        if m_col:
            col = m_col.group(0).replace("gray", "grey")
            col = re.sub(r"\s+", " ", col).strip()
            return f"a {col}"
    return "a simple mask that covered their face"

def _clothing_list_from_facts(facts_low: str) -> list[str]:
    """Parse clothing items with their locally-associated colours from free-text fact lines.

    Each line is split on commas and "and" before matching, so colour and garment are
    extracted from the same short chunk. This prevents colour bleed — e.g., "white trainers,
    black jeans" must not produce "white jeans". Non-clothing items (bags, weapons) are
    explicitly excluded so they don't absorb stray colour words.
    Returns a deduplicated list in source order, e.g. ["white trainers", "black jeans"].
    """
    if not facts_low:
        return []
    lines = [ln.strip() for ln in (facts_low or "").splitlines() if ln.strip()]

    garments = {
        "sweatshirt": r"(?:hoodie|sweat\s*shirt|sweatshirt|jumper)",
        "jeans": r"(?:jeans)",
        "trousers": r"(?:trousers|pants)",
        "trainers": r"(?:trainers|sneakers|running\s*shoes)",
        "shoes": r"(?:shoes|boots)",
        "tshirt": r"(?:t\s*-?shirt|tee\s*shirt)",
        "jacket": r"(?:jacket|coat)",
        "cap": r"(?:cap|hat|beanie)",
    }
    colour_pat = r"(black|white|grey|gray|blue|red|green|brown|dark|light|navy)"

    out, seen = [], set()
    for ln in lines:
        # Split list-style descriptions to reduce colour bleed across items
        chunks = re.split(r"\s*(?:,| and )\s*", ln, flags=re.IGNORECASE)
        for chunk in chunks:
            clow = chunk.lower()
            # Non-clothing items can carry stray colour words that would pollute results.
            if re.search(r"\b(backpack|bag|knife|weapon)\b", clow):
                continue
            for canonical, gpat in garments.items():
                # Pattern 1 — colour before garment: "white trainers"
                m1 = re.search(rf"\b{colour_pat}\s+{gpat}\b", clow, re.IGNORECASE)
                # Pattern 2 — garment then colour within a short window: "trainers … white"
                m2 = None
                if not m1:
                    m2 = re.search(
                        rf"\b{gpat}\b(?:\s+(?:that\s+were|that\s+was|were|was))?\s*(?:\w+\s+)?\b{colour_pat}\b",
                        clow, re.IGNORECASE
                    )

                if m1 or m2:
                    col = (m1.group(1) if m1 else m2.group(1)).replace("gray", "grey")
                    name = "T-shirt" if canonical == "tshirt" else canonical
                    phrase = f"{col} {name}"
                    if phrase not in seen:
                        seen.add(phrase)
                        out.append(phrase)
                    continue

                # Pattern 3 — garment mentioned with no associated colour
                if re.search(rf"\b{gpat}\b", clow, re.IGNORECASE):
                    name = "T-shirt" if canonical == "tshirt" else canonical
                    if name not in seen:
                        seen.add(name)
                        out.append(name)

    return out

def _sweatshirt_description(facts_low: str) -> str | None:
    for ln in (facts_low or "").splitlines():
        if re.search(r"\b(hoodie|sweat\s*shirt|sweatshirt|jumper)\b", ln):
            col = re.search(r"\b(black|white|grey|gray|blue|red|green|brown|dark|light|navy)\b", ln)
            if col:
                c = col.group(0).replace("gray", "grey")
                return f"It was just {c}, with nothing else on it."
            return "It looked like a plain sweatshirt."
    return None

# Returns True if markings/text/images on sweatshirt are mentioned, False if garment is mentioned but with no markings, None if not mentioned at all.
def _sweatshirt_has_markings_from_facts(facts_low: str) -> bool | None:
    """Check whether the facts describe markings (text, logo, graphic) on the hoodie/sweatshirt.

    Returns:
        True  — the garment is mentioned and markings are explicitly described.
        False — the garment is mentioned but no markings are described (treat as plain).
        None  — the garment is not mentioned at all; the question cannot be answered.

    A three-way return is needed because "No" and "I don't know" are meaningfully different
    answers to "Did it have any writing on it?"
    """
    if not facts_low:
        return None
    lines = [ln.strip().lower() for ln in (facts_low or "").splitlines() if ln.strip()]
    gpat = re.compile(r"\b(hoodie|sweat\s*shirt|sweatshirt|jumper)\b")
    mark_pat = re.compile(r"\b(text|writing|logo|brand|image|graphic|picture|print|printed)\b")
    seen_garment = False
    for ln in lines:
        if gpat.search(ln):
            seen_garment = True
            if mark_pat.search(ln):
                return True
    if seen_garment:
        return False
    return None

def _facts_have_shaved_head(facts_low: str) -> bool:
    """Return True if the facts mention a shaved head.

    Used post-LLM to override any hair-colour mentions the model may have invented —
    a person with a shaved head cannot also have "brown hair."
    """
    return bool(re.search(r"\bshaved\s+head\b", facts_low))

def _facts_have_beard(facts_low: str) -> bool:
    """Return True if the facts mention a beard.

    Used post-LLM to strip out beard references when the persona's facts make no mention
    of one — the model occasionally invents facial hair.
    """
    return bool(re.search(r"\bbeard\b", facts_low))

def _build_first_overview(facts_list: list[dict]) -> str:
    """Build a deterministic 2–3 sentence incident overview from structured persona facts.

    This was an early alternative to the LLM-based opener and is kept as a fallback.
    It assembles fixed sentences from the movement mode, location, event type, time,
    and key witness actions extracted from facts — no LLM call is made.
    """
    facts_text = "\n".join((f.get("fact", "") for f in facts_list if isinstance(f, dict)))
    facts_low = facts_text.lower()

    time_phrase = _extract_time_phrase(facts_low)
    street_anchor = _street_anchor_phrase(facts_text)

    movement = _movement_mode_from_facts(facts_low)
    event = _event_from_facts(facts_low)
    subarea = _subarea_from_facts(facts_low)
    place_cat = _place_category_from_facts(facts_low)
    place_anchor = _place_anchor_from_category(place_cat)

    lead_bits = []
    if street_anchor:
        lead_bits.append(street_anchor)
    if time_phrase:
        lead_bits.append(time_phrase)
    lead = (" " + " ".join(lead_bits)) if lead_bits else ""

    s1 = f"I was {movement}{lead} when I saw {event}"
    if subarea:
        s1 += f" {subarea}"
    if place_anchor and place_cat in {"residential","commercial"}:
        if ("house" in place_anchor and "house" not in (subarea or "")) or place_cat == "commercial":
            s1 += f" {place_anchor}"
    s1 = s1.strip() + "."

    actions = []
    if re.search(r"\bstopped?\s+to\s+investigate\b", facts_low):
        actions.append("I stopped to check")
    if re.search(r"\bput\s+(it|the\s+fire)\s+out\b", facts_low):
        actions.append("I put it out")
    if re.search(r"\bcalled?\s+(111|105)\b|\bcall\s+(111|105)\b", facts_low):
        actions.append("I called police")
    s2 = ""
    if actions:
        if len(actions) == 1:
            s2 = actions[0].capitalize() + "."
        elif len(actions) == 2:
            s2 = actions[0].capitalize() + " and " + actions[1] + "."
        else:
            s2 = actions[0].capitalize() + ", " + actions[1] + ", and " + actions[2] + "."

    out = s1
    if s2:
        out += " " + s2
    out += " That's all I saw at that moment."
    return re.sub(r"\s{2,}", " ", out).strip()

def _extract_last_distance_from_history(message_history) -> str | None:
    """Return the most recent distance (in metres) the assistant has already stated.

    Used when handling distance-contradiction challenges — the assistant needs to quote
    back the same figure it gave earlier rather than extracting a fresh one from facts.
    """
    if not message_history:
        return None
    dist_pat = re.compile(r"\babout\s+(\d{1,4})\s*(metres|meters|m)\b|\b(\d{1,4})\s*(metres|meters|m)\b", re.IGNORECASE)
    for m in reversed(message_history):
        if m.get("role") != "assistant":
            continue
        txt = (m.get("content") or "")
        mm = dist_pat.search(txt)
        if mm:
            num = mm.group(1) or mm.group(3)
            unit = mm.group(2) or mm.group(4)
            unit_norm = "metres" if unit and unit.lower().startswith(("m", "meter")) else unit
            return f"{num} {unit_norm}".strip()
    return None

def _assistant_mentioned_pass_by(message_history) -> bool:
    """Return True if the assistant has previously described the witness passing by a premises.

    Used together with distance information to resolve contradiction challenges like
    "But you said you walked past the shop — if you were 50 metres away, how did you see it?"
    """
    if not message_history:
        return False
    pat = re.compile(r"\b(passed\s+by|walked\s+past|went\s+past)\b.*\b(shop|store|bottle\s*store|premises)\b", re.IGNORECASE)
    for m in reversed(message_history):
        if m.get("role") != "assistant":
            continue
        if pat.search(m.get("content") or ""):
            return True
    return False

def _is_distance_contradiction_challenge(text: str) -> bool:
    """Return True if the interviewer is challenging the witness by citing their own earlier distance claim.

    Example: "But you said you were 50 metres away — you can't have seen that."
    Detected by requiring both a prior-statement reference ("you've said", "you said") AND
    a metric distance figure in the same question.
    """
    if not text:
        return False
    t = text.strip().lower()
    if "?" not in t:
        return False
    return bool(
        re.search(r"\b(but|why)\b.*\b(you('?| )?ve|you\s+have|you)\s+(just\s+)?said\b.*\b(pass(?:ed)?\s+by|walk(?:ed)?\s+past|went\s+past)\b", t)
        and re.search(r"\b(\d{1,4})\s*(metres|meters|m)\b", t)
    )

def _extract_drive_direction(facts_text: str, facts_low: str) -> str | None:
    m = re.search(r"\b(drove|drove\s+off|drove\s+away|headed|went)\s+(towards|to|away\s+from)\s+([a-z' \-]+)", facts_low)
    if m:
        phrase = (m.group(2) + " " + m.group(3)).strip()
        return re.sub(r"\s+", " ", phrase).rstrip(" .")
    m2 = re.search(r"\b(drove|headed|went)\s+(north|south|east|west)\b", facts_low)
    if m2:
        return m2.group(2)
    return None

def _looks_like_question(text: str) -> bool:
    """Return True if the message appears to be a question, whether or not it includes a "?".

    Interviewers often omit question marks in transcribed speech, so keywords like "tell me"
    and "describe" are treated as implicit question indicators.
    """
    t = (text or "").lower().strip()
    if not t:
        return False
    if "?" in t:
        return True
    keywords = [
        "tell me", "what happened", "what did you see", "what did you hear",
        "what ", "when ", "where ", "who ", "how ", "why ", "which ",
        "can you", "could you", "would you", "please describe", "walk me through",
        "explain", "describe", "witness"
    ]
    return any(k in t for k in keywords)

def _extract_age_from_facts(facts_low: str) -> str | None:
    """Extract an age or age range for the observed person and format it as a natural phrase.

    Handles exact ages ("aged 25"), qualified ages ("about 30"), ranges ("mid-20s"),
    decade only ("20s"), and written forms ("thirties"). Returns a capitalised phrase
    like "About 25." or "Mid-20s." or None if no age data is present.
    """
    if not facts_low:
        return None
    # Exact age with optional qualifier ("aged around 25", "approximately 25 years old", etc.)
    m = re.search(r"\baged?\s+(?:around\s+|about\s+|approximately\s+|roughly\s+)?(\d{1,3})\b", facts_low)
    if not m:
        m = re.search(r"\b(?:around\s+|about\s+|approximately\s+|roughly\s+)?(\d{1,3})\s*(?:years?\s+old)\b", facts_low)
    if m:
        n = m.group(1)
        return f"About {n}."

    # Decade range with qualifier: "mid-20s", "early 30s", "late teens"
    m = re.search(r"\b(?:and\s+)?(?:aged?s?\s+)?(early|mid|late)[\s\-]*(\d{2})(?:'?s)?\b", facts_low)
    if m:
        qualifier = m.group(1).capitalize()
        decade = m.group(2)
        return f"{qualifier}-{decade}s."

    # Bare decade: "20s" or "30's" — less precise than mid/early/late qualifiers above
    m = re.search(r"\b(\d{2})(?:'?s)\b", facts_low)
    if m:
        return f"In his {m.group(1)}s."

    # Written decade forms: "twenties", "thirties", "teens"
    word_ages = {
        "teens": "In his teens.",
        "twenties": "In his twenties.",
        "thirties": "In his thirties.",
        "forties": "In his forties.",
        "fifties": "In his fifties.",
        "sixties": "In his sixties."
    }
    for word, response in word_ages.items():
        if re.search(rf"\b{word}\b", facts_low):
            return response

    return None

def _extract_weapons_only(facts_text: str) -> str | None:
    """
    Scenario-agnostic extraction of WEAPONS ONLY (Issue #3 fix: exclude bags/backpacks).
    Returns a natural language phrase of weapons only (e.g., "A knife.").
    """
    if not facts_text:
        return None

    facts_low = facts_text.lower()
    weapons = []

    weapon_patterns = [
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(knife|machete|blade)\b", "knife"),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(gun|pistol|firearm|rifle|shotgun)\b", "gun"),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(bat|club|stick|pipe)\b", "bat"),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(crowbar|jemmy|pry\s*bar)\b", "crowbar"),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(axe|hatchet)\b", "axe"),
    ]

    for pattern, generic_name in weapon_patterns:
        m = re.search(pattern, facts_low)
        if m and generic_name:
            if not any(generic_name in item.lower() for item in weapons):
                weapons.append(f"a {generic_name}")

    if not weapons:
        return None

    # Join weapons naturally
    if len(weapons) == 1:
        return weapons[0].capitalize() + "."
    elif len(weapons) == 2:
        return f"{weapons[0].capitalize()} and {weapons[1]}."
    else:
        return f"{', '.join([w.capitalize() if i == 0 else w for i, w in enumerate(weapons[:-1])])}, and {weapons[-1]}."

def _extract_carried_items(facts_text: str) -> str | None:
    """
    Scenario-agnostic extraction of items being carried/held by the observed person.
    Returns a natural language phrase combining all found items (e.g., "A knife and a small backpack.").
    """
    if not facts_text:
        return None

    facts_low = facts_text.lower()
    items = []

    # Weapons
    weapon_patterns = [
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(knife|machete|blade)\b", "knife"),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(gun|pistol|firearm|rifle|shotgun)\b", "gun"),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(bat|club|stick|pipe)\b", "bat"),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(crowbar|jemmy|pry\s*bar)\b", "crowbar"),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(axe|hatchet)\b", "axe"),
    ]

    # Containers/bags
    bag_patterns = [
        (r"\b(carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(backpack|rucksack)\b", None),  # None = extract full phrase
        (r"\b(carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(bag|duffel|suitcase|holdall)\b", None),
        (r"\b(carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(briefcase|satchel)\b", None),
    ]

    # Other items
    other_patterns = [
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(torch|flashlight)\b", None),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(phone|mobile)\b", None),
        (r"\b(holding|carried|carrying|had|with)\s+(?:a\s+)?(\w+\s+)?(can|container|bottle)\b", None),
    ]

    # Extract weapons with generic names
    for pattern, generic_name in weapon_patterns:
        m = re.search(pattern, facts_low)
        if m and generic_name:
            # Check if we already have this type
            if not any(generic_name in item.lower() for item in items):
                items.append(f"a {generic_name}")

    # Extract containers/bags with full descriptive phrase
    for pattern, _ in bag_patterns:
        m = re.search(pattern, facts_low)
        if m:
            # Extract descriptor (group 2) and item type (group 3) from the match
            descriptor = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else ""
            item_type = m.group(3).lower() if m.lastindex >= 3 else None
            if item_type:
                full_item = f"{descriptor} {item_type}".strip() if descriptor else item_type
                if not any(item_type in item.lower() for item in items):
                    items.append(f"a {full_item}")

    # Extract other items
    for pattern, _ in other_patterns:
        m = re.search(pattern, facts_low)
        if m:
            item_name = m.group(3) if m.lastindex >= 3 else None
            if item_name and not any(item_name in item.lower() for item in items):
                items.append(f"a {item_name}")

    if not items:
        return None

    # Join items naturally
    if len(items) == 1:
        return items[0].capitalize() + "."
    elif len(items) == 2:
        return f"{items[0].capitalize()} and {items[1]}."
    else:
        # Oxford comma for 3+
        all_but_last = ", ".join(items[:-1])
        return f"{all_but_last.capitalize()}, and {items[-1]}."

def _extract_witness_steps(facts_low: str) -> list[str]:
    """Extract the ordered list of actions the witness took after the incident.

    Returns canonical short sentences in source order (e.g., "I stopped to investigate.",
    "I called police.", "I went home."). These are used by the "what did you do next?"
    deterministic handler to advance through the witness timeline one step at a time,
    preventing the LLM from either skipping steps or repeating steps already covered.
    """
    steps = []
    if re.search(r"\bstopped?\s+to\s+(investigate|check)", facts_low):
        # Distinguish "stopped to investigate the fire" from a more general investigation.
        if re.search(r"\b(fire|smoke|burning)\b", facts_low):
            steps.append("I stopped to investigate the fire.")
        else:
            steps.append("I stopped to investigate.")

    if re.search(r"\bput\s+(it|the\s+fire)\s+out\b", facts_low):
        steps.append("I put the fire out.")

    if re.search(r"\b(got\s+into\s+my\s+car|drove\s+along|drove\s+down|looked\s+for|went\s+looking\s+for)\b", facts_low):
        steps.append("I went looking for the person.")

    if re.search(r"\bstopped?\s+(?:a\s+)?(pedestrian|witness|passer-?by|person)\b", facts_low) and \
       re.search(r"\b(said|told|pointed|directed|indicated)\b", facts_low):
        steps.append("I stopped a pedestrian who gave me information.")

    # Stepped onto a vantage point (deck, porch, balcony) to get a better look.
    if re.search(r"\bi\s+(stepped|went)\s+(?:out\s+)?(?:onto|to|on)\s+(?:my\s+)?(deck|porch|balcony|garden|yard)\b", facts_low):
        location_match = re.search(r"\bi\s+(stepped|went)\s+(?:out\s+)?(?:onto|to|on)\s+(?:my\s+)?(deck|porch|balcony|garden|yard)\b", facts_low)
        if location_match:
            location = location_match.group(2)
            steps.append(f"I stepped onto my {location}.")

    if re.search(r"\bi\s+followed\s+(?:out|outside|him|her|them)", facts_low):
        steps.append("I went outside to see where he went.")

    # [^\n,.] captures the full destination phrase rather than stopping at the first comma.
    # Facts are newline-separated, so \n is a safe sentence boundary here.
    evac_match = re.search(r"\b(we\s+)?evacuated\s+to\s+(the\s+)?([^\n,.]+)", facts_low)
    if evac_match:
        location = evac_match.group(3).strip()
        steps.append(f"We evacuated to the {location}.")

    # "I went back" only — don't match if it's the suspect or someone else who returned.
    if re.search(r"\bi\s+(went\s+back|returned)\b", facts_low):
        if re.search(r"\b(fire|check\s+it\s+was\s+out)\b", facts_low):
            steps.append("I went back to check the fire was out.")
        else:
            steps.append("I went back to the spot to make sure things were okay.")

    if re.search(r"\b(ran|went)\s+inside\b", facts_low):
        if re.search(r"\b(told|informed|notified|alerted)\s+(the\s+)?(staff|supermarket\s+staff|store\s+staff|shop\s+staff)\b", facts_low):
            steps.append("I ran inside and told the staff.")
        else:
            steps.append("I went inside.")

    if re.search(r"\bcalled?\s+(111|105)\b|\bcall\s+(111|105)\b", facts_low):
        steps.append("I called police.")
    if re.search(r"\b(checked\s+on|went\s+to\s+check\s+on|spoke\s+to|talked\s+to)\s+(the\s+)?(owner|staff|clerk|victim|person|employee)\b", facts_low):
        steps.append("I checked on the victim.")
    if re.search(r"\bwaited?\s+(?:until|for)\s+(?:the\s+)?(police|ambulance|officer)", facts_low):
        steps.append("I waited for the police to arrive.")
    if re.search(r"\b(gave|provided|left)\s+(my\s+)?details\b|\breassured\b.*\bpolice\b", facts_low):
        steps.append("I left my details.")
    if re.search(r"\b(did\s+my\s+shopping|continued\s+shopping|went\s+shopping|carried\s+on\s+shopping|finished\s+my\s+shopping)\b", facts_low):
        steps.append("I did my shopping.")
    if re.search(r"\b(had\s+to|needed\s+to)\s+(get\s+home|go\s+home|leave)\b", facts_low):
        steps.append("I had to get home.")
    if re.search(r"\b(cook|prepare\s+dinner|make\s+dinner|feed\s+the\s+kids|feed\s+my\s+family)\b", facts_low):
        steps.append("I had to cook dinner.")
    if re.search(r"\b(went\s+home|left\s+the\s+area|headed\s+home)\b", facts_low):
        steps.append("I went home.")
    out, seen = [], set()
    for s in steps:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def _history_completed_steps(message_history: list[dict]) -> list[str]:
    """Return the subset of canonical witness-step sentences already mentioned by the assistant.

    Cross-references _extract_witness_steps() output against the conversation history so the
    "what did you do next?" handler can skip steps the witness has already described. Short
    "Yes." affirmations to direct questions are also matched so implicit confirmations count.
    Multi-sentence summary responses (the incident opener) are excluded from step-pattern
    matching — they mention many steps in one go, which would falsely mark all as completed
    before the interviewer has had a chance to ask about each one individually.
    """
    if not message_history:
        return []
    patt_map = {
        "I stopped to investigate the fire.": re.compile(r"\bi\s+stopped\s+to\s+(investigate|check|put\s+out).*\b(fire|smoke|burning)\b", re.IGNORECASE),
        "I stopped to investigate.": re.compile(r"\bi\s+stopped\s+to\s+(investigate|check)\b", re.IGNORECASE),
        "I put the fire out.": re.compile(r"\bi\s+(?:stopped\s+to\s+)?put\s+(?:out\s+)?(?:it|the\s+fire)\s+out\b|\bi\s+extinguished\b|\bput\s+out\s+the\s+fire\b", re.IGNORECASE),
        "I went looking for the person.": re.compile(r"\bi\s+went\s+looking\s+for\b|\bi\s+(drove|went)\s+(along|down)\b.*\blooking\b", re.IGNORECASE),
        "I stopped a pedestrian who gave me information.": re.compile(r"\bi\s+(?:stopped|found|spoke\s+to|talked\s+to)\s+(?:a\s+)?(pedestrian|witness|passer-?by|person)\b", re.IGNORECASE),
        # The same pattern covers all vantage-point locations; canonical key differs per location.
        "I stepped onto my deck.": re.compile(r"\bi\s+(stepped|went)\s+(?:out\s+)?(?:onto|to|on)\s+(?:my\s+)?(deck|porch|balcony|garden|yard)\b", re.IGNORECASE),
        "I stepped onto my porch.": re.compile(r"\bi\s+(stepped|went)\s+(?:out\s+)?(?:onto|to|on)\s+(?:my\s+)?(deck|porch|balcony|garden|yard)\b", re.IGNORECASE),
        "I stepped onto my balcony.": re.compile(r"\bi\s+(stepped|went)\s+(?:out\s+)?(?:onto|to|on)\s+(?:my\s+)?(deck|porch|balcony|garden|yard)\b", re.IGNORECASE),
        "I stepped onto my garden.": re.compile(r"\bi\s+(stepped|went)\s+(?:out\s+)?(?:onto|to|on)\s+(?:my\s+)?(deck|porch|balcony|garden|yard)\b", re.IGNORECASE),
        "I stepped onto my yard.": re.compile(r"\bi\s+(stepped|went)\s+(?:out\s+)?(?:onto|to|on)\s+(?:my\s+)?(deck|porch|balcony|garden|yard)\b", re.IGNORECASE),
        "I went outside to see where he went.": re.compile(r"\bi\s+(?:followed|went)\s+(?:out|outside)(?:\s+to\s+see\s+where)?\b", re.IGNORECASE),
        "I went back to check the fire was out.": re.compile(r"\bi\s+(went\s+back|returned).*\b(fire|check.*out)\b", re.IGNORECASE),
        "I went back to the spot to make sure things were okay.": re.compile(r"\bi\s+(went\s+back|returned)\b", re.IGNORECASE),
        "I ran inside and told the staff.": re.compile(r"\bi\s+(ran|went)\s+(?:inside|into\s+the\s+\w+)\s+(?:and\s+)?(?:to\s+)?(?:tell|told|inform|informed|notify|notified|alert|alerted|report)\s+(?:the\s+)?(?:staff|incident)\b", re.IGNORECASE),
        "I called police.": re.compile(r"\bi\s+(call|called|dialed|phoned|rang)\s+(111|105|police)\b", re.IGNORECASE),
        "I checked on the victim.": re.compile(r"\bi\s+(checked\s+on|spoke\s+to|talked\s+to)\s+(the\s+)?(owner|staff|clerk|victim|person|employee)\b", re.IGNORECASE),
        "I waited for the police to arrive.": re.compile(r"\bi\s+waited?\s+(?:until|for)\s+(?:the\s+)?(police|ambulance|officer)(?:\s+(?:to\s+)?arrive)?", re.IGNORECASE),
        "I left my details.": re.compile(r"\bi\s+(gave|provided|left)\s+(my\s+)?details\b", re.IGNORECASE),
        "I did my shopping.": re.compile(r"\bi\s+(did|continued|finished|carried\s+on)\s+(?:my\s+)?shopping\b", re.IGNORECASE),
        "I had to get home.": re.compile(r"\bi\s+(had\s+to|needed\s+to)\s+(get\s+home|go\s+home|leave)\b", re.IGNORECASE),
        "I had to cook dinner.": re.compile(r"\bi\s+(had\s+to|needed\s+to)\s+(cook|prepare\s+dinner|make\s+dinner)\b", re.IGNORECASE),
        "I went home.": re.compile(r"\bi\s+(went|headed)\s+home\b|\bi\s+left\s+the\s+area\b", re.IGNORECASE),
    }
    said = []
    for i, m in enumerate(message_history):
        if m.get("role") != "assistant":
            continue
        txt = m.get("content") or ""

        # Skip comprehensive opener summaries for step-pattern matching.
        # Openers summarise all events broadly (long, multi-sentence), so any individual
        # step they mention would be falsely marked as "already covered" before the
        # interviewer has a chance to ask about it directly.
        # Direct answers to "what did you do next?" are always short (single sentence).
        is_summary = len(txt) > 200 and len(re.findall(r'[.!?]', txt)) >= 3
        if not is_summary:
            for canon, patt in patt_map.items():
                if canon not in said and patt.search(txt):
                    said.append(canon)

        # A bare "Yes." reply to a direct "Did you…?" question implicitly confirms that step.
        if txt.strip().lower() in ["yes.", "yes"]:
            if i > 0 and message_history[i-1].get("role") == "user":
                prev_q = message_history[i-1].get("content", "").lower()
                if re.search(r"\bdid\s+you\s+follow\s+(?:him|her|them|the\s+\w+)\s+(?:outside|out)", prev_q):
                    if "I went outside to see where he went." not in said:
                        said.append("I went outside to see where he went.")
                if re.search(r"\bdid\s+you\s+wait\s+(?:for|until)\s+(?:the\s+)?(police|ambulance|officer)", prev_q):
                    if "I waited for the police to arrive." not in said:
                        said.append("I waited for the police to arrive.")
                if re.search(r"\bdid\s+you\s+call\s+(?:111|105|(?:the\s+)?police)", prev_q):
                    if "I called police." not in said:
                        said.append("I called police.")
    return said

def _extract_suspect_steps(facts_text: str) -> list[str]:
    """Extract the ordered list of movement steps the suspect made after the incident.

    Returns canonical third-person sentences in source order, e.g.:
      "He went down Philip's Lane."
      "He got into a light grey Mazda2 without plates."
    Used by the "did you see where they went?" handler to reveal steps one at a time,
    matching the same pattern as _extract_witness_steps() for the witness's own actions.
    """
    if not facts_text:
        return []
    low = facts_text.lower()
    lines = [ln.strip() for ln in facts_text.splitlines() if ln.strip()]
    out = []

    # Local helper — extracts a named street or lane from a single fact line.
    def _route_from_line(ln: str) -> str | None:
        m1 = re.search(r"\b([A-Z][A-Za-z'’\-]+)'s\s+(lane|alley|alleyway|path|track)\b", ln)
        if m1:
            return f"{m1.group(1).title()}'s {m1.group(2).title()}"
        m2 = re.search(r"\b([A-Z][A-Za-z'’\-]+)\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Way)\b", ln)
        if m2:
            return f"{m2.group(1).title()} {m2.group(2).title()}"
        return None

    # Pre-extract the vehicle phrase across all facts; per-line extraction below may override it
    # when a more specific colour/model appears on the same line as the vehicle event.
    vehicle_phrase_global = _vehicle_details_phrase(low)

    for ln in lines:
        lnl = ln.lower()

        # Ran/went down a lane/route
        if re.search(r"\b(ran|went|headed|continued)\s+down\b", lnl):
            route = _route_from_line(ln) or _extract_named_route(ln)
            if route:
                out.append(f"He went down {route}.")
            else:
                out.append("He went down the road.")

        # Entered a street/area
        if re.search(r"\b(into|onto|to)\s+[A-Z][A-Za-z'’\-]+\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Way)\b", ln):
            m = re.search(r"\b([A-Z][A-Za-z'’\-]+)\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Way)\b", ln)
            if m:
                out.append(f"He went into {m.group(1).title()} {m.group(2).title()}.")

        # Emerged into / got into a vehicle — prefer the per-line extraction so that a
        # vehicle described on the same line as the event takes precedence.
        if re.search(r"\b(emerged|came\s+out)\b", lnl) or re.search(r"\b(got\s+into|got\s+in)\b\s+(a|the)\s+(car|vehicle|van|ute|truck|hatch|sedan|motorbike|scooter)\b", lnl):
            vphrase_line = _vehicle_details_phrase(ln.lower())
            vphrase = vphrase_line or vehicle_phrase_global
            if vphrase:
                out.append(f"He got into {vphrase}.")

        # Fled/left
        if re.search(r"\b(fled|left|took\s+off|drove\s+away|drove\s+off)\b", lnl):
            out.append("He left the area.")

    # Deduplicate preserving order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq


def _history_completed_suspect_steps(message_history: list[dict]) -> list[str]:
    """Return the subset of canonical suspect-step sentences already stated by the assistant.

    The matching patterns are intentionally flexible (e.g., "got into" or "jumped into" both
    mark the vehicle step as done) to avoid replaying a step the witness already implied.
    Generic canonical keys ("He got into a vehicle.") are used when the exact model/colour
    cannot be recovered from the assistant's phrasing without re-parsing.
    """
    if not message_history:
        return []
    said = []
    # Patterns aligned to the canonical sentences produced in _extract_suspect_steps
    patt_map = {
        "He went down the road.": re.compile(r"\bhe\s+went\s+down\s+the\s+road\b", re.IGNORECASE),
        "He left the area.": re.compile(r"\bhe\s+left\s+the\s+area\b", re.IGNORECASE),
    }

    route_pat = re.compile(r"\b(?:he|she|they)\s+(?:went|ran)\s+down\s+([A-Z][A-Za-z''\-]+(?:'s)?\s+(?:Lane|Alley|Alleyway|Path|Track|Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Way))", re.IGNORECASE)
    # All common vehicle-entry phrasings map to the same "got into a vehicle" canonical step.
    vehicle_pat = re.compile(r"\b(?:he|she|they)\s+(?:got\s+into|got\s+in|emerged\s+in|was\s+in|jumped\s+into|climbed\s+into)\s+(?:a|an|the)\s+", re.IGNORECASE)
    drove_away_pat = re.compile(r"\b(?:he|she|they|it)\s+(?:drove|sped|took\s+off)\s+(?:away|off)\b", re.IGNORECASE)

    for m in message_history:
        if m.get("role") != "assistant":
            continue
        txt = m.get("content") or ""
        for canon, patt in patt_map.items():
            if canon not in said and patt.search(txt):
                said.append(canon)
        if route_pat.search(txt):
            # canonicalise to the exact sentence from the text
            match = route_pat.search(txt)
            route = match.group(1) if match else None
            if route:
                canon = f"He went down {route}."
                if canon not in said:
                    said.append(canon)
        if vehicle_pat.search(txt):
            # The exact vehicle description varies — normalise to the generic canonical key.
            canon = "He got into a vehicle."
            if canon not in said:
                said.append(canon)
        if drove_away_pat.search(txt):
            # Mark that driving away has been mentioned
            canon = "He drove away."
            if canon not in said:
                said.append(canon)
    return said


def _extract_chronological_facts(persona_facts_text: str) -> list[str]:
    """Extract all observable facts from the persona in source (chronological) order.

    Converts bullet-point facts into normalised first-person statements by substituting
    "you/your" pronouns with "I/my". Pure context facts (setting the scene without any
    observation or action) and physical description facts are excluded because they are
    not timeline events — they have their own dedicated question handlers.

    Works with any scenario type without hardcoding incident-specific terms.
    """
    if not persona_facts_text:
        return []

    timeline = []
    lines = persona_facts_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line or not line.startswith('-'):
            continue

        # Strip the bullet dash and any trailing certainty annotation from the raw fact text.
        fact = re.sub(r'^\s*-\s*', '', line)
        fact = re.sub(r'\s*\(Certainty:.*?\)\s*$', '', fact, flags=re.IGNORECASE).strip()

        if fact.lower() == "none":
            continue

        fact_low = fact.lower()

        # Pure context facts (location/time without any observation verb) set the scene
        # but are not interview-relevant events, so skip them here.
        is_pure_context = False
        if re.match(r'^\s*(you were|i was|at about|at around|around)\s+(walking|standing|at|in|on|near|about|driving)', fact_low):
            if not re.search(r'\b(saw|heard|noticed|smelled|felt|argued|punched|ran|went|stopped|called|checked|followed|entered|exited|spoke|pushed|used|evacuated|stepped|left|gave|provided)\b', fact_low):
                is_pure_context = True
        if is_pure_context:
            continue

        # Physical description facts ("He was white, shaved head…") are handled by their
        # own deterministic question handlers, not the timeline stepper.
        # Exception: weapon possession ("He had a knife") is an observable event.
        is_description = False
        if re.match(r'^\s*(he|she|they)\s+(was|were|had)\s+(a|an|white|black|brown|tall|short|wearing|aged?|about)', fact_low):
            if not re.search(r'\b(knife|gun|weapon|crowbar|bat)\b', fact_low):
                is_description = True
        if is_description:
            continue

        # Normalise second-person pronouns to first-person — persona facts are sometimes
        # authored in "you" voice ("You saw the man run in") to address the witness directly.
        normalized = fact

        normalized = re.sub(r'^You were\s+', 'I was ', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'^You\s+(saw|heard|stopped|called|went|followed|didn\'t|had|got|drove|walked|ran|entered|exited|spoke|checked|left|gave|provided|noticed|felt|smelled|stepped|used|evacuated|pushed|put|couldn\'t|did)', r'I \1', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\band you\s+(saw|heard|stopped|called|went|followed|didn\'t|had|got|drove|walked|ran|entered|exited|spoke|checked|left|gave|provided|noticed|felt|smelled|stepped|used|evacuated|pushed|put|couldn\'t|did)', r'and I \1', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'([,\.])\s+you\s+(saw|heard|stopped|called|went|followed|didn\'t|had|got|drove|walked|ran|entered|exited|spoke|checked|left|gave|provided|noticed|felt|smelled|stepped|used|evacuated|pushed|put|couldn\'t|did)', r'\1 I \2', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\b[Aa]s you\s+', 'As I ', normalized)
        normalized = re.sub(r'\byour\s+', 'my ', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\b(towards?|to|at|past|for|with|by|near)\s+you\b', r'\1 me', normalized, flags=re.IGNORECASE)

        # "I was walking home when I saw smoke" → extract just the observation ("I saw smoke")
        # so the timeline step is a discrete event, not a contextual sentence.
        if 'when' in fact_low and ('saw' in fact_low or 'heard' in fact_low):
            parts = re.split(r'\s+when\s+', normalized, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                after = parts[1]
                if re.search(r'\b(saw|heard)\b', after, re.IGNORECASE):
                    normalized = after.strip()
                    if normalized:
                        normalized = normalized[0].upper() + normalized[1:]

        # Ensure it ends with a period
        if normalized and not normalized.endswith(('.', '!', '?')):
            normalized += '.'

        timeline.append(normalized)

    seen = set()
    unique_timeline = []
    for fact in timeline:
        if fact and fact not in seen:
            unique_timeline.append(fact)
            seen.add(fact)

    return unique_timeline


def _history_said_fact(message_history: list[dict], fact: str) -> bool:
    """Return True if the essence of a given fact has already appeared in the assistant's replies.

    Used by _next_timeline_step() to skip facts the witness has already shared, even if the
    exact wording differs. "Ran past me" is treated as covering "ran out of store" because
    the former logically implies the latter.
    Delegates to the specialised step-completion helpers for canonical witness/suspect steps
    so the matching logic does not need to be duplicated here.
    """
    if not message_history:
        return False

    fact_low = fact.lower()
    # Extract key phrases from the fact
    key_phrases = []

    # Observation facts
    # Match "saw [someone/a man/a woman/a person] [action verb] into/in"
    # Also match when "saw" is implicit: "a man ran into" means "I saw a man run into"
    if re.search(r"\bsaw\s+(?:a\s+)?(?:man|woman|person|someone)", fact_low) and "into" in fact_low:
        # Pattern 1: Explicit "saw/see" - "I saw a man run into"
        key_phrases.append(r"\b(saw|see)\s+(?:a\s+)?(?:\w+\s+)?(?:man|person|someone|woman|him|her)\s+(?:run|ran|walk|walked|go|went|enter|entered|step|stepped|burst)\s+(?:in|into)")
        # Pattern 2: Implicit observation - "a man ran into" (from opener where "saw" is implicit)
        key_phrases.append(r"\b(?:a|the)\s+(?:\w+\s+)?(?:man|person|woman)\s+(?:ran|walked|went|entered|stepped|burst)\s+(?:in|into)")
    # Match "saw [someone/him/her/them/the man] [action verb] out"
    if re.search(r"\bsaw\s+(?:a\s+)?(?:the\s+)?(?:man|woman|person|someone|him|her|them)", fact_low) and "out" in fact_low:
        # Pattern 1: Explicit "saw/see" - "I saw the man run out"
        key_phrases.append(r"\b(saw|see|the\s+man\s+then)\s+(?:a\s+|the\s+)?(?:\w+\s+)?(?:man|person|him|her|them)?\s*(?:run|ran|walk|walked|come|came|go|went|exit|exited)\s+out")
        # Pattern 2: Implicit observation - "the man ran out" or "he ran out" (from opener)
        key_phrases.append(r"\b(?:the|a|he|she|they)\s+(?:\w+\s+)?(?:man|person|woman)?\s*(?:ran|walked|came|went|exited)\s+out")
        # Pattern 3: Indirect mention - if "ran past" was mentioned, "run out of store" is implied
        # because you can't run past someone without first exiting the store
        key_phrases.append(r"\b(?:the\s+)?(?:man|person|he|she|they)\s+(?:ran|walked|went)\s+past\s+(?:me|on)")
    # Also detect "ran past" as covering the "run out" fact
    if re.search(r"\brun\s+out\s+of\s+(?:the\s+)?(?:store|shop|building)", fact_low):
        key_phrases.append(r"\b(?:the\s+)?(?:man|person|he|she|they)\s+(?:ran|walked|went)\s+past\s+(?:me|on)")
    if "heard" in fact_low:
        # Be flexible: "screaming", "shouting", "yelling" all indicate audible disturbance
        if "screaming" in fact_low or "shouting" in fact_low or "glass" in fact_low:
            key_phrases.append(r"\bheard\s+(?:\w+\s+)*(screaming|shouting|yelling|noises?|sounds?|glass)")
    if "ran past" in fact_low or "passed me" in fact_low:
        key_phrases.append(r"\b(ran|walked|went|passed)\s+past\s+(?:me)?")
    if "took his" in fact_low and "off" in fact_low:
        # Extract item name (mask, hat, jacket, etc.)
        item_match = re.search(r"took his (\w+) off", fact_low)
        if item_match:
            item = item_match.group(1)
            key_phrases.append(rf"\btook\s+(?:his|her|their|the)\s+{item}\s+off")

    # Check suspect/witness steps using existing helpers
    if fact.startswith("He went down") or fact.startswith("He got into") or fact.startswith("He left"):
        return fact in _history_completed_suspect_steps(message_history)
    if fact.startswith("I called") or fact.startswith("I checked") or fact.startswith("I left my") or fact.startswith("I had to") or fact.startswith("I went"):
        return fact in _history_completed_steps(message_history)

    # Check for vehicle-related facts (emerged in, got into, drove away)
    # These might not start with "He" but contain vehicle references
    if re.search(r"\b(emerged|got)\s+(in|into)\s+(?:a|an|the)\s+", fact_low):
        completed = _history_completed_suspect_steps(message_history)
        if "He got into a vehicle." in completed:
            return True
    if re.search(r"\bdrove\s+away\b", fact_low):
        completed = _history_completed_suspect_steps(message_history)
        if "He drove away." in completed:
            return True
    if re.search(r"\b(?:went|ran)\s+down\s+[A-Z]", fact):
        completed = _history_completed_suspect_steps(message_history)
        # Check if any route was mentioned
        for step in completed:
            if step.startswith("He went down"):
                return True

    # Check if any key phrase appears in message history
    for m in message_history:
        if m.get("role") != "assistant":
            continue
        content = (m.get("content") or "").lower()
        for pattern in key_phrases:
            if re.search(pattern, content):
                return True

    return False


def _next_timeline_step(message_history: list[dict], persona_facts_text: str) -> str | None:
    """Return the next unseen fact from the chronological timeline, or None if all have been covered.

    Called by the "what happened next?" handler to advance the conversation step-by-step
    through the witness's account without the LLM skipping ahead or repeating itself.
    """
    if not persona_facts_text:
        return None

    all_facts = _extract_chronological_facts(persona_facts_text)

    for fact in all_facts:
        if not _history_said_fact(message_history, fact):
            return fact

    return None

def ask_ollama(message_history, persona_data):
    """Generate a response for the most recent interviewer message, using the three-tier strategy.

    Tier 1 — Deterministic handlers: structured questions about facts (distance, time, clothing,
      vehicle, identity fields, timeline steps) are answered directly from the persona JSON
      without any LLM call. Responses are exact and cannot hallucinate.

    Tier 2 — Constrained micro-LLM: the first open-ended incident question and occupation/
      job-enjoyment questions use a tightly scoped prompt with specific rules and length limits.

    Tier 3 — Full LLM: all other questions are sent to Ollama with the full system context,
      then post-processed through multiple guardrail layers to remove artifacts, fix attribution
      errors, enforce NZ English, and prevent hallucination.

    Persona metadata (voice model, speaking tone, etc.) is stripped before being passed to
    the LLM to avoid the model breaking character by referencing its own configuration.

    'interview_instructions' is intentionally excluded from the allowlist below — it is
    guidance for the human interviewer only and must never be sent to the LLM. If the LLM
    received it, the persona could inadvertently reveal what the interviewer is supposed to
    ask or probe, breaking the realism of the exercise.
    """
    # Allowlist approach: only keys listed here are forwarded to the LLM.
    # Any field not in this list (including 'interview_instructions', voice/audio config,
    # and any future persona fields) is silently dropped before the prompt is built.
    allowed_keys = [
        "full_name", "persona_type", "persona_prompt", "facts_to_provide",
        "date_of_birth", "home_address", "business_address", "employed_by",
        "occupation", "home_phone", "work_phone", "cell_phone",
        "ethnicity", "gender", "email", "social_networking",
        "hidden_motivations", "trigger_topics"
    ]
    filtered_persona = {k: v for k, v in persona_data.items() if k in allowed_keys}

    META_KEYS = {"speaking_tone","speaking_speed","speaking_accent","vocal_quirks","interaction_style","voice_id","persona_voice_model","persona_voice_speaker_id","preferences","internal_notes"}

    IMPORTANT_KEYS = [
        "full_name","date_of_birth","home_address","business_address",
        "employed_by","occupation","ethnicity","gender","email","social_networking"
    ]

    def _clean_value(v: str) -> str:
        """Return an empty string for unfilled template placeholders like "(insert name)"."""
        if not isinstance(v, str):
            return v
        s = v.strip()
        if s.lower().startswith("(insert ") or "insert" in s.lower():
            return ""
        return s

    def sanitize_persona(p: dict) -> dict:
        """Remove metadata keys and empty/placeholder values before sending to the LLM.

        Metadata fields (voice model, speaking tone, etc.) must not reach the LLM —
        the model will recite them as character traits ("I speak with a nervous tone"),
        which breaks immersion.
        """
        q = {}
        for k, v in p.items():
            if k in META_KEYS:
                continue
            if v is None:
                continue
            if isinstance(v, str) and not _clean_value(v):
                continue
            q[k] = _clean_value(v) if isinstance(v, str) else v
        return q

    def persona_background_message(p: dict) -> dict:
        """Build the system message that provides personal identity details to the LLM.

        Only the most important identity fields are listed here (name, DOB, address, etc.).
        The instruction "do not recite unless asked" prevents the model from volunteering
        personal information unprompted, which would feel unrealistic in an interview.
        """
        p2 = sanitize_persona(p)
        lines = []
        for k in IMPORTANT_KEYS:
            val = p2.get(k)
            if isinstance(val, str) and val.strip():
                label = k.replace('_', ' ').title()
                lines.append(f"{label}: {val}")
        return {
            "role": "system",
            "content": (
                "BACKGROUND (for staying in character only; do not recite unless asked):\n" +
                ("\n".join(lines) if lines else "(none)") +
                "\nOnly reveal background details if directly asked."
            )
        }

    persona_prompt = filtered_persona.get("persona_prompt", "")

    persona_facts = filtered_persona.get("facts_to_provide", [])
    facts = ""
    for f in persona_facts:
        fact_text = f.get("fact", "").strip()
        certainty = f.get("certainty", "").strip()
        reason = f.get("reason", "").strip()
        if fact_text:
            if certainty and reason:
                facts += f"- {fact_text} (Certainty: {certainty}, Reason: {reason})\n"
            elif certainty:
                facts += f"- {fact_text} (Certainty: {certainty})\n"
            else:
                facts += f"- {fact_text}\n"

    hidden_motivations = filtered_persona.get("hidden_motivations", [])
    withholds = ""
    for item in hidden_motivations:
        item = item.strip()
        if item.lower() != "none" and item:
            withholds += f"- {item}\n"

    trigger_topics = filtered_persona.get("trigger_topics", [])
    triggers = ""
    for topic in trigger_topics:
        topic = topic.strip()
        if topic.lower() != "none" and topic:
            triggers += f"- {topic}\n"

    BEHAVIOUR_RULES = {
        "role": "system",
        "content": (
            "ROLE: You are the interviewee in a New Zealand police interview. "
            "STYLE: brief, natural NZ English. "
            "INTERACTION: Answer only what’s asked. Do not volunteer extra details. Do not begin describing the incident unless the interviewer explicitly asks about it. "
            "If unsure, say so. If a question seems irrelevant, politely deflect (e.g., ‘I’m not sure how that’s relevant’). "
            "NO META: Never mention internal attributes (tone, accent, speaking speed) or placeholder text. "
            "PRIVACY: Do not provide personal identifiers unless directly asked. "
            "OUTPUT FORMAT: Respond with spoken words only. Never include stage directions, physical actions, or emotes in asterisks (e.g. *nervous smile*, *looks away*, *sighs*). "
            "ROLE GUARD: You are not the interviewer. Never write lines prefixed with ‘Interviewer:’, ‘Detective:’, or ‘Officer:’; do not restate or quote the interviewer’s words."
            " Do not ask the interviewer any questions unless they explicitly invite you to (e.g., 'Do you have any questions?') or you must ask a single brief clarification to answer."
            " You may ask short clarifying questions if you genuinely need more information to answer. "
            "ACTOR CLARITY: Describe your own actions with 'I'; refer to others with explicit nouns (e.g., 'the teen', 'the driver'). Avoid ambiguous 'he/she/they' when two actors are in scope, and never attribute my actions to someone else."
            " QA FORMAT: Answer every part of the latest question. If the user asked two or more things (e.g., name and date of birth), provide all parts succinctly."
            " CONTACTS: Never invent phone numbers or email addresses. If asked and you do not have this in your background data, say you don’t know or prefer not to share."
            " If the interviewer makes a factual assertion about you (e.g., 'I understand you used to be a police officer.'), treat it as a yes/no prompt and briefly confirm or correct it without adding extra detail."
        )
    }

    FIRST_TURN_RULE = {
        "role": "system",
        "content": (
            "When asked an explicit open question about the incident (e.g., ‘Tell me what you saw’, ‘What happened?’), "
            "give a concise overview in 3 to 4 short sentences. Explain enough for the officer to grasp the event — time, general place, "
            "what the person did, and the immediate outcome — but do not give every detail yet. It’s fine to include one or two anchor specifics if they help "
            "paint the picture (e.g., a street name or vehicle type/colour). Do not list long strings of properties (heights, clothing colours, makes/models/plates, "
            "distances, or full addresses) unless asked. Stop after your 3–4 sentences."
        )
    }

    DEFLECT_EXAMPLE = {
        "role": "system",
        "content": "If asked something unrelated to the interview (e.g., ‘Do you like bananas?’), respond naturally: ‘I’m not sure how that’s relevant.’ If pressed, answer briefly and return to the topic."
    }

    today_nz = datetime.now().strftime("%-d %B %Y")
    system_instructions = f"Today is {today_nz}."

    # Consent prompts get a fixed reply before any other processing — no LLM needed.
    if message_history and message_history[-1].get("role") == "user" and _is_consent_prompt(message_history[-1].get("content", "")):
        return "Yes, that's okay."

    # ---------------- Tier 1: Deterministic handlers (facts-only, no LLM) ----------------
    if message_history and message_history[-1]["role"] == "user":
        raw_last_user = message_history[-1]["content"]
        last_user_lower = raw_last_user.lower().strip()

        facts_text = "\n".join((f.get("fact", "") for f in persona_data.get("facts_to_provide", []) if isinstance(f, dict)))
        facts_low = facts_text.lower()
        q = last_user_lower

        # Name spelling: produce "J-O-H-N S-M-I-T-H" format — uppercase, hyphens within words,
        # spaces between words. Deterministic because the LLM frequently misformats this.
        if re.search(r"\b(spell|spelling)\b.*\b(full\s+)?name\b", q) or re.search(r"\bcan\s+you\s+spell\b.*\bname\b", q):
            full = (persona_data.get("full_name") or "").strip()
            if full:
                parts = re.split(r"\s+", full)
                spelled = []
                for p in parts:
                    letters = "-".join(list(p.upper()))
                    spelled.append(letters)
                response = " ".join(spelled)
                return response
            response = "I'm not sure."
            return response

        # "Why didn't you…?" — extract the reason field from the matching negative fact rather
        # than letting the LLM invent a justification.
        if re.search(r"\bwhy\s+(didn'?t|did\s+not|couldn'?t|could\s+not)\s+you\s+", q):
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    reason_text = fact_item.get("reason", "")
                    fact_low = fact_text.lower()

                    # Licence plate is the most common "why didn't you" topic — check it first.
                    if re.search(r"\b(license|number|registration)\s*plate\b", q):
                        if re.search(r"\b(didn'?t\s+get|didn'?t\s+see|no|not)\s+(?:the\s+)?(license|number|registration)\s*plate\b", fact_low):
                            if reason_text and reason_text.strip().lower() not in ["", "none"]:
                                # Convert concise reason notes to natural first-person sentences.
                                # e.g. "Car left too quickly, angle didn't allow plate visibility"
                                # → "The car left too quickly and the angle didn't allow me to see the plate."
                                reason_clean = reason_text.strip()
                                if not re.match(r"^(I|The)\s+", reason_clean, re.IGNORECASE):
                                    reason_clean = reason_clean.replace("didn't allow plate visibility", "didn't allow me to see the plate")
                                    reason_clean = reason_clean.replace("plate visibility", "me to see the plate")
                                    reason_clean = reason_clean.replace(",", " and")
                                    if not reason_clean.lower().startswith("the "):
                                        reason_clean = "The " + reason_clean[0].lower() + reason_clean[1:]
                                response = reason_clean[0].upper() + reason_clean[1:] + ("." if not reason_clean.endswith(".") else "")
                                return response

                    # General case: match the verb from the question against negative statements in facts.
                    verb_match = re.search(r"why\s+(?:didn'?t|couldn'?t)\s+you\s+(\w+)", q)
                    if verb_match:
                        verb = verb_match.group(1).lower()
                        # Check if fact contains negative statement about that verb
                        if verb in fact_low or re.search(r"\b(didn'?t|couldn'?t|not)\b", fact_low):
                            # Try to match topic keywords
                            keywords = re.findall(r"\b(plate|weapon|knife|gun|mask|face|description|name|details?|number|registration|call|follow|record|write)\b", q)
                            if any(kw in fact_low for kw in keywords):
                                if reason_text and reason_text.strip().lower() not in ["", "none"]:
                                    reason_clean = reason_text.strip()
                                    if not re.match(r"^(I|The)\s+", reason_clean, re.IGNORECASE):
                                        reason_clean = reason_clean.replace(",", " and")
                                        if not reason_clean.lower().startswith("the "):
                                            reason_clean = "The " + reason_clean[0].lower() + reason_clean[1:]
                                    response = reason_clean[0].upper() + reason_clean[1:] + ("." if not reason_clean.endswith(".") else "")
                                    return response

        # "Why not?" / "Why?" — find the most recent negative answer and look up the reason.
        if re.search(r"^why\s+not\s*\??$", q) or re.search(r"^why\s*\??$", q):
            if len(message_history) >= 2:
                prev_user_msg = None
                prev_assistant_msg = None
                for i in range(len(message_history) - 2, -1, -1):
                    if message_history[i]["role"] == "user" and prev_user_msg is None:
                        prev_user_msg = message_history[i]["content"]
                    elif message_history[i]["role"] == "assistant" and prev_assistant_msg is None:
                        prev_assistant_msg = message_history[i]["content"]
                    if prev_user_msg and prev_assistant_msg:
                        break

                # Only continue if the previous answer was a bare "No." — any other response
                # means the context is ambiguous and the LLM should handle it.
                if prev_assistant_msg and prev_assistant_msg.strip().lower() in ["no", "no."]:
                    prev_q_lower = prev_user_msg.lower()

                    for fact_item in filtered_persona.get("facts_to_provide", []):
                        if isinstance(fact_item, dict):
                            fact_text = fact_item.get("fact", "")
                            reason_text = fact_item.get("reason", "")
                            fact_low = fact_text.lower()

                            if re.search(r"\b(license|number|registration)\s*plate\b", prev_q_lower):
                                if re.search(r"\b(didn'?t\s+get|no|not)\s+(the\s+)?(license|number|registration)\s*plate\b", fact_low):
                                    if reason_text and reason_text.strip().lower() not in ["", "none"]:
                                        # Same reason-to-sentence conversion as the "why didn't you" handler above.
                                        reason_clean = reason_text.strip()
                                        if not re.match(r"^(I|The)\s+", reason_clean, re.IGNORECASE):
                                            reason_clean = reason_clean.replace("didn't allow plate visibility", "didn't allow me to see the plate")
                                            reason_clean = reason_clean.replace(",", " and")
                                            if not reason_clean.lower().startswith("the "):
                                                reason_clean = "The " + reason_clean[0].lower() + reason_clean[1:]
                                        response = reason_clean[0].upper() + reason_clean[1:] + ("." if not reason_clean.endswith(".") else "")
                                        return response

                            # General case: match topic nouns from the previous question against facts.
                            prev_keywords = re.findall(r"\b(plate|weapon|knife|gun|mask|face|description|name|details?)\b", prev_q_lower)
                            if prev_keywords:
                                for keyword in prev_keywords:
                                    if keyword in fact_low and re.search(r"\b(didn'?t|not|no|couldn'?t)\b", fact_low):
                                        if reason_text and reason_text.strip().lower() not in ["", "none"]:
                                            # Convert reason to natural sentence
                                            reason_clean = reason_text.strip()
                                            if not re.match(r"^(I|The)\s+", reason_clean, re.IGNORECASE):
                                                reason_clean = reason_clean.replace(",", " and")
                                                if not reason_clean.lower().startswith("the ") and not reason_clean.lower().startswith("i "):
                                                    reason_clean = "The " + reason_clean[0].lower() + reason_clean[1:]
                                            response = reason_clean[0].upper() + reason_clean[1:] + ("." if not reason_clean.endswith(".") else "")
                                            return response

            response = "I'm not sure."
            return response

        # Distance
        if re.search(r"\bhow\s+far\b.*\b(away|from|were\s+you)\b", q) or re.search(r"\b(distance)\b", q):
            dist = _extract_distance_phrase(facts_low)
            if dist:
                return f"About {dist}."
            return "I'm not sure."

        # "Describe what was happening" / "what was happening as you were walking?" — extract
        # movement context from facts to avoid spatial confabulation by the LLM.
        if re.search(r"\bdescribe\s+what\s+was\s+happening\b", q) or \
           re.search(r"\bwhat\s+was\s+happening\s+(as|when|while)\s+you\s+were\b", q):
            parts = []
            if re.search(r"\bwalking\s+", facts_low):
                # Match "walking home along Street" first because it is more specific.
                m = re.search(r"\bwalking\s+home\s+(?:down|along|towards?)\s+([A-Z][A-Za-z''\-]+(?:\s+[A-Z][A-Za-z''\-]+)?(?:\s+Street|St|Road|Rd|Avenue|Ave)?)\b", facts_text)
                if m:
                    place = m.group(1)
                    parts.append(f"I was walking home along {place}")
                else:
                    # Try "walking [direction] Street"
                    m = re.search(r"\bwalking\s+(home|down|towards|along)\s+([A-Z][A-Za-z''\-]+(?:\s+[A-Z][A-Za-z''\-]+)?(?:\s+Street|St|Road|Rd|Avenue|Ave)?)\b", facts_text)
                    if m:
                        direction = m.group(1)
                        place = m.group(2)
                        parts.append(f"I was walking {direction} {place}")
                    else:
                        parts.append("I was walking")
            elif re.search(r"\bdriving\s+", facts_low):
                # Try to match "driving home along Street" first (more specific)
                m = re.search(r"\bdriving\s+home\s+(?:down|along|towards?)\s+([A-Z][A-Za-z''\-]+(?:\s+[A-Z][A-Za-z''\-]+)?(?:\s+Street|St|Road|Rd|Avenue|Ave)?)\b", facts_text)
                if m:
                    place = m.group(1)
                    parts.append(f"I was driving home along {place}")
                else:
                    # Try "driving [direction] Street"
                    m = re.search(r"\bdriving\s+(home|down|towards|along)\s+([A-Z][A-Za-z''\-]+(?:\s+[A-Z][A-Za-z''\-]+)?(?:\s+Street|St|Road|Rd|Avenue|Ave)?)\b", facts_text)
                    if m:
                        direction = m.group(1)
                        place = m.group(2)
                        parts.append(f"I was driving {direction} {place}")
                    else:
                        parts.append("I was driving")

            # Towards intersection
            if re.search(r"\btowards?\s+(?:the\s+)?intersection\s+(?:with|of)\s+([A-Z][A-Za-z''\-]+)", facts_text):
                m = re.search(r"\btowards?\s+(?:the\s+)?intersection\s+(?:with|of)\s+([A-Z][A-Za-z''\-]+)", facts_text)
                if m:
                    street = m.group(1)
                    parts.append(f"heading towards the intersection with {street}")

            # Heard/saw something
            heard_match = re.search(r"\bheard\s+(screaming|shouting|glass\s+breaking|a\s+noise|voices)\b", facts_low)
            if heard_match:
                # Use EXACT word from facts - don't substitute synonyms
                heard_what = heard_match.group(1)
                parts.append(f"when I heard {heard_what}")
            elif re.search(r"\bsaw\s+(?:a\s+)?(?:teen(?:\s+boy)?|man|woman|person|someone)\s+(run|ran|running)\s+(?:out|into|in)\b", facts_low):
                # Saw someone running out/in - check if also saw smoke/fire
                if re.search(r"\bsaw\s+smoke\b", facts_low) or re.search(r"\bsaw.*smoke\b", facts_low):
                    parts.append("when I saw someone run out and saw smoke")
                else:
                    parts.append("when I saw someone run out")
            elif re.search(r"\bsaw\s+(?:a\s+)?(?:teen(?:\s+boy)?|man|woman|person|someone)\s+(enter|entering)\b", facts_low):
                parts.append("when I saw someone enter")
            elif re.search(r"\bsaw\s+smoke\b", facts_low):
                parts.append("when I saw smoke")

            if parts:
                # Join parts naturally
                response = parts[0]
                if len(parts) > 1:
                    response += ", " + ", ".join(parts[1:])
                return response + "."
            # Fallback to generic if no specific movement found
            return "I was on my way when I noticed the incident."

        # Time of incident — guard against matching "Where were you when…" questions,
        # which start with "where" and are location questions, not time questions.
        if not re.search(r"^where\s+", q) and re.search(r"\b(what\s+time|when)\b.*\b(happen(ed)?|was\s+this|did\s+this(\s+\w+)?\s+occur|did\s+it(\s+\w+)?\s+happen)\b", q):
            tphrase = _extract_time_phrase(facts_low)
            if tphrase:
                return (tphrase[0].upper() + tphrase[1:] + ".")
            return "I'm not sure."

        # Cause of commotion / glass breaking / shouting
        if re.search(r"\bwhat\s+was\s+caus(?:ing|e)\b", q) or \
           re.search(r"\bwhat\s+caused\s+(it|that|the\s+noise|the\s+commotion|the\s+glass\s+breaking|the\s+shouting)\b", q) or \
           re.search(r"\bwhat\s+was\s+(the\s+)?(noise|commotion)\b", q):
            heard_noise = bool(re.search(r"\b(shouting|screaming|glass\s+breaking|smashed\s+glass)\b", facts_low))
            saw_inside = bool(re.search(r"\b(saw|witnessed)\b.*\binside\b", facts_low))
            went_inside = bool(re.search(r"\b(ran|went)\s+into\b.*\b(shop|store|premises|house|building)\b", facts_low))
            if heard_noise and (saw_inside or went_inside):
                return "It sounded like something happening inside the premises."
            if re.search(r"\b(fire|arson)\b", facts_low):
                return "A fire."
            if heard_noise:
                return "It sounded like shouting and glass breaking."
            return "I'm not sure."

        # Mask
        if re.search(r"\b(what\s+(kind|type)\s+of\s+mask|what\s+mask|describe\s+the\s+mask)\b", q):
            md = _mask_description_from_facts(facts_low)
            if md:
                desc = md.strip()
                if not re.match(r"^(a|an|the)\b", desc, flags=re.IGNORECASE):
                    desc = "a " + desc
                return f"It was {desc}."
            return "I'm not sure."

        # Clothing — "was" and "were" are both valid ("What was he wearing?" / "What were they wearing?").
        if re.search(r"\b(what\s+(was|were)\s+(he|she|they|the\s+\w+)\s+wearing|what\s+clothing|clothes)\b", q):
            items = _clothing_list_from_facts(facts_low)
            if items:
                return ", ".join(items) + "."
            return "I'm not sure."

        # Sweatshirt/jumper questions — handle both "describe the hoodie" and
        # "did it have any text/logo on it?" variants.
        if re.search(r"\b(describe|what\s+did\s+the)\s+(hoodie|sweat\s*shirt|sweatshirt|jumper)\b", q) or \
           re.search(r"\b(did\s+(the\s+)?)?(hoodie|sweat\s*shirt|sweatshirt|jumper)\b.*\b(text|writing|logo|image|graphic|picture|print|printed)\b", q) or \
           re.search(r"\b(any|have|with)\b.*\b(text|writing|logo|image|graphic|picture|print|printed)\b.*\b(hoodie|sweat\s*shirt|sweatshirt|jumper)\b", q):
            # Markings questions need a yes/no answer; general description questions need the colour/style.
            if re.search(r"\b(text|writing|logo|image|graphic|picture|print|printed)\b", q):
                has_mark = _sweatshirt_has_markings_from_facts(facts_low)
                if has_mark is True:
                    return "Yes, it had some writing on it."
                if has_mark is False:
                    return "No, it looked plain — no text or images."
                return "I'm not sure."
            # Otherwise provide a simple description
            sw = _sweatshirt_description(facts_low)
            return sw if sw else "I'm not sure."

        # Footwear
        if re.search(r"\b(what\s+type|what\s+kind)\s+of\s+(shoes|trainers|sneakers)\b", q) or \
           re.search(r"\bwhat\s+(shoes|footwear)\b", q) or \
           re.search(r"\btype\s+of\s+shoes\b", q):
            m_shoes = re.search(r"\b(black|white|grey|gray|blue|red|green|brown|dark|light)\b\s+(trainers|sneakers|shoes|boots)\b", facts_low)
            if m_shoes:
                colour = m_shoes.group(1).replace("gray", "grey")
                kind = m_shoes.group(2)
                return f"{colour.capitalize()} {kind}."
            if re.search(r"\b(trainers|sneakers|shoes|boots)\b", facts_low):
                kind = re.search(r"\b(trainers|sneakers|shoes|boots)\b", facts_low).group(1)
                return f"{kind.capitalize()}."
            return "I'm not sure."

        # Age of the observed person — not the witness's own age (that is handled by the identity fields block below).
        if re.search(r"\bhow\s+old\s+(was|were)\s+(he|she|they|the\s+person|the\s+man|the\s+woman|the\s+offender|the\s+suspect)\b", q):
            age_phrase = _extract_age_from_facts(facts_low)
            if age_phrase:
                return age_phrase
            return "I'm not sure."

        # Weapon-specific questions — use _extract_weapons_only so bags/backpacks are not included.
        # "Was he carrying a weapon?" must not return "a backpack."
        if re.search(r"\b(carrying|holding|have|had)\s+(?:a\s+)?(weapon|weapons)\b", q) or \
           re.search(r"\bwas\s+(he|she|they)\s+carrying\s+(?:a\s+)?(weapon|weapons)\b", q):
            weapons = _extract_weapons_only(facts_text)
            if weapons:
                return weapons
            return "No."

        # What was the person carrying/holding?
        if re.search(r"\b(what\s+was|what\s+did)\s+(he|she|they|the\s+\w+)\s+(carrying|holding|have|hold)\b", q) or \
           re.search(r"\b(was\s+he|was\s+she|were\s+they)\s+(carrying|holding)\b", q):
            items = _extract_carried_items(facts_text)
            if items:
                return items
            return "I'm not sure."

        # "Could you see what he was holding as he ran into the shop?" — visibility questions
        # about events inside a building must be grounded in facts to prevent fabrication.
        # A witness standing outside can only describe what was visible from the street.
        if re.search(r"\b(as|when)\s+(he|she|they|the\s+\w+)\s+(ran|went|entered|walked)\s+(into|inside|in)\s+(the\s+)?(shop|store|premises|building|house)\b", q) and \
           re.search(r"\b(could|did|can)\s+you\s+see\s+(what|him|her|them)\b", q):
            asking_about_holding = re.search(r"\b(holding|carrying|had|have)\b", q)

            saw_run_in_with_item = re.search(r"\bsaw\s+.*\b(run|running|ran)\s+(into|in)\b.*\b(holding|carrying|with)\s+(a|an|the)\s+(\w+)\b", facts_low)

            if asking_about_holding and saw_run_in_with_item:
                # Extract what they were holding from facts
                item_match = re.search(r"\b(holding|carrying|with)\s+(a|an|the)\s+(\w+)\b", facts_low)
                if item_match:
                    item = item_match.group(3)
                    return f"Yes, I could see that they were holding a {item}."

            # If asking about actions INSIDE (what they were doing, not just holding)
            if re.search(r"\b(doing|do|did|put|take|took|grab|grabbed)\b", q):
                # Check distance - if far away, can't see inside
                dist = _extract_distance_phrase(facts_low)
                if dist:
                    dist_match = re.search(r"(\d+)\s*metres?", dist)
                    if dist_match and int(dist_match.group(1)) > 50:
                        # Too far to see inside — describe what the witness could hear instead.
                        if re.search(r"\bheard\s+(screaming|shouting|glass\s+breaking|noise|sounds?)\b", facts_low):
                            return "I couldn't see what was happening inside. I heard screaming and glass breaking coming from inside."
                        return "No, I was too far away to see inside."

                # If the facts confirm the witness went inside or saw inside, let the LLM handle it.
                if re.search(r"\bsaw\s+(him|her|them)\s+inside\b", facts_low) or re.search(r"\bwent\s+inside\b.*\bsaw\b", facts_low):
                    return None

                if re.search(r"\bheard\s+(screaming|shouting|glass\s+breaking)\b", facts_low):
                    return "I couldn't see inside. I heard screaming and glass breaking."
                return "I couldn't see what was happening inside."

            # Generic "could you see" about the entry moment — fall through to the carried-items handler.
            return None

        # "Did you hear any noise?" — extract the specific sound terms from facts rather than
        # letting the LLM substitute synonyms (e.g., "shouting" when facts say "screaming").
        if re.search(r"\b(did|could)\s+you\s+hear\s+(any|anything|a)\s+(noise|sound|sounds?|noises?)\b", q) or \
           re.search(r"\bhear\s+(?:any\s+)?noise\s+(?:coming\s+)?from\s+(?:the\s+)?(shop|store|premises|building)\b", q):
            if re.search(r"\bheard\s+(screaming|shouting|yelling|glass\s+breaking|smashed\s+glass)\b", facts_low):
                if re.search(r"\bscreaming\b", facts_low) and re.search(r"\bglass\s+breaking\b", facts_low):
                    return "Yes. I heard screaming and the sound of glass breaking."
                if re.search(r"\bscreaming\b", facts_low):
                    return "Yes. I heard screaming."
                if re.search(r"\bglass\s+breaking\b", facts_low):
                    return "Yes. I heard the sound of glass breaking."
                if re.search(r"\b(shouting|yelling)\b", facts_low):
                    return "Yes. I heard shouting."
                # Generic: heard something
                return "Yes. I heard noises coming from inside."
            return "No, I didn't hear anything."

        # Quoted speech — extract the exact words from the fact if they are in quotes.
        if re.search(r"\bwhat\s+did\s+(?:the\s+)?(?:man|woman|person|he|she|they)\s+(?:yell|say|scream|shout)\b", q, re.IGNORECASE):
            speech_match = re.search(r"(?:yelling|saying|screaming|shouting)\s+['\"]([^'\"]+)['\"]", facts_text, re.IGNORECASE)
            if speech_match:
                spoken_words = speech_match.group(1)
                # Check if it was repeated
                if re.search(r"twice|two\s+times", facts_text, re.IGNORECASE):
                    return f"She yelled '{spoken_words}' twice."
                else:
                    return f"She yelled '{spoken_words}'."
            return "I'm not sure exactly what was said."

        # Vehicle occupants — the LLM tends to invent passengers, so answer from facts only.
        if re.search(r"\bwas\s+(?:anyone|anybody)\s+else\s+(?:in|inside)\s+(the\s+)?(car|vehicle)\b", q, re.IGNORECASE) or \
           re.search(r"\b(?:anyone|anybody)\s+else\s+(?:in|inside)\s+(the\s+)?(car|vehicle)\b", q, re.IGNORECASE):
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    fact_low = fact_text.lower()

                    # Check for negative patterns (no one else)
                    if re.search(r"\b(no\s+one|nobody)\s+else\s+(?:was\s+)?(?:in|inside)\b", fact_low):
                        return "No, no one else was inside the car."
                    if re.search(r"\bonly\s+(?:the\s+)?driver\b", fact_low):
                        return "No, only the driver."
                    # Check for positive patterns (someone else was there)
                    if re.search(r"\b(someone|person|passenger)\s+(?:else\s+)?(?:was\s+)?(?:in|inside)\b", fact_low):
                        return "Yes."

            return "I'm not sure."

        # Vehicle make/model — honour any certainty markers in the fact (e.g., "not 100% sure")
        # by adding hedging language to the response.
        if re.search(r"\bwhat\s+(type|make|model|kind)\s+of\s+(car|vehicle)\b", q, re.IGNORECASE) or \
           re.search(r"\bwhat\s+(car|vehicle)\s+(was\s+it|type)\b", q, re.IGNORECASE):
            vehicle_desc = _vehicle_details_phrase(facts_low)
            if vehicle_desc:
                is_uncertain = False
                for fact_item in filtered_persona.get("facts_to_provide", []):
                    if isinstance(fact_item, dict):
                        fact_text = fact_item.get("fact", "")
                        fact_low_check = fact_text.lower()
                        certainty = fact_item.get("certainty", "")

                        if re.search(r"\b(car|vehicle|mazda|toyota|honda|nissan|ford|holden|subaru|hyundai)\b", fact_low_check):
                            if certainty.lower() == "unsure" or re.search(r"\b(think|maybe|possibly|not\s+sure|not\s+100%)\b", fact_low_check):
                                is_uncertain = True
                                if re.search(r"\bnot\s+100%\s+sure\b", fact_low_check):
                                    return f"I think it was {vehicle_desc} but I'm not 100% sure."
                                break

                if is_uncertain:
                    return f"I think it was {vehicle_desc}, but I'm not certain."
                return f"It was {vehicle_desc}."

            return "I'm not sure."

        # Vehicle colour — iterates facts in order so the colour from the incident fact is
        # preferred over incidental colour mentions elsewhere.
        if re.search(r"\bwhat\s+(was\s+the\s+)?colo[u]?r\s+(was\s+|of\s+)?(the\s+)?(car|vehicle)\b", q, re.IGNORECASE):
            colours_pattern = r"\b(light\s+grey|light\s+gray|dark\s+grey|dark\s+gray|silver|grey|gray|black|white|red|blue|green|brown|yellow|orange|purple|pink|gold)\b"

            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    fact_low = fact_text.lower()

                    # Check if fact mentions a vehicle
                    if re.search(r"\b(car|vehicle|hatchback|sedan|wagon|suv|ute|van|truck)\b", fact_low):
                        # Extract colour from vehicle description
                        colour_match = re.search(colours_pattern, fact_low)
                        if colour_match:
                            colour = colour_match.group(1)
                            response = f"It was {colour}."
                            return response

            # If no colour found, return uncertain
            return "I'm not sure."

        # "What did you hear first?" — respect the fact ordering so the chronologically
        # earliest sound is returned, not the one the LLM happens to prioritise.
        if re.search(r"\bwhat\s+did\s+you\s+hear\s+first\b", q, re.IGNORECASE):
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    fact_low = fact_text.lower()

                    # Check if this fact mentions hearing something
                    if re.search(r"\b(?:i\s+)?heard\b", fact_low):
                        # Extract what was heard
                        # Pattern: "heard [something]"
                        heard_match = re.search(r"\bheard\s+(?:a\s+)?(.+?)(?:\s+and\s+saw|\s+and\s+then|,|\.|$)", fact_low)
                        if heard_match:
                            what_heard = heard_match.group(1).strip()
                            return f"I heard a {what_heard}."
                        # Fallback: just say what the fact says about hearing
                        return fact_text

            return "I'm not sure."

        # "What did you hear?" — quoted speech takes priority; then specific sound terms;
        # then generic sounds. The exact words from facts are used to prevent synonym drift.
        if re.search(r"\bwhat\s+(?:exactly\s+)?did\s+you\s+hear\b", q, re.IGNORECASE):
            speech_match = re.search(r"yelling\s+['\"]([^'\"]+)['\"]", facts_text, re.IGNORECASE)
            if speech_match:
                spoken_words = speech_match.group(1)
                # Check if it was repeated
                if re.search(r"twice|two\s+times", facts_text, re.IGNORECASE):
                    return f"I heard shouting and a woman yelling '{spoken_words}' twice."
                else:
                    return f"I heard shouting and a woman yelling '{spoken_words}'."
            # Generic shouting/screaming
            if re.search(r"\bheard\s+(screaming|shouting|yelling)\b", facts_low):
                if re.search(r"\bglass\s+breaking\b", facts_low):
                    return "I heard shouting and the sound of glass breaking."
                return "I heard shouting."
            # Glass breaking or window being smashed
            if re.search(r"\bglass\s+breaking\b", facts_low):
                return "I heard the sound of glass breaking."
            if re.search(r"\b(car\s+)?window\s+being\s+(smashed|broken|shattered)\b", facts_low):
                response = "I heard a car window being smashed."
                return response
            # Search both fact text and reason fields for bang/explosion — reason fields often
            # explain context ("heard a loud bang — it was the car backfiring") that facts mention briefly.
            all_text = facts_low
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    reason = fact_item.get("reason", "")
                    if reason:
                        all_text += " " + reason.lower()

            if re.search(r"\b(bang|explosion|blast)\b", all_text):
                # Check if it was "loud bang" or "big bang"
                if re.search(r"\b(loud|big)\s+(bang|explosion|blast)\b", all_text):
                    return "I heard a loud bang."
                return "I heard a bang."
            # Generic sound/noise
            if re.search(r"\bheard\s+(a\s+)?(noise|sound|thud)\b", facts_low):
                return "I heard a loud noise."
            return "I'm not sure."

        # "What happened when they ran out?" — extract the immediate next action from facts
        # in chronological order rather than letting the LLM summarise the whole sequence.
        if re.search(r"\bwhat(\s+\w+)?\s+happened\s+(?:when|after)\s+(?:the\s+)?(?:man|person|woman|he|she|they)\s+(?:ran|came|went|left|exited)\s+(?:out|away)\b", q) or \
           re.search(r"\bwhat\s+(?:did\s+)?(?:happen|he\s+do|she\s+do|they\s+do)\s+(?:when|after)\s+(?:he|she|they)\s+(?:ran|came|went|left|exited)\s+(?:out|away)\b", q):
            # Look for the immediate next action in chronological order
            # Check for "ran/walked/went past" (these come BEFORE later movement like "ran down [lane]")
            if re.search(r"\b(ran|walked|went)\s+past\s+(?:me|us|on\s+the\s+other\s+side)\b", facts_low):
                # Check if they removed something (mask, hat, jacket, etc.) while passing
                removal_match = re.search(r"\btook\s+(?:his|her|their|the)\s+(\w+)\s+off\s+as\s+(?:he|she|they)\s+(ran|walked|went)\s+past\b", facts_low)
                if removal_match:
                    item = removal_match.group(1)
                    action = removal_match.group(2)
                    return f"He {action} past me on the other side of the road and took his {item} off as he {action} past."
                # No removal, just passed
                return "He ran past me on the other side of the road."

            # Check for "ran/walked/went towards me/us"
            if re.search(r"\b(ran|walked|went)\s+(?:out.*)?towards\s+(?:me|us)\b", facts_low):
                return "He ran towards me."

            # Generic: just exited (don't hardcode specific locations)
            return "He left the premises."

        # Did you see the argument/fight/altercation take place?
        if re.search(r"\bdid\s+you\s+see\s+(?:the\s+)?(argument|fight|altercation|assault|attack|incident|confrontation)\s+(?:take\s+place|happen|occur)\b", q, re.IGNORECASE):
            # Check if facts mention seeing an argument, fight, or similar event
            if re.search(r"\b(argued|argument|fight|fighting|fought|punched|hit|struck|attacked|assaulted)\b", facts_low):
                return "Yes."
            return "No."

        # Did you follow him/her outside/out?
        if re.search(r"\bdid\s+you\s+follow\s+(?:him|her|them|the\s+\w+)\s+(outside|out|after\s+(?:him|her|them))\b", q, re.IGNORECASE):
            # Check if facts mention following someone
            if re.search(r"\bi\s+followed\s+(?:out|outside|him|her|them|the\s+\w+)", facts_low):
                return "Yes."
            return "No."

        # "What did he do after he punched him?" — find the next chronological fact after
        # the one that mentions the action from the question.
        if re.search(r"\bwhat\s+did\s+(?:he|she|they|the\s+(?:man|woman|person))\s+do\s+after\s+(?:he|she|they)\s+(\w+)", q, re.IGNORECASE):
            action_match = re.search(r"\bafter\s+(?:he|she|they)\s+(\w+)", q, re.IGNORECASE)
            if action_match:
                action_mentioned = action_match.group(1).lower()
                facts_list = filtered_persona.get("facts_to_provide", [])
                for i, fact_item in enumerate(facts_list):
                    if isinstance(fact_item, dict):
                        fact_text = fact_item.get("fact", "")
                        fact_low = fact_text.lower()
                        if action_mentioned in fact_low or re.search(rf"\b{action_mentioned}", fact_low):
                            # Check if the continuation is in the same fact ("and then ran out") …
                            if re.search(r"\b(?:and|then)\s+(?:he|she|they|the\s+\w+)\s+(ran|went|walked|left|fled|drove)\s+(out|away|off|to|into|down)", fact_low):
                                next_action_match = re.search(r"\b(?:and|then)\s+(?:he|she|they|the\s+\w+)\s+(ran|went|walked|left|fled|drove)\s+(out|away|off|to|into|down)(?:\s+through\s+the\s+\w+)?(?:\s+onto\s+\w+)?", fact_low)
                                if next_action_match:
                                    verb = next_action_match.group(1)
                                    direction = next_action_match.group(2)
                                    # Extract fuller context
                                    fuller_match = re.search(rf"{verb}\s+{direction}(?:\s+through\s+the\s+([^\.]+?))?(?:\s+onto\s+([^\.]+?))?", fact_low)
                                    if fuller_match:
                                        if fuller_match.group(1):
                                            return f"He {verb} {direction} through the {fuller_match.group(1).strip()}."
                                        else:
                                            return f"He {verb} {direction}."
                            # … or look at the very next fact for the continuation.
                            if i + 1 < len(facts_list) and isinstance(facts_list[i + 1], dict):
                                next_fact = facts_list[i + 1].get("fact", "").lower()
                                if re.search(r"\b(?:he|she|they|the\s+\w+)\s+(ran|went|walked|left|fled|drove)\s+(out|away|off|to|into|down)", next_fact):
                                    action_match = re.search(r"\b(?:he|she|they|the\s+one\s+who\s+\w+)\s+(ran|went|walked|left|fled|drove)\s+(out|away|off)(?:\s+through\s+the\s+([^\.]+?))?", next_fact)
                                    if action_match:
                                        verb = action_match.group(1)
                                        direction = action_match.group(2)
                                        if action_match.group(3):  # through the main doors
                                            return f"He {verb} {direction} through the {action_match.group(3).strip()}."
                                        else:
                                            return f"He {verb} {direction}."
            return "I'm not sure."

        # "Did you find him?" — look for an explicit outcome in the facts; don't guess.
        if re.search(r"\b(and\s+)?(did|were\s+you\s+able\s+to|could\s+you)\s+(?:you\s+)?(find|locate|catch)\s+(him|her|them|the\s+\w+)\b", q, re.IGNORECASE):
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    fact_low = fact_text.lower()

                    # Negative: couldn't find / didn't find
                    if re.search(r"\b(couldn'?t|could\s+not|did\s+not|didn'?t)\s+find\s+(him|her|them|the\s+\w+)\b", fact_low):
                        response = "No, I couldn't find him."
                        return response

                    # Positive: found / located / caught up with
                    if re.search(r"\b(found|located|caught\s+up\s+with|spotted)\s+(him|her|them|the\s+\w+)\b", fact_low):
                        response = "Yes."
                        return response

            # If unclear from facts, return unsure
            return "I'm not sure."

        # "Did you see where they went?" — reveal suspect steps one at a time.
        # Optional filler words ("did you actually see where…") are allowed in the pattern.
        if re.search(r"\b(did|could)\s+you(\s+\w+)?\s+see\s+where\s+(he|she|they|the\s+\w+)(\s+\w+)?\s+(went|ran|headed|drove)\b", q):
            suspect_steps = _extract_suspect_steps(facts_text)
            if suspect_steps:
                # Check if we've already mentioned steps
                steps_said = _history_completed_suspect_steps(message_history)
                # If no steps mentioned yet, give first one
                for step in suspect_steps:
                    if step not in steps_said and "He got into" not in step:
                        return "Yes, " + step[0].lower() + step[1:]
                # If all route steps mentioned, return Yes
                return "Yes."
            return "No."

        # "What did you do next?" — advance the witness timeline one step at a time using
        # _extract_witness_steps() and _history_completed_steps() to track progress.
        if re.search(r"\b(what\s+did\s+you(\s+\w+)?\s+do\s+next|what\s+did\s+you(\s+\w+)?\s+do\s+then|and\s+what\s+did\s+you(\s+\w+)?\s+do\s+next|then\s+what\s+did\s+you\s+do)\b", q):
            steps_all = _extract_witness_steps(facts_low)
            steps_said = _history_completed_steps(message_history)
            for s in steps_all:
                if s not in steps_said:
                    return s
            response = "That was all I did."
            return response

        # Identity/contact fields — build patterns for each field and return all matches combined.
        # Occupation is handled by a micro-LLM call; other fields return the raw persona value.
        if "spell" in last_user_lower:
            pass
        elif not _is_consent_prompt(raw_last_user):
            QUESTION_PATTERNS = {
                # Guard against matching "What's the pedestrian's name?" — only match the witness's own name.
                "full_name": re.compile(r"\b(what(?:'s| is)?\s+your\s+(full\s+)?name|your\s+(full\s+)?name|state\s+your\s+name|who\s+are\s+you)\b(?!\s+(?:of|for)\s+(?:the|that))", re.I),
                "date_of_birth": re.compile(r"\b(date\s+of\s+birth|dob|when\s+were\s+you\s+born)\b", re.I),
                "home_address": re.compile(r"\b(where\s+do\s+you\s+live|home\s+address|residential\s+address|street\s+address|your\s+address|and\s+(?:your\s+)?address)\b", re.I),
                "business_address": re.compile(
                    r"\b(work\s+address|business\s+address|office\s+address|workplace\s+address|where\s+is\s+(your\s+)?(job|work)\s+located|where\s+are\s+they\s+located|where\s+are\s+they\s+based|what\s+is\s+(the\s+)?(work|office|business)\s+address)\b",
                    re.I),
                "occupation": re.compile(
                    r"\b("
                    r"what(?:'s| is)?\s+your\s+(occupation|job|job\s+title)"
                    r"|what\s+job\s+(?:do\s+)?you\s+do"
                    r"|what\s+do\s+you\s+do(?:\s+for\s+(work|a\s+living))?"
                    r"|your\s+job\s+title"
                    r")\b",
                    re.I),
                "employed_by": re.compile(r"\b(who\s+(?:do\s+)?you\s+work\s+for|your\s+employer|where\s+do\s+you\s+work(?:\s+at)?|place\s+of\s+work|work\s+at)\b", re.I),
                "age": re.compile(r"\b(how\s+old\s+are\s+you|what(?:'s| is)?\s+your\s+age|your\s+age)\b", re.I),
                "home_phone": re.compile(r"\b(home\s+phone|home\s+number|landline)\b", re.I),
                "work_phone": re.compile(r"\b(work\s+(phone|telephone|tel)|office\s+(phone|number|telephone|tel)|business\s+(phone|telephone|tel))\b", re.I),
                "cell_phone": re.compile(r"\b(cell\s+phone|mobile(\s+number)?|cell\s+number)\b", re.I),
                "email": re.compile(r"\b(email(\s+address)?|e-?mail)\b", re.I),
                "social_networking": re.compile(r"\b(social\s+(?:media|network(?:ing)?)|facebook|twitter|instagram|linkedin|tiktok|handle|username)\b", re.I),
            }
            filtered_persona_ids = {k: v for k, v in persona_data.items() if k in [
                "full_name","date_of_birth","home_address","business_address",
                "employed_by","occupation","home_phone","work_phone","cell_phone",
                "email","social_networking"
            ]}
            matches = []
            for field, pattern in QUESTION_PATTERNS.items():
                if field == "age" or field in filtered_persona_ids:
                    if pattern.search(raw_last_user):
                        matches.append(field)
            if matches == ["occupation"]:
                # Occupation answers require nuance — route to a constrained micro-LLM call
                # rather than returning the raw persona field verbatim.
                today_nz = datetime.now().strftime("%-d %B %Y")
                base_msgs_for_micro = [
                    {"role": "system", "content": f"Today is {today_nz}."},
                    BEHAVIOUR_RULES,
                    persona_background_message(filtered_persona),
                ]
                return _answer_occupation_via_llm(base_msgs_for_micro, persona_data, raw_last_user)
            if matches:
                parts_out = []
                for field in matches:
                    value = filtered_persona_ids.get(field, "").strip() if field in filtered_persona_ids else ""
                    if field == "full_name" and value:
                        parts_out.append(f"My name is {value}.")
                    elif field == "date_of_birth" and value:
                        try:
                            parts = value.split("-")
                            day = int(parts[2])
                            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
                            month_str = month_name[int(parts[1])]
                            parts_out.append(f"My date of birth is the {day}{suffix} {month_str} {parts[0]}.")
                        except:
                            parts_out.append(f"My date of birth is {value}.")
                    elif field == "home_address" and value:
                        parts_out.append(f"I live at {value}.")
                    elif field == "business_address" and value:
                        parts_out.append(f"My work address is {value}.")
                    elif field in ("home_phone", "work_phone", "cell_phone"):
                        nice = {"home_phone": "home phone", "work_phone": "work phone", "cell_phone": "mobile number"}[field]
                        if value:
                            parts_out.append(f"My {nice} is {value}.")
                        else:
                            parts_out.append(f"Sorry, I don't have a {nice}.")
                    elif field == "email":
                        if value:
                            parts_out.append(f"My email is {value}.")
                        else:
                            parts_out.append("Sorry, I don't have an email address.")
                    elif field == "occupation":
                        if value and value.lower() not in ("unemployed", "not employed", "not working"):
                            parts_out.append(f"I'm a {value}.")
                        else:
                            parts_out.append("I'm not currently working.")
                    elif field == "employed_by" and value:
                        v = value.strip()
                        if v.lower() in ("unemployed", "not employed", "not working", "jobless"):
                            parts_out.append("I'm not working at the moment.")
                        elif v.lower().startswith("self-employed"):
                            rest = v[len("self-employed"):].strip()
                            if rest.lower().startswith("owner of"):
                                parts_out.append(f"I'm the self-employed {rest}.")
                            elif rest:
                                parts_out.append(f"I'm a self-employed {rest}.")
                            else:
                                parts_out.append("I'm self-employed.")
                        else:
                            parts_out.append(f"I work for {v}.")
                    elif field == "social_networking":
                        if value:
                            parts_out.append(f"I can be found online at {value}.")
                        else:
                            parts_out.append("No, I don't have any social media accounts.")
                    elif field == "age":
                        # Calculate age in NZ local time so it's correct around the birthday.
                        dob_str = persona_data.get("date_of_birth", "")
                        try:
                            from dateutil import tz
                            dob = parse_date(dob_str).date()
                            today = datetime.now(tz.gettz("Pacific/Auckland")).date()
                            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                            parts_out.append(f"I am currently {age} years old.")
                        except:
                            parts_out.append("I'm not sure.")
                if parts_out:
                    return " ".join(parts_out)

    last_user_text = next((m.get("content", "") for m in reversed(message_history) if m.get("role") == "user"), "")

    # Distance contradiction — quote back the same distance the witness already gave,
    # then clarify the apparent inconsistency without changing the stated facts.
    if _is_distance_contradiction_challenge(last_user_text):
        dist_text = _extract_last_distance_from_history(message_history) or "some distance"
        if _assistant_mentioned_pass_by(message_history):
            return f"I meant that I walked past the premises along the street earlier. When the incident happened, I was about {dist_text} away down the road, so I couldn't see much inside."
        else:
            return f"At the moment I saw the person, I was about {dist_text} away down the road."

    has_assistant = any(m.get("role") == "assistant" for m in message_history)

    if _is_job_enjoyment_prompt(last_user_text):
        today_nz = datetime.now().strftime("%-d %B %Y")
        base_msgs_for_micro = [
            {"role": "system", "content": f"Today is {today_nz}."},
            BEHAVIOUR_RULES,
            persona_background_message(filtered_persona),
        ]
        return _answer_job_enjoyment_via_llm(base_msgs_for_micro, persona_data, last_user_text)

    # "What did you do after he drove away?" — a common wrap-up question that needs a
    # concise factual answer (called police, checked on victim, etc.) rather than a narrative.
    AFTER_YOU_DID_RE = re.compile(
        r"\bwhat\s+did\s+you\s+do\s+(after|then|next|afterwards)\b.*"
        r"(?:(he|she|they|the\s+(man|woman|person|suspect|offender|teen|team|boy|girl))\s+)?"
        r"(left|drove\s+away|drove\s+off|ran\s+off|ran\s+away|took\s+off|left\s+the\s+scene)",
        re.IGNORECASE,
    )
    if AFTER_YOU_DID_RE.search(last_user_text.lower()):
        facts_text = "\n".join((f.get("fact", "") for f in filtered_persona.get("facts_to_provide", []) if isinstance(f, dict)))
        facts_low = facts_text.lower()
        steps = []
        if re.search(r"\bput\s+(it|the\s+fire)\s+out\b", facts_low):
            steps.append("put the fire out")
        if re.search(r"\bcalled?\s+(111|105)\b|\bcall\s+(111|105)\b", facts_low):
            steps.append("called police")
        if re.search(r"\bchecked\s+on\s+(the\s+)?(owner|person|people|staff)\b", facts_low):
            steps.append("checked on the people nearby")
        if steps:
            s = ", ".join(steps[:2]) + (", and " + steps[2] if len(steps) > 2 else "")
            return ("I " + s + ".").replace("..", ".")
        return "I left the area."

    def _prev_answer_was_incident(msgs):
        prev = next((m.get("content", "") for m in reversed(msgs[:-1]) if m.get("role") == "assistant"), "")
        return _contains_unsolicited_narrative(prev)
    if re.search(r"\b(what\s+did\s+you\s+do\s+(next|then)|and\s+what\s+did\s+you\s+do\s+next|what\s+happened\s+then)\b", last_user_text.lower()) and _prev_answer_was_incident(message_history):
        pass

    if _is_smalltalk_question(last_user_text):
        if _is_how_are_you_question(last_user_text):
            return "I'm good, thank you."
        if _is_hows_day_question(last_user_text):
            return "Great, thanks."
        return "Good, thanks."

    # Tag questions ("Is that correct?", "Right?") always get a simple affirmation.
    # Confirming the interviewer's summary rather than re-answering avoids redundant repetition.
    if _is_confirmation_question(last_user_text):
        return "Yes, that's correct."

    if not has_assistant and not _looks_like_question(last_user_text) and not _is_persona_assertion(last_user_text):
        t = last_user_text.lower()
        if "thank" in t:
            return "You're welcome."
        if re.search(r"\b(hi|hello|kia ora|morning|afternoon|evening)\b", t):
            return "I'm ready when you are."
        return "I'm ready when you are."

    # "Thanks — can you tell me more about the car?" → acknowledge the thanks, then answer the question.
    # Pure "thank you" with no question gets a short response and exits early.
    has_thank_you = has_assistant and re.search(r"\b(thanks|thank\s+you)\b", last_user_text.lower())
    prepend_thanks_response = False
    if has_thank_you:
        if not _looks_like_question(last_user_text):
            return "You're welcome."
        else:
            prepend_thanks_response = True

    if _is_trivial_pref_question(last_user_text):
        return "I'm not sure how that's relevant."

    # Assemble the internal system context that guides the LLM without being spoken aloud.
    # Facts are passed in chronological order so the LLM can honour the timeline constraint.
    # Withholds and triggers are included so the LLM can modulate evasiveness when appropriate.
    internal_sections = []
    if facts.strip():
        internal_sections.append(
            "Facts to share if asked:\n" + facts.strip() + "\n\n"
            "IMPORTANT: These facts are listed in chronological order (the sequence in which events occurred). "
            "When describing what happened, maintain this chronological sequence. Do not rearrange events or suggest "
            "that later events happened before earlier ones. If asked about a specific moment in time, only reference "
            "facts that occurred at or before that moment."
        )
    if withholds.strip():
        internal_sections.append("Hidden motivations (do not volunteer):\n" + withholds.strip())
    if triggers.strip():
        internal_sections.append("Trigger topics (may cause hesitation):\n" + triggers.strip())
    internal_blob = "\n\n".join(internal_sections) if internal_sections else ""
    internal_note_msg = {"role": "system", "content": internal_blob} if internal_blob else None

    # "Where was the car parked?" — if the facts show the car was driven away, it was never parked.
    if re.search(r"\bwhere\s+was\s+the\s+(car|vehicle)\s+(parked|left|located)\b", q):
        if re.search(r"\b(drove|driving|drove\s+away|sped\s+away)\b", facts_low):
            return "I didn't see it parked. I saw them drive away."
        return "I'm not sure."

    # Direction questions — resolve in priority order: cardinal → away-from → towards →
    # "in the direction of" → street name → generic speed description → unsure.
    if re.search(r"\bwhat\s+direction\s+did\s+(he|she|they|the\s+\w+)(\s+\w+)?\s+(drive|go|head|leave|run|walk)(\s+off|\s+away)?(\s+to|\s+in|\s+towards?)?\b", q) or \
       re.search(r"\b(which\s+way|what\s+way|which\s+direction)\s+did\s+(he|she|they|the\s+\w+)(\s+\w+)?\s+(drive|go|head|leave|run|walk)(\s+off|\s+away)?(\s+to|\s+in|\s+towards?)?\b", q):
        # Cardinal directions are the most specific — prefer these over landmark-relative descriptions.
        cardinal_match = re.search(r"\b(?:head|headed|went|ran|drove|walked|fled|sped)\s+(north|south|east|west)(?:\s+down)?\b", facts_low)
        if cardinal_match:
            direction = cardinal_match.group(1)
            response = f"He went {direction}."
            return response

        # "away from" descriptions (e.g., "drove away from the dairy") are next most specific.
        away_from_match = re.search(r"\b(?:drove|headed|went|ran|walked|fled|sped)\s+away\s+from\s+(?:the\s+)?([A-Za-z][A-Za-z'\-\s]+?)(?:\s+and|\s+at|\s+in|\.|,|$)", facts_text, re.IGNORECASE)
        if away_from_match:
            location = away_from_match.group(1).strip()
            response = f"He drove away from the {location}."
            return response

        # "towards" expressions — the (?:off|away)? group handles "drove off towards" as well as plain "drove towards".
        towards_match = re.search(r"\b(?:drove|headed|went|ran|walked|fled|sped)\s+(?:(?:off|away)\s+)?towards?\s+(?:the\s+)?([A-Za-z][A-Za-z'\-\s]+?)(?:\s+and|\s+at|\s+in|\.|,|$)", facts_text, re.IGNORECASE)
        if towards_match:
            location = towards_match.group(1).strip()
            response = f"He drove towards {location}."
            return response

        # "in the direction of [location]" — e.g. "drove off in the direction of Murphy Street"
        direction_of_match = re.search(r"\bin\s+the\s+direction\s+of\s+(?:the\s+)?([A-Za-z][A-Za-z'\-\s]+?)(?:\s+and|\s+at|\s+in|\.|,|$)", facts_text, re.IGNORECASE)
        if direction_of_match:
            location = direction_of_match.group(1).strip()
            response = f"He went in the direction of {location}."
            return response

        # "drove/ran off down/up/along/onto [street]" — e.g. "drove off down Murphy Street"
        street_match = re.search(r"\b(?:drove|headed|went|ran|walked|fled|sped)\s+(?:off\s+)?(?:up|down|along|onto)\s+(?:the\s+)?([A-Za-z][A-Za-z'\-\s]+?)(?:\s+and|\s+at|\s+in|\.|,|$)", facts_text, re.IGNORECASE)
        if street_match:
            location = street_match.group(1).strip()
            response = f"He went down {location}."
            return response

        # "headed for [location]" — e.g. "headed for the exit", "ran for Murphy Street"
        headed_for_match = re.search(r"\b(?:drove|headed|went|ran|walked|fled|sped)\s+(?:off\s+)?for\s+(?:the\s+)?([A-Za-z][A-Za-z'\-\s]+?)(?:\s+and|\s+at|\s+in|\.|,|$)", facts_text, re.IGNORECASE)
        if headed_for_match:
            location = headed_for_match.group(1).strip()
            response = f"He headed for {location}."
            return response

        # If facts only say "drove away at speed" with no directional qualifier, the witness genuinely doesn't know.
        if re.search(r"\bdrove\s+(?:away|off)\s+at\s+speed\b", facts_low) and not re.search(r"\b(?:from|towards?)\b", facts_low):
            response = "They drove away at speed. I didn't see which direction."
            return response

        return "I'm not sure."

    # "Did you go into the shop/store/house/property at all?"
    if re.search(r"\bdid\s+you\s+(go|walk|enter|step)\s+(into|inside|in)\s+(the\s+)?(shop|store|premises|building|house|property)\b", q):
        # Check if facts say witness went inside
        if re.search(r"\bi\s+(went|walked|entered|stepped)\s+(into|inside|in)\s+(the\s+)?(shop|store|premises|building|house|property)\b", facts_low):
            return "Yes."
        # Check if facts say witness checked on someone (extract who they checked on)
        checked_match = re.search(r"\bchecked\s+on\s+(the\s+)?(\w+(?:\s+\w+)?)\b", facts_low)
        if checked_match:
            who = checked_match.group(2)
            return f"Yes, I went in to check on the {who}."
        return "No, I didn't go inside."

    # "How do you know it was 5:30pm?" — extract the reason field from the matching time fact
    # and convert it to natural first-person speech rather than delegating to the LLM,
    # which may invent a plausible-sounding but incorrect explanation.
    if re.search(r"\b(how|why)\s+(?:are\s+you|can\s+you\s+be|do\s+you\s+know\s+you'?re)\s+(?:so\s+)?(?:sure|certain)\s+(?:of\s+|about\s+|that\s+it\s+was\s+)?(?:the\s+|that\s+|this\s+)?(?:time|\d{1,2}[:.\\s]\d{2}\s*(?:am|pm|a\.m|p\.m)?)\b", q, re.IGNORECASE) or \
       re.search(r"\b(how|why)\s+do\s+you\s+know\s+(?:the\s+|it\s+was\s+|that\s+)?(?:the\s+|that\s+)?(?:time|\d{1,2}[:.\\s]\d{2}\s*(?:am|pm|a\.m|p\.m)?)\b", q, re.IGNORECASE):
        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                reason_text = fact_item.get("reason", "")
                # Find the first fact that contains a clock time — that's the one being asked about.
                if re.search(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)\b", fact_text, re.IGNORECASE) or \
                   re.search(r"\bat\s+\d{1,2}\s*(?:am|pm)\b", fact_text, re.IGNORECASE):
                    if reason_text and reason_text.lower() not in ["", "none"]:
                        reason_low = reason_text.lower()

                        # Specific time-verification methods are checked first. The generic "routine" and
                        # "clear memory" patterns are intentionally last to avoid premature matches
                        # on reasons like "regular station noted time" that contain those keywords incidentally.

                        # Pattern: "checked watch/phone/clock"
                        if re.search(r"\bchecked\s+(my\s+)?(watch|phone|clock|time)\b", reason_low):
                            match = re.search(r"\bchecked\s+(my\s+)?(watch|phone|clock)\b", reason_low)
                            if match:
                                device = match.group(2)
                                return f"I checked my {device}."
                            return "I checked my watch."

                        # Pattern: "looked at clock", "noted time on wall clock", "saw time on phone", etc.
                        # Longer device names (mobile phone, cell phone) are matched before the shorter "phone"
                        # to avoid accidentally truncating the description.
                        if re.search(r"\b(looked|glanced|noted|saw)\s+(?:at\s+|the\s+)?(?:time\s+on\s+)?(?:my\s+|the\s+)?(mobile\s+phone|cell\s+phone|wall\s+clock|watch|phone|clock|time)\b", reason_low):
                            match = re.search(r"\b(looked|glanced|noted|saw)\s+(?:at\s+|the\s+)?(?:time\s+on\s+)?(?:my\s+|the\s+)?(mobile\s+phone|cell\s+phone|wall\s+clock|watch|phone|clock)\b", reason_low)
                            if match:
                                verb = match.group(1)
                                device = match.group(2) if match.group(2) else "time"
                                # "noted time on wall clock" → "I noted the time on the wall clock"
                                if verb == "noted" and "wall clock" in device:
                                    return "I noted the time on the wall clock."
                                elif verb == "noted" and "mobile phone" in device:
                                    return "I noted the time on my mobile phone."
                                elif verb == "noted" and "cell phone" in device:
                                    return "I noted the time on my cell phone."
                                elif verb == "saw" and "phone" in device:
                                    return "I saw the time on my phone."
                                elif verb == "saw":
                                    return f"I saw the time on the {device}."
                                elif verb == "noted":
                                    return f"I noted the {device}."
                                elif verb == "glanced":
                                    return f"I glanced at the {device}."
                                else:
                                    return f"I looked at the {device}."
                            return "I looked at the time."

                        # Pattern: "heard time on radio" / "radio announced" / "news said"
                        if re.search(r"\b(heard|radio|news)\b.*\b(announced|said|reported|mentioned)\b", reason_low) or \
                           re.search(r"\b(radio|news)\s+(announced|said|gave|reported)\s+(the\s+)?time\b", reason_low):
                            if "radio" in reason_low:
                                return "I heard the time announced on the radio."
                            elif "news" in reason_low:
                                return "I heard the time on the news."
                            return "I heard the time announced."

                        # Pattern: "church bells" / "clock tower" / "town clock chimed"
                        if re.search(r"\b(church\s+bells?|clock\s+tower|town\s+clock|bells?\s+chimed?)\b", reason_low):
                            if re.search(r"\bchimed?\b", reason_low):
                                return "I heard the clock tower chime."
                            return "I heard the church bells."

                        # Pattern: "alarm went off" / "phone alarm" / "reminder"
                        if re.search(r"\b(alarm|reminder)\s+(went\s+off|rang|sounded)\b", reason_low):
                            if "phone" in reason_low:
                                return "My phone alarm went off at that time."
                            return "My alarm went off at that time."

                        # Pattern: "asked someone" / "someone told me" / "checked with colleague"
                        if re.search(r"\b(asked|someone\s+told|colleague\s+said|confirmed\s+with)\b", reason_low):
                            return "I asked someone what time it was."

                        # Pattern: "receipt shows" / "timestamp" / "CCTV showed"
                        if re.search(r"\b(receipt|timestamp|cctv|recording|footage)\b", reason_low):
                            if "receipt" in reason_low:
                                return "The receipt shows the time."
                            elif "cctv" in reason_low or "footage" in reason_low:
                                return "The CCTV footage shows the time."
                            return "The timestamp confirms it."

                        # Pattern: mentions TV show, appointment, schedule
                        if re.search(r"\b(tv\s+show|appointment|scheduled?|meeting)\b", reason_low):
                            if "tv" in reason_low:
                                return "I know because my TV show starts at that time."
                            return "I remember because I had something scheduled at that time."

                        # Pattern: "routine walk home" / "usual route" / "regular time".
                        # Routine-based patterns are checked before the generic "clear memory" guard —
                        # "routine" can appear incidentally in reasons that do not describe a habitual schedule.
                        if re.search(r"\broutine\s+walk\b", reason_low) or re.search(r"\busual.*walk\b", reason_low):
                            return "I'm certain because it's my usual walk home from work at that time."
                        # "routine" only matches here when the reason explicitly links it to a work schedule.
                        if re.search(r"\b(my\s+)?routine\b", reason_low) and re.search(r"\bwork\b", reason_low):
                            return "It's my regular routine."
                        if re.search(r"\b(regular|usual)\s+time\b", reason_low):
                            return "It's my regular routine."

                        # "clear memory" is the most generic fallback — intentionally checked last.
                        if re.search(r"\bclear\s+memory\b", reason_low):
                            return "I have a clear memory of what time it was."

                        # If the reason is already written in first person, use it directly.
                        if reason_text.startswith("I "):
                            return reason_text.capitalize() if not reason_text[0].isupper() else reason_text

                        return "I'm certain about the time."
                    else:
                        return "I'm certain about the time."
        return "I'm not sure."

    # "Did you recognise the person?" — default to No if the facts don't mention recognition at all.
    # The LLM will hallucinate "Yes" responses without explicit grounding, so we intercept here.
    # Object-recognition questions (car make, registration, etc.) are excluded — they go to the LLM.
    if re.search(r"\brecogni[sz]e?\b", q, re.IGNORECASE):
        if not re.search(r"\b(car|vehicle|plate|number\s*plate|registration|building|place|street|road|shop|store)\b", q, re.IGNORECASE):
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_low = fact_item.get("fact", "").lower()
                    if re.search(r"\brecogni[sz]e?\b", fact_low):
                        # Explicit recognition fact found — delegate to LLM for nuanced handling.
                        break
            else:
                response = "No, I didn't recognise them."
                return response

    # "How do you know it was a Mazda 2?" — extract and convert the certainty reason from the matching fact.
    # Reason fields describe the basis for the witness's knowledge (e.g., "I'm familiar with Mazda cars").
    how_know_match = re.search(
        r"\b(?:how|why)\s+(?:do\s+you\s+know|can\s+you\s+(?:be\s+)?(?:so\s+)?(?:sure|certain)|are\s+you\s+(?:so\s+)?(?:sure|certain))\s+"
        r"(?:it\s+was\s+|that\s+it\s+was\s+|that\s+|it'?s\s+)?(?:a\s+|an\s+|the\s+)?(.+?)[\?\.]*$",
        q, re.IGNORECASE
    )
    if how_know_match:
        subject = how_know_match.group(1).strip().lower()
        # Clean up common trailing words
        subject = re.sub(r"\s+(then|there|though)$", "", subject)


        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                reason_text = fact_item.get("reason", "")
                fact_low = fact_text.lower()

                # Match facts by keyword overlap — require at least one meaningful word (>2 chars)
                # from the question subject to appear in the fact text.
                subject_words = subject.split()
                match_count = sum(1 for word in subject_words if len(word) > 2 and word in fact_low)

                if match_count >= 1 or subject in fact_low:
                    if reason_text and reason_text.lower() not in ["", "none"]:
                        reason_low = reason_text.lower()

                        # Convert the reason field to first-person speech using pattern priority order.
                        # Pattern: "familiar with X" → "I'm familiar with X"
                        if re.search(r"\bfamiliar\s+with\b", reason_low):
                            familiar_match = re.search(r"\bfamiliar\s+with\s+(.+?)(?:,|$)", reason_low)
                            if familiar_match:
                                familiar_with = familiar_match.group(1).strip()
                                response = f"I'm familiar with {familiar_with}."
                                return response

                        # Pattern: "good view" / "clear view" / "close enough"
                        if re.search(r"\b(good|clear|close)\s+(view|look|enough)\b", reason_low):
                            response = "I got a good look at it."
                            return response

                        # Pattern: "noticed" / "saw clearly" / "observed"
                        if re.search(r"\b(noticed|saw\s+clearly|observed|direct\s+line\s+of\s+sight)\b", reason_low):
                            response = "I saw it clearly."
                            return response

                        # Pattern: "recognized" / "knew from before"
                        if re.search(r"\b(recogni[sz]ed?|knew|know)\b", reason_low):
                            response = "I recognised it."
                            return response

                        # Pattern: already starts with "I " - use as-is
                        if reason_text.strip().startswith("I "):
                            response = reason_text.strip()
                            if not response.endswith("."):
                                response += "."
                            return response

                        # Short reasons can be wrapped as "Because [reason]." — long ones fall back to generic.
                        if len(reason_text) < 60:
                            response = f"Because {reason_text.lower().rstrip('.')}."
                            return response

                        # Fallback for long reasons
                        response = "I'm certain of what I saw."
                        return response

        # No matching fact found — the LLM won't do better without grounding, so return uncertain.
        return "I'm not sure."

    # Handle "What was happening [as/when/while] [context]?" questions
    # Example: "What was happening as you drove past the house?"
    what_happening_match = re.search(r"\bwhat\s+was\s+happening\s+(as|when|while)\s+(?:you\s+)?(.+?)[\?\.]*$", q, re.IGNORECASE)
    if what_happening_match:
        temporal_context = what_happening_match.group(2).strip()  # e.g., "you drove past the house"


        # Search facts for mentions of this context (e.g., "drove past")
        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                fact_low = fact_text.lower()

                # Extract key verbs/nouns from the temporal context
                # Common patterns: "drove past", "walked by", "arrived at", "left"
                context_lower = temporal_context.lower()

                # Check if this fact contains the temporal context
                # Look for "as I [action]" or "when I [action]" or direct mention
                if re.search(r"\b(as|when|while)\s+i\s+", fact_low):
                    # Extract what happened: pattern "as I [action], I [observation]"
                    observation_match = re.search(r"\b(?:as|when|while)\s+i\s+[^,]+,\s+i\s+(.+?)(?:\.|$)", fact_text, re.IGNORECASE)
                    if observation_match:
                        observation = observation_match.group(1).strip()
                        response = f"I {observation}."
                        return response

                # Alternative pattern: Check if fact describes what witness saw during the action mentioned
                # Example: context="drove past the house" → fact mentions "drove past" + what they saw
                # Look for key action words from context in the fact
                action_words = re.findall(r"\b(drove|walked|ran|arrived|left|passed|went|came)\b", context_lower)
                for action in action_words:
                    if action in fact_low:
                        # This fact is likely relevant - extract what they saw/observed
                        # Pattern: "I [saw/heard/noticed] [something]"
                        saw_match = re.search(r"\bi\s+(saw|heard|noticed|observed)\s+(.+?)(?:\s+and\s+saw\s+(.+?))?(?:\.|,|$)", fact_text, re.IGNORECASE)
                        if saw_match:
                            observation1 = saw_match.group(2).strip()
                            observation2 = saw_match.group(3)

                            if observation2:
                                response = f"I saw {observation1} and saw {observation2}."
                            else:
                                response = f"I saw {observation1}."

                            return response

        return "I'm not sure."

    # Handle "Where were you when [event] happened?" or "Where were you as the incident happened?" questions
    # Also handle typos like "when the car park as the incident happened"
    # Extract location from facts mentioning position/distance
    if re.search(r"\bwhere\s+were\s+you\s+(when|as|during|at)\s+", q, re.IGNORECASE):
        # Check for generic "the incident/event happened" pattern (flexible to handle typos)
        if re.search(r"\b(incident|event|this)\s+(happened|occurred|took\s+place)\b", q, re.IGNORECASE):
            # Search through facts for location/distance information
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    fact_low = fact_text.lower()

                    # Look for distance + position pattern (most specific)
                    distance_position = re.search(r"\bi\s+was\s+about\s+(\d+)\s+metres?\s+away\s+(?:near|from|at|by)\s+(?:the\s+)?([^,\.]+?)(?:\s+when|\s+and|,|\.|$)", fact_low)
                    if distance_position:
                        distance = distance_position.group(1)
                        position = distance_position.group(2).strip()
                        response = f"I was about {distance} metres away near the {position}."
                        return response

                    # Look for simple position pattern
                    position_match = re.search(r"\bi\s+was\s+(?:about\s+)?(\d+)\s+metres?\s+away\s+near\s+(?:the\s+)?([^,\.]+)", fact_low)
                    if position_match:
                        distance = position_match.group(1)
                        position = position_match.group(2).strip()
                        response = f"About {distance} metres away near the {position}."
                        return response

        # Original specific event handler
        event_match = re.search(r"\bwhere\s+were\s+you\s+when\s+(the\s+)?(\w+)\s+(happened|occurred|took\s+place)\b", q, re.IGNORECASE)
        if event_match:
            event_word = event_match.group(2).lower()  # e.g., "explosion", "robbery", "fire"

            # Search through facts for one mentioning this event
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    fact_low = fact_text.lower()

                    # Check if this fact mentions the event
                    if event_word in fact_low or re.search(r"\b(bang|explosion|fire|incident|robbery|assault|fight|attack)\b", fact_low):
                        # Extract location from "I was [preposition] [location] when..." pattern
                        # Capture the preposition and location separately to reconstruct properly
                        location_match = re.search(r"\bi\s+was\s+(on\s+the\s+|in\s+the\s+|at\s+the\s+|inside\s+|in\s+|on\s+|at\s+)([^,]+?)\s+when\b", fact_low)
                        if location_match:
                            preposition = location_match.group(1).strip()
                            location = location_match.group(2).strip()
                            response = f"I was {preposition} {location}."
                            return response

                        # Alternative pattern: "[time], I was [location] when [event]" or "I was [location], and [event]"
                        alt_match = re.search(r"\bi\s+was\s+(on\s+the\s+|in\s+the\s+|at\s+the\s+|inside\s+|in\s+|on\s+|at\s+)([^,]+?),?\s+(?:and\s+)?(?:when|as|and)\b", fact_low)
                        if alt_match:
                            preposition = alt_match.group(1).strip()
                            location = alt_match.group(2).strip()
                            response = f"I was {preposition} {location}."
                            return response

            response = "I'm not sure."
            return response

    # Handle "Where were you in/at [location]?" questions (e.g., "Where were you in the car park?")
    if re.search(r"\bwhere\s+were\s+you\s+(in|at)\s+(the\s+)?(\w+(?:\s+\w+)?)\b", q, re.IGNORECASE):
        location_match = re.search(r"\bwhere\s+were\s+you\s+(in|at)\s+(the\s+)?(\w+(?:\s+\w+)?)\b", q, re.IGNORECASE)
        if location_match:
            context_location = location_match.group(3).lower()  # e.g., "car park", "building", "store"
            # Also check for "carpark" (one word variant)
            context_variants = [context_location, context_location.replace(" ", "")]

            # Search ALL facts for specific position details (near/by/at)
            # IMPORTANT: Only match facts that describe the WITNESS's location (start with "I was" or "I had")
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    fact_low = fact_text.lower()

                    # Require first-person witness location indicators anywhere in the fact.
                    # Facts may start with a timestamp, e.g. "At about 10:15pm I was at the front bar".
                    # re.search (not re.match) handles that case while still filtering third-party
                    # descriptions like "saw a man enter by the driver's door".
                    if not re.search(r"\bi\s+(was|had|stood|walked|went|ran|drove)", fact_low):
                        continue

                    # Pattern: "I was at/in the [position] of the [venue]" — extract just the position
                    # within the venue, not the venue name itself.
                    # e.g. "At 10:15pm I was at the front bar of The Kingfisher Pub" → "I was at the front bar."
                    position_of_venue = re.search(r"\bi\s+was\s+(at|in)\s+the\s+([^,\.]+?)\s+of\s+the\b", fact_low)
                    if position_of_venue:
                        preposition = position_of_venue.group(1)
                        position = position_of_venue.group(2).strip()
                        return f"I was {preposition} the {position}."

                    # Pattern: "about X metres away near/from [place]"
                    distance_position = re.search(r"\babout\s+(\d+)\s+metres?\s+away\s+(?:near|from|at)\s+(?:the\s+)?([^,\.]+?)(?:\s+when|\s+and|,|\.|$)", fact_low)
                    if distance_position:
                        distance = distance_position.group(1)
                        position = distance_position.group(2).strip()
                        return f"I was about {distance} metres away near the {position}."

                    # Pattern: "near the [place]" or "by the [place]" or "at the [place]"
                    position_match = re.search(r"\b(near|by|at)\s+the\s+([^,\.]+?)(?:\s+when|\s+and|,|\.|$)", fact_low)
                    if position_match:
                        preposition = position_match.group(1)
                        position = position_match.group(2).strip()
                        return f"I was {preposition} the {position}."

            return "I'm not sure."

    # Handle "Where did you evacuate to?" questions
    if re.search(r"\bwhere\s+did\s+(?:you|we)\s+evacuate\s+(?:to)?\b", q, re.IGNORECASE):
        # Search for evacuation location in facts
        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                fact_low = fact_text.lower()
                # Pattern: "evacuated to [location]" or "We evacuated to [location]"
                evac_match = re.search(r"\bevacuated\s+to\s+(?:the\s+)?(.+?)(?:\s+by\s+the\s+.+)?(?:\.|,|$)", fact_low)
                if evac_match:
                    location = evac_match.group(1).strip()
                    # Include "by the X" if present (e.g., "carpark by the main gate")
                    full_match = re.search(r"\bevacuated\s+to\s+(?:the\s+)?(.+?)(?:\.|,|$)", fact_low)
                    if full_match:
                        full_location = full_match.group(1).strip()
                        return f"We evacuated to the {full_location}."
        return "I'm not sure."

    # Handle "Did you wait for police/ambulance?" questions
    # Note: "wait until police arrived" means waiting for them to arrive IN PERSON, not just speaking on phone
    if re.search(r"\bdid\s+you\s+wait\s+(?:for|until)\s+(?:the\s+)?(police|ambulance|officer)(?:\s+(?:to\s+)?arrive)?\b", q, re.IGNORECASE):
        # Search for information about waiting
        waited_for_arrival = False
        left_without_waiting = False

        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                fact_low = fact_text.lower()

                # Pattern: "waited until/for police/ambulance arrived"
                if re.search(r"\bwaited?\s+(?:until|for)\s+(?:the\s+)?(police|ambulance|officer)(?:\s+(?:to\s+)?arrive)", fact_low):
                    waited_for_arrival = True

                # Pattern: "did not wait" or "left before" or "went home (after calling)"
                if re.search(r"\b(did\s+not|didn'?t)\s+wait\b", fact_low):
                    left_without_waiting = True
                if re.search(r"\bleft\s+before\b", fact_low):
                    left_without_waiting = True

                # Pattern: "called police/105/111... and went home" (implies didn't wait for arrival)
                if re.search(r"\b(called|phoned|rang|contacted)\s+(?:(?:the\s+)?(?:police|nz\s+police)|111|105)\b", fact_low):
                    # Check if same fact mentions going home/leaving after
                    if re.search(r"\b(?:and|then)?\s+(?:went|headed)\s+home\b", fact_low):
                        left_without_waiting = True
                    if re.search(r"\b(?:and|then)?\s+left\s+(?:the\s+)?(?:area|scene)\b", fact_low):
                        left_without_waiting = True

                # Pattern: "gave details... and went home" or "spoke to police... and went home" (implies phone conversation, not waiting)
                # This covers cases where witness had phone conversation with police then left
                if re.search(r"\b(gave|provided|left)\s+(?:my\s+)?details?\b", fact_low):
                    # If mentions police/105/111 in same fact AND going home, they didn't wait
                    if re.search(r"\b(?:police|nz\s+police|105|111)\b", fact_low):
                        if re.search(r"\b(?:and|then)?\s+(?:went|headed)\s+home\b", fact_low):
                            left_without_waiting = True

                # Pattern: "reassured/spoke to police... and went home"
                if re.search(r"\b(reassured|spoke\s+to|talked\s+to)\s+(?:the\s+)?(?:police|nz\s+police)\b", fact_low):
                    if re.search(r"\b(?:and|then)?\s+(?:went|headed)\s+home\b", fact_low):
                        left_without_waiting = True

        if waited_for_arrival:
            response = "Yes."
            return response
        if left_without_waiting:
            response = "No, I went home."
            return response

        return "I'm not sure."

    # Handle "Who did you [verb] [object] with?" questions
    # Examples: "Who did you leave your details with?", "Who did you speak with?"
    who_with_match = re.search(r"\bwho\s+did\s+you\s+(leave|give|provide|share|speak|talk|discuss)\s+(?:your\s+)?(\w+)?\s+(?:to|with)\b", q, re.IGNORECASE)
    if who_with_match:
        action = who_with_match.group(1)  # e.g., "leave", "give"
        object_word = who_with_match.group(2) if who_with_match.group(2) else ""  # e.g., "details"


        # Search facts for pattern: "I [action]... [entity]"
        # Examples: "I gave my details... Police", "I left my details with staff"
        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                fact_low = fact_text.lower()

                # Look for the action verb in the fact
                if re.search(rf"\b{re.escape(action)}\b", fact_low):
                    # Extract entity after common patterns
                    # Pattern 1: "gave/left details... Police"
                    entity_match = re.search(r"\b(?:to|with|the)\s+(?:the\s+)?(NZ\s+)?([Pp]olice|[Ss]taff|[Cc]lerk|[Oo]fficer|[Mm]anager|[Ss]ecurity|[Rr]eceptionist)\b", fact_text)
                    if entity_match:
                        entity = entity_match.group(2)  # e.g., "Police"
                        # Capitalize properly
                        if entity.lower() == "police":
                            response = "The police."
                        else:
                            response = f"The {entity.lower()}."

                        return response

                    # Pattern 2: Check if "105" or "111" is mentioned (implies police)
                    if re.search(r"\b(105|111)\b", fact_text):
                        response = "The police."
                        return response

        return "I'm not sure."

    # Handle "Did you call 111/105/police?" questions
    if re.search(r"\bdid\s+you\s+(?:call|phone|ring|dial|contact)\s+(?:the\s+)?(111|105|police|emergency\s+services)\b", q, re.IGNORECASE):
        # Search for information about calling police/emergency
        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                fact_low = fact_text.lower()
                # Pattern: "called 111" or "called 105" or "phoned police"
                # IMPORTANT: Can appear anywhere in the fact, even after "and"
                if re.search(r"\b(called?|phoned?|rang|dialed?|contacted?)\s+(?:the\s+)?(111|105|police|emergency\s+services)\b", fact_low):
                    return "Yes."
                # Pattern: "didn't call" or "did not phone"
                if re.search(r"\b(didn'?t|did\s+not|never)\s+(call|phone|ring|dial|contact)\b", fact_low):
                    return "No."
        return "I'm not sure."

    # Handle "Was anyone injured?" questions
    if re.search(r"\bwas\s+(?:anyone|anybody)\s+(?:injured|hurt)\b", q, re.IGNORECASE) or \
       re.search(r"\b(?:any|were\s+there\s+any)\s+(?:injuries|people\s+injured)\b", q, re.IGNORECASE):
        # Search for injury information in facts
        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                fact_low = fact_text.lower()
                # Pattern: "didn't see anyone injured" or "no one injured"
                if re.search(r"\b(didn'?t\s+see\s+anyone|no\s+one|nobody)\s+(injured|hurt)\b", fact_low):
                    return "No, I didn't see anyone injured."
                # Pattern: "no injuries"
                if re.search(r"\bno\s+injuries\b", fact_low):
                    return "No."
                # Pattern: "X was injured"
                if re.search(r"\b(someone|person|man|woman|people)\s+(was|were)\s+(injured|hurt)\b", fact_low):
                    return "Yes, someone was injured."
        return "I'm not sure."

    # Handle "Where were you when you [first] heard [something]?" questions
    # Extract location from facts before the hearing event
    if re.search(r"\bwhere\s+were\s+you\s+when\s+you\s+(?:first\s+)?heard\b", q, re.IGNORECASE):
        facts_list = filtered_persona.get("facts_to_provide", [])
        # Find the first fact that mentions hearing
        hearing_fact_index = -1
        for i, fact_item in enumerate(facts_list):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                if re.search(r"\bheard\b", fact_text.lower()):
                    hearing_fact_index = i
                    break

        # If we found a hearing fact, look for location in that fact and earlier facts
        if hearing_fact_index >= 0:
            # First check if the hearing fact itself contains location
            hearing_fact = facts_list[hearing_fact_index].get("fact", "")
            hearing_low = hearing_fact.lower()

            # Check if hearing fact has location info (e.g., "I was at home... and heard")
            if re.search(r"\bi\s+was\s+(?:at\s+home|in\s+my|sitting|standing)", hearing_low):
                # Extract from the hearing fact itself
                if re.search(r"\bsitting\s+in\s+my\s+lounge\b", hearing_low):
                    if re.search(r"\bsliding\s+door\s+partly\s+open\b", hearing_low):
                        return "I was sitting in my lounge with the sliding door partly open."
                    return "I was sitting in my lounge."
                if re.search(r"\bat\s+home\b", hearing_low):
                    return "I was at home."

            # Otherwise, look in previous facts for location
            for i in range(hearing_fact_index - 1, -1, -1):
                if isinstance(facts_list[i], dict):
                    prev_fact = facts_list[i].get("fact", "")
                    prev_low = prev_fact.lower()
                    # Look for location patterns
                    if re.search(r"\bsitting\s+in\s+my\s+lounge\b", prev_low):
                        if re.search(r"\bsliding\s+door\s+partly\s+open\b", prev_low):
                            return "I was sitting in my lounge with the sliding door partly open."
                        return "I was sitting in my lounge."
                    if re.search(r"\bat\s+home\b", prev_low):
                        return "I was at home."
                    if re.search(r"\bin\s+my\s+(\w+)\b", prev_low):
                        location_match = re.search(r"\bin\s+my\s+(\w+)\b", prev_low)
                        if location_match:
                            return f"I was in my {location_match.group(1)}."

        return "I'm not sure."

    # Handle work schedule questions: "What time did you leave/finish work?"
    # Extract time from "walking home from work at X:XXpm" facts
    if re.search(r"\bwhat\s+time\s+did\s+you\s+(leave|finish|get\s+off)\s+work\b", q) or \
       re.search(r"\bwhen\s+did\s+you\s+(leave|finish|get\s+off)\s+work\b", q):
        # Look for facts mentioning "walking home from work" or similar with a time
        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "")
                fact_low = fact_text.lower()
                # Check if fact mentions walking/going home from work with a time
                if re.search(r"\b(walking|going|headed?)\s+home\b.*\bwork\b", fact_low) or \
                   re.search(r"\bwork\b.*\b(walking|going|headed?)\s+home\b", fact_low):
                    # Extract time from this fact
                    time_match = re.search(r"\b(\d{1,2}[:.]\d{2}\s*(?:am|pm))\b", fact_text, re.IGNORECASE)
                    if time_match:
                        time_str = time_match.group(1)
                        return f"I finished around {time_str}. That's when I was walking home."
        # Fallback if no time found
        return "I'm not sure exactly. It was around my usual finishing time."

    # Handle "When did [person/thing] arrive?" questions
    if re.search(r"\bwhen\s+did\s+(?:the\s+)?(\w+)\s+arrive\b", q, re.IGNORECASE):
        # Extract what/who arrived from the question
        arrive_match = re.search(r"\bwhen\s+did\s+(?:the\s+)?(\w+)\s+arrive\b", q, re.IGNORECASE)
        if arrive_match:
            who_what = arrive_match.group(1).lower()
            # Look through facts for "X arrived [timing]"
            for fact_item in filtered_persona.get("facts_to_provide", []):
                if isinstance(fact_item, dict):
                    fact_text = fact_item.get("fact", "")
                    fact_low = fact_text.lower()
                    # Check if this fact mentions the arrival
                    if re.search(rf"\b{who_what}\s+arrived\b", fact_low):
                        # Extract the timing - could be "after X", "before X", "at X time"
                        # Pattern: "arrived after/before [something]"
                        timing_match = re.search(r"\barrived\s+(after|before)\s+(.+?)(?:\.|,|$)", fact_low)
                        if timing_match:
                            when_word = timing_match.group(1)  # after/before
                            timing = timing_match.group(2).strip()
                            return f"They arrived {when_word} {timing}."
                        # Pattern: "arrived at [time]"
                        time_match = re.search(r"\barrived\s+at\s+(\d{1,2}[:.]\d{2}\s*(?:am|pm))\b", fact_low, re.IGNORECASE)
                        if time_match:
                            return f"They arrived at {time_match.group(1)}."
                        # Generic: just says "arrived"
                        return "They arrived after the incident."
        return "I'm not sure."

    # Handle open-ended "any other information" questions
    # These are too vague - encourage the interviewer to be more specific
    if re.search(r"\b(?:do\s+you\s+have|is\s+there)\s+(?:any|anything)\s+(?:other|else|more)\s+(?:information|details?)\b", q) or \
       re.search(r"\banything\s+else\s+(?:you\s+)?(?:can\s+)?(?:tell|remember|recall)\b", q) or \
       re.search(r"\bany\s+other\s+(?:information|details?)\b", q):
        return "I'm not sure. What sort of information are you wanting from me?"

    # "When could you see his face?" — temporal form must be handled before the yes/no face-visibility handler
    # below, otherwise the question pattern would match and return an incorrect Yes/No answer.
    q = (last_user_text or "").lower().strip()
    if re.search(r"\b(when|at\s+what\s+(?:point|time|moment))\s+(did|could)\s+you\s+(see|get\s+(?:to\s+)?(?:a\s+)?look\s+at)\s+(his|her|their|the\s+(?:person'?s|man'?s|woman'?s))\s+face\b", q):
        facts_text_guard = "\n".join((f.get("fact", "") for f in filtered_persona.get("facts_to_provide", []) if isinstance(f, dict)))
        facts_low_guard = facts_text_guard.lower()
        if re.search(r"\btook\s+(his|her|their|the)\s+mask\s+off\b", facts_low_guard) or re.search(r"\bremoved\s+(his|her|their|the)\s+mask\b", facts_low_guard):
            return "When he took off his mask as he ran past me."
        if re.search(r"\bmask\s+(came|fell)\s+off\b", facts_low_guard):
            return "When his mask came off."
        if re.search(r"\bran\s+past\b", facts_low_guard):
            return "As he ran past me."
        return "I'm not sure when exactly."

    # Yes/no face-visibility question — check facts for explicit negation first, then positive indicators.
    if re.search(r"\b(did|could)\s+you\s+(see|get\s+(?:to\s+)?(?:a\s+)?look\s+at)\s+(his|her|their|the\s+(?:person'?s|man'?s|woman'?s))\s+face\b", q):
        facts_text_guard = "\n".join((f.get("fact", "") for f in filtered_persona.get("facts_to_provide", []) if isinstance(f, dict)))
        facts_low_guard = facts_text_guard.lower()
        if re.search(r"\b(did\s+not|didn't|could\s+not|couldn't)\s+see\s+(his|the)\s+face\b", facts_low_guard) or re.search(r"\bdid\s+not\s+see\s+his\s+face\b", facts_low_guard):
            return "No, I didn't see his face."
        if re.search(r"\b(shaved\s+head|mousta?che|beard|scar|tattoo|glasses|white|brown|black)\b", facts_low_guard):
            return "Yes, briefly."
        return "I'm not sure."

    # Face description request → only from facts
    if re.search(r"\b(can\s+you\s+)?describe\s+(his|the)\s+face\b", q) or \
       re.search(r"\b(what\s+did\s+(his|the)\s+face\s+look\s+like)\b", q):
        facts_text_guard = "\n".join((f.get("fact", "") for f in filtered_persona.get("facts_to_provide", []) if isinstance(f, dict)))
        facts_low_guard = facts_text_guard.lower()
        bits = []
        if re.search(r"\bwhite\b", facts_low_guard):
            bits.append("He was white")
        if _facts_have_shaved_head(facts_low_guard):
            bits.append("had a shaved head")
        if re.search(r"\bmousta?che\b", facts_low_guard):
            m_col = re.search(r"\b(black|brown|blond|blonde|ginger|red|dark|light)\b\s+mousta?che\b|\bmousta?che\b\s*(?:that\s+was\s+)?\b(black|brown|blond|blonde|ginger|red|dark|light)\b", facts_low_guard)
            if m_col:
                col = (m_col.group(1) or m_col.group(2) or "").replace("gray", "grey")
                bits.append(f"and a {col} moustache" if bits else f"He had a {col} moustache")
            else:
                bits.append("and a moustache" if bits else "He had a moustache")
        if bits:
            sent = bits[0]
            if len(bits) > 1:
                sent += " " + " ".join(bits[1:])
            if not sent.endswith("."):
                sent += "."
            return sent
        return "I'm not sure."

    # Distinguishing marks — tattoos, scars, birthmarks, piercings, moles.
    # Hairstyle (shaved head, moustache, beard) is excluded; those are normal grooming features,
    # not marks in the forensic sense.
    if re.search(r"\b(did\s+(he|she|they)|does\s+(he|she))\s+have\s+(?:any\s+)?(?:distinguishing|distinctive|noticeable|visible)\s+(marks?|features?|characteristics?)\b", q) or \
       re.search(r"\b(any|were\s+there\s+any)\s+(?:distinguishing|distinctive|noticeable|visible)\s+(marks?|features?)\b", q) or \
       re.search(r"\b(did\s+you\s+(?:see|notice)\s+any)\s+(?:distinguishing|distinctive)?\s*(marks?|features?|tattoos?|scars?)\b", q):
        facts_text_guard = "\n".join((f.get("fact", "") for f in filtered_persona.get("facts_to_provide", []) if isinstance(f, dict)))
        facts_low_guard = facts_text_guard.lower()

        marks_found = []
        if re.search(r"\b(tattoo|tattooed)\b", facts_low_guard):
            tattoo_match = re.search(r"\b(tattoo|tattooed)\s+(?:of\s+)?(?:a\s+)?([^,\.]+)", facts_low_guard)
            if tattoo_match:
                marks_found.append(f"a tattoo of {tattoo_match.group(2).strip()}")
            else:
                marks_found.append("a tattoo")
        if re.search(r"\bscar\b", facts_low_guard):
            scar_match = re.search(r"\bscar\s+(?:on\s+)?(?:his\s+|her\s+|their\s+)?([^,\.]+)", facts_low_guard)
            if scar_match:
                marks_found.append(f"a scar on his {scar_match.group(1).strip()}")
            else:
                marks_found.append("a scar")
        if re.search(r"\bbirthmark\b", facts_low_guard):
            marks_found.append("a birthmark")
        if re.search(r"\b(piercing|pierced)\b", facts_low_guard):
            marks_found.append("a piercing")
        if re.search(r"\bmole\b", facts_low_guard):
            marks_found.append("a mole")

        if marks_found:
            response = f"Yes, he had {', '.join(marks_found)}."
            return response
        else:
            response = "I didn't notice any distinguishing marks."
            return response

    # ---------------- Tier 3: Assemble system messages for full LLM call ----------------
    base_msgs = [
        {"role": "system", "content": system_instructions},
        BEHAVIOUR_RULES,
        DEFLECT_EXAMPLE,
    ]
    if persona_prompt.strip():
        base_msgs.append({"role": "system", "content": f"SCENARIO: {persona_prompt.strip()}"})
    base_msgs.append(persona_background_message(filtered_persona))
    if internal_note_msg:
        base_msgs.append(internal_note_msg)
    # FIRST_TURN_RULE is intentionally omitted here — it is added in the default LLM path below
    # to avoid conflicting with the opener_rules that the constrained opener uses instead.

    # Spelling questions get an extra system rule to enforce the correct letter-by-letter format.
    if re.search(r"\b(?:how\s+do\s+you\s+spell|can\s+you\s+spell|please\s+spell|spell|spelling)\b", last_user_text, re.IGNORECASE):
        SPELLING_HINT = {
            "role": "system",
            "content": (
                "When asked to spell any word(s), respond only with the spelling of the requested word(s): "
                "UPPERCASE letters, hyphens between letters, and a single space between words. "
                "Do not include examples, placeholders, or extra commentary. If unclear which word to spell, ask a brief clarifying question."
            )
        }
        base_msgs.append(SPELLING_HINT)

    # "What happened after he ran past you?" — extracted from facts rather than delegated to the LLM
    # because the LLM tends to omit or invent the vehicle detail in this specific follow-up pattern.
    if re.search(r"\bwhat\s+happened\s+after\s+he\s+ran\s+past\s+you\b", q) or \
       re.search(r"\bafter\s+he\s+ran\s+past\s+you,\s*what\s+happened\b", q):
        facts_text2 = "\n".join((f.get("fact", "") for f in filtered_persona.get("facts_to_provide", []) if isinstance(f, dict)))
        facts_low2 = facts_text2.lower()
        route_name = _extract_named_route(facts_text2)
        vphrase = _vehicle_details_phrase(facts_low2)
        if route_name and re.search(r"\b(emerged|came\s+out)\b", facts_low2):
            if vphrase:
                # "Emerged in" means the suspect was already inside the vehicle (e.g., appeared from an alley
                # already seated), whereas "got into" describes boarding a vehicle that was waiting nearby.
                if re.search(r"\bemerged\s+in\s+", facts_low2):
                    return f"He went down {route_name} and emerged in {vphrase}."
                else:
                    return f"He went down {route_name} and got into {vphrase}."
            return f"He went down {route_name}."
        if vphrase:
            return f"He got into {vphrase}."
        return "I'm not sure."

    # Generic 'what happened next?' — advance along the facts-driven timeline
    # Allow optional filler words: "what actually happened next", "then what exactly happened", etc.
    if re.search(r"\b(what(\s+\w+)?\s+happened\s+next|then\s+what(\s+\w+)?\s+happened|and\s+what(\s+\w+)?\s+happened\s+next|what\s+next)\b", q, re.IGNORECASE):
        facts_text_all = "\n".join(("- " + f.get("fact", "") for f in filtered_persona.get("facts_to_provide", []) if isinstance(f, dict)))
        nxt = _next_timeline_step(message_history, facts_text_all)
        if nxt:
            return nxt
        return "I'm not sure."

    # First open-ended incident overview → constrained LLM, facts-only
    if _is_open_incident_question(last_user_text) and not any(
        _is_open_incident_question(m.get("content", ""))
        for m in message_history[:-1]
        if m.get("role") == "user"
    ):
        source_facts_text = facts.strip()
        # Extract incident timeframe from persona_prompt (e.g., "three days ago", "yesterday")
        incident_timeframe = _extract_incident_timeframe(persona_prompt)
        timeframe_instruction = ""
        if incident_timeframe:
            timeframe_instruction = f"- INCIDENT TIMEFRAME: The incident happened '{incident_timeframe}' - use this exact phrase, NOT 'yesterday' unless that's the actual timeframe.\n"

        opener_rules = {
            "role": "system",
            "content": (
                "CRITICAL CONSTRAINT: You MUST write EXACTLY 3 SHORT sentences in plain NZ English. NOT 4, NOT 5, EXACTLY 3. "
                "This is a SUMMARY OVERVIEW only - the officer will ask follow-up questions for details.\n\n"

                "Here are examples of well-written 3-sentence overviews:\n\n"

                "EXAMPLE 1 (Robbery):\n"
                "Facts: At 3.15pm three days ago, witnessed person in black mask run into dairy on High Street. Heard shouting. Person ran out carrying bag. Called 111.\n"
                "Good opener: 'At around 3.15pm three days ago, I witnessed a robbery at the dairy on High Street. I saw a person wearing a black mask run inside and heard shouting from within. The person ran back out carrying a bag and I immediately called 111.'\n\n"

                "EXAMPLE 2 (Fire incident):\n"
                "Facts: Driving home on Queen Street at 5pm. Saw teen run out of derelict house's front gate. Noticed smoke from front lawn. Stopped and put out fire. Looked for teen.\n"
                "Good opener: 'I was driving along Queen Street at around 5pm when I saw a teen run out of a derelict house's front gate. I noticed smoke coming from the front lawn and stopped to investigate. I put out the fire and then tried to find the teen.'\n\n"

                "EXAMPLE 3 (Assault):\n"
                "Facts: Walking on Beach Road at 9pm. Saw two people arguing near bus stop. One person punched the other. Victim fell. Attacker ran towards car park. Called 111.\n"
                "Good opener: 'At about 9pm, I was walking on Beach Road when I saw two people arguing near the bus stop. One person punched the other and the victim fell to the ground. The attacker ran towards the car park and I called 111.'\n\n"

                "KEY PRINCIPLES FROM EXAMPLES:\n"
                "- Sentence 1: Time/place + event type (robbery/fire/assault)\n"
                "- Sentence 2: What you saw the person do (1-2 key actions)\n"
                "- Sentence 3: Immediate outcome or what you did next\n"
                f"{timeframe_instruction}"
                "- Use natural phrasing: 'I saw X and Y' not 'X with Y visible'\n"
                "- Use possessives correctly: 'house's gate' not 'house gate'\n"
                "- Separate observations with 'and': 'ran out and I saw smoke' not 'ran out with smoke'\n"
                "- Don't fabricate: only include what's in SOURCE FACTS\n"
                "- Don't add movement details not stated: if facts say 'ran out', don't add 'down the street'\n"
                "- CRITICAL: Use EXACT WORDS from SOURCE FACTS - if facts say 'screaming', do NOT write 'shouting'. If facts say 'running', do NOT write 'jogging'. Copy the precise terminology.\n\n"

                "Now write YOUR 3-sentence overview based on SOURCE FACTS below, following the same style and structure as these examples. "
                "Use only information from SOURCE FACTS. Do not add venues, lanes, shops, vehicles, or people not in SOURCE FACTS. "
                "Do not ask questions, do not address the interviewer, do not include labels or metadata, and do not mention files or uploads. "
                "EXACTLY 3 sentences ending with periods."
            )
        }
        opener_user = {
            "role": "user",
            "content": (
                f"SOURCE FACTS (verbatim bullets from persona):\n{source_facts_text if source_facts_text else '- (no explicit bullets provided)'}\n\n"
                "TASK: Based only on SOURCE FACTS, write a 3-sentence overview of what I personally witnessed. "
                "Keep it factual and minimal. If a key detail is unknown in SOURCE FACTS, leave it out."
            )
        }

        def _call_llm(msgs):
            payload = {"model": OLLAMA_MODEL, "stream": True, "messages": msgs}
            r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=45, stream=True)
            if r.status_code != 200:
                return f"Error: LLM HTTP {r.status_code} — {r.text[:300]}"
            text = ""
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                content = chunk.get("message", {}).get("content", "")
                if content:
                    text += content
                if chunk.get("done", False):
                    break
            return (text or "").strip()

        def _has_invented_terms(resp: str, facts_text: str) -> bool:
            if not resp:
                return False
            low_resp = resp.lower()
            low_facts = (facts_text or "").lower()
            banned = ["shop", "store", "lane", "alley", "car that came up behind", "user uploaded a file"]
            for term in banned:
                if term in low_resp and term not in low_facts:
                    return True
            return False

        opener_text = _call_llm(base_msgs + [opener_rules, opener_user])
        if _has_invented_terms(opener_text, source_facts_text):
            opener_rules_fix = {
                "role": "system",
                "content": (
                    "Revise your last answer to remove any terms not present in SOURCE FACTS (e.g., shop, store, lane, alley, or upload artefacts). "
                    "Only keep details explicitly supported by SOURCE FACTS. Output exactly 3 sentences."
                )
            }
            opener_text = _call_llm(base_msgs + [opener_rules, opener_rules_fix, opener_user])

        # Enforce exactly 3 sentences; if fewer, ask for a compliant rewrite
        def _count_sentences(s: str) -> int:
            # More robust sentence counting - split on periods, exclamation, question marks
            # but not on abbreviations like "5.30pm" or "Mr."
            text = (s or "").strip()
            if not text:
                return 0
            # Replace common abbreviations to avoid false splits
            text = re.sub(r'\b(\d+)\.(\d+)(am|pm)\b', r'\1:\2\3', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof)\.', r'\1', text, flags=re.IGNORECASE)
            # Split on sentence-ending punctuation followed by space or end of string
            sentences = re.split(r'[.!?]+(?:\s+|$)', text)
            # Filter out empty strings
            return len([s for s in sentences if s.strip()])

        sentence_count = _count_sentences(opener_text)
        if sentence_count < 2:
            opener_rules_fix_len = {
                "role": "system",
                "content": (
                    "Your previous response only contained 1 sentence. You MUST rewrite it to be exactly 3 complete sentences. "
                    "Sentence 1: time/place + high-level event label (e.g., 'At 5:30pm on Johnson Street, I witnessed a robbery.'). "
                    "Sentence 2: what the person did in brief (e.g., 'A man ran into the store wearing a mask and holding a knife.'). "
                    "Sentence 3: the immediate outcome (e.g., 'He ran out and escaped in a grey vehicle.'). "
                    "Use only SOURCE FACTS; do not add details not present there. Each sentence must end with a period."
                )
            }
            opener_text = _call_llm(base_msgs + [opener_rules, opener_rules_fix_len, opener_user])
        elif sentence_count == 2:
            # Got 2 sentences, need to expand to 3
            opener_rules_fix_len = {
                "role": "system",
                "content": (
                    "Your previous response only contained 2 sentences. You MUST expand it to exactly 3 complete sentences. "
                    "Add more detail from SOURCE FACTS to create a third sentence about what happened next or the outcome. "
                    "Each sentence must end with a period."
                )
            }
            opener_text = _call_llm(base_msgs + [opener_rules, opener_rules_fix_len, opener_user])

        # Essential artifact cleanup (before self-review)
        opener_text = re.sub(r"(?im)^user uploaded a file:.*$", "", opener_text).strip()
        opener_text = re.sub(r"\s{2,}", " ", opener_text).strip()

        # LLM Self-Review: Iterative refinement loop
        # Keep refining until BOTH facts and grammar are correct
        # Maximum 2 iterations to prevent infinite loops while allowing correction

        fact_check_prompt = {
            "role": "system",
            "content": (
                "CRITICAL: Review your 3-sentence overview for FACTUAL ACCURACY against SOURCE FACTS.\n\n"
                "Check each sentence:\n"
                "1. CHRONOLOGICAL ORDER: Are events in the correct sequence as listed in SOURCE FACTS?\n"
                "   - Did you describe something happening BEFORE it actually occurred?\n"
                "   - Example error: 'He lit a fire before running back' when facts show he ran OUT and fire was already burning\n"
                "   - CRITICAL: Check temporal words 'after', 'before', 'then', 'while'\n"
                "   - Example error: 'drove away after hearing screaming' when screaming happened BEFORE driving away\n"
                "   - If using 'after X', verify X happened BEFORE the main action in SOURCE FACTS timeline\n\n"
                "2. CAUSAL CONNECTIONS: Did you invent cause-and-effect relationships NOT in SOURCE FACTS?\n"
                "   - Example error: 'after lighting the fire' when facts don't say witness saw them light it\n"
                "   - Only state what the witness DIRECTLY observed\n\n"
                "3. DIRECTION/MOVEMENT: Is the person's movement direction correct?\n"
                "   - 'ran out' vs 'ran back' vs 'ran towards' - these are VERY different\n"
                "   - 'ran away from' vs 'ran into' - check the direction matches facts\n\n"
                "4. WHO DID WHAT: Are actions attributed to the correct person?\n"
                "   - Did the WITNESS do it or did the OTHER PERSON do it?\n"
                "   - Example error: 'he put out the fire' when facts say 'I put out the fire'\n\n"
                "5. FABRICATED DETAILS: Did you add locations, objects, or actions NOT in SOURCE FACTS?\n"
                "   - Example error: 'ran down the street' when facts only say 'ran out of gate'\n"
                "   - Example error: 'before disappearing from view' when facts don't mention this\n\n"
                "If you find ANY factual errors, rewrite to fix them.\n"
                "If factually correct, return unchanged.\n"
                "Output ONLY the 3 sentences, nothing else."
            )
        }

        grammar_prompt = {
            "role": "system",
            "content": (
                "Now check for GRAMMAR and NATURAL PHRASING. The facts are correct, just improve the language.\n\n"
                "Fix these if present:\n"
                "- Missing possessives: 'building front gate' → 'building's front gate'\n"
                "- Awkward 'with' phrases: 'ran with smoke visible' → 'ran out and I saw smoke'\n"
                "- Awkward 'with' constructions: 'out of a house with smoke visible' → 'out of a house and I saw smoke'\n"
                "- Passive voice: 'was closed' → 'had closed'\n"
                "- Dangling modifiers: 'which was damaged' when unclear what 'which' refers to\n\n"
                "CRITICAL: Do NOT add new details or change facts while fixing grammar.\n"
                "Keep it natural and concise. Don't change the facts or sequence.\n"
                "Output ONLY the 3 sentences, nothing else."
            )
        }

        # Iterative refinement loop (max 2 iterations)
        for iteration in range(2):
            # Fact check pass
            fact_checked = _call_llm(base_msgs + [
                opener_rules,
                opener_user,
                {"role": "assistant", "content": opener_text},
                fact_check_prompt
            ])

            if fact_checked and len(fact_checked.strip()) > 20:
                opener_text = fact_checked.strip()

            # Grammar check pass
            grammar_checked = _call_llm(base_msgs + [
                opener_rules,
                opener_user,
                {"role": "assistant", "content": opener_text},
                grammar_prompt
            ])

            if grammar_checked and len(grammar_checked.strip()) > 20:
                opener_text = grammar_checked.strip()

        # Minimal essential guardrails (keep only for critical artifacts LLM consistently misses)
        # Remove "user uploaded" artifacts if they slip through
        opener_text = re.sub(r"(?im)^user uploaded a file:.*$", "", opener_text).strip()
        opener_text = re.sub(r"\buser uploaded\b", "", opener_text, flags=re.IGNORECASE)

        # Fix "emerged from" / "emerge from" → "get into" if facts say "emerged in"
        # The bare infinitive form ("take off his mask, and emerge from a car") is not caught
        # by the past-tense pattern, so it must be handled separately.
        source_facts_lower = source_facts_text.lower() if source_facts_text else ""
        if re.search(r"\bemerged?\s+in\s+(?:a|an|the)\s+", source_facts_lower):
            opener_text = re.sub(r"\bemerged\s+from\s+(?:a|an|the)\s+", "emerged in a ", opener_text, flags=re.IGNORECASE)
            opener_text = re.sub(r"\bemerge\s+from\s+(?:a|an|the)\s+", "get into a ", opener_text, flags=re.IGNORECASE)

        # Fix vehicle fabrications - extract actual vehicle from facts and fix common errors
        # Pattern: "emerged in a [color] [make/model]" from facts like "emerged in a light grey Mazda2"
        vehicle_match = re.search(
            r"\bemerged?\s+in\s+(?:a|an|the)\s+((?:light|dark|bright|pale|medium)?\s*(?:grey|gray|black|white|blue|red|green|silver|gold|yellow|brown|beige|tan)?\s*\w+(?:\d+)?(?:\s+\w+)?)",
            source_facts_lower
        )

        if vehicle_match:
            # Extract vehicle description and clean it up
            vehicle_desc = vehicle_match.group(1).strip()
            # Remove "with no licence plates" or similar phrases
            vehicle_desc = re.sub(r"\s+(?:with\s+)?(?:no|without)\s+licence.*$", "", vehicle_desc).strip()

            # Fix "emerged in a store" → "emerged in a [vehicle]"
            opener_text = re.sub(
                r"\bemerged\s+in\s+a\s+store\b",
                f"emerged in a {vehicle_desc}",
                opener_text,
                flags=re.IGNORECASE
            )

            # Fix "unknown vehicle" → actual vehicle
            opener_text = re.sub(
                r"\bunknown\s+vehicle\b",
                vehicle_desc,
                opener_text,
                flags=re.IGNORECASE
            )

            # Fix "vehicle that matched his path" or similar nonsense → actual vehicle
            opener_text = re.sub(
                r"\bvehicle\s+that\s+(?:matched|followed|traced)\s+(?:his|her|their)\s+path\b",
                vehicle_desc,
                opener_text,
                flags=re.IGNORECASE
            )

            # Fix fabricated uncertainty: "I couldn't see its colour or make" when facts DO contain this info
            # Pattern: "a car - I couldn't see its colour or make" → "a light grey Mazda2"
            # Also handles: "a car but I couldn't see", "a car, I didn't notice", etc.
            opener_text = re.sub(
                r"\ba\s+(?:car|vehicle)\s*[-–—,]\s*I\s+(?:couldn't|could\s+not|didn't|did\s+not)\s+(?:see|notice|make\s+out)\s+(?:its|the)\s+(?:colour|color|make|model|colour\s+or\s+make|make\s+or\s+colou?r)[^\.!?]*",
                f"a {vehicle_desc}",
                opener_text,
                flags=re.IGNORECASE
            )
            # Also handle "in a car (I couldn't see...)" variant
            opener_text = re.sub(
                r"\bin\s+(?:a|an)\s+(?:car|vehicle)\s*\([^)]*(?:couldn't|could\s+not|didn't)\s+(?:see|notice)[^)]*\)",
                f"in a {vehicle_desc}",
                opener_text,
                flags=re.IGNORECASE
            )

        # Synonym substitution for opener - replace LLM synonyms with exact wording from facts
        # This ensures factual precision in the opener (e.g., if facts say "screaming", don't allow "shouting")
        synonym_pairs = [
            ("screaming", ["shouting", "yelling", "crying out"]),
            ("shouting", ["yelling", "calling out"]),
            ("yelling", ["calling out"]),
            ("crying", ["weeping", "sobbing"]),
            ("running", ["jogging", "sprinting"]),
            ("walking", ["strolling"]),
        ]

        for fact_word, synonyms in synonym_pairs:
            # Check if the fact_word appears in source facts
            if re.search(rf"\b{fact_word}\b", source_facts_lower, re.IGNORECASE):
                # Replace any synonyms with the exact fact word
                for synonym in synonyms:
                    opener_text = re.sub(
                        rf"\b{synonym}\b",
                        fact_word,
                        opener_text,
                        flags=re.IGNORECASE
                    )

        # Fix chronological errors: "X after hearing Y" when Y happened BEFORE X
        # Pattern: "drove away... after hearing screaming" is wrong if screaming fact comes BEFORE driving fact
        # This catches cases where temporal connectors are used incorrectly
        if re.search(r"after\s+hearing\s+(screaming|shouting|glass|noise)", opener_text, re.IGNORECASE):
            # Remove the "after hearing X" phrase - it's in the wrong chronological position
            # The hearing event should have been mentioned earlier in the timeline
            opener_text = re.sub(
                r"\s+after\s+hearing\s+(?:screaming|shouting|glass\s+breaking|(?:a\s+)?noise|sounds?)(?:\s+and\s+(?:the\s+)?(?:sound\s+of\s+)?glass\s+breaking)?(?:\s+coming\s+from[^.]*)?",
                "",
                opener_text,
                flags=re.IGNORECASE
            )

        # Fix vague "in that direction" when facts specify the actual direction
        # The LLM sometimes substitutes a vague phrase for a clear directional fact ("towards me")
        if re.search(r"\btowards\s+me\b", source_facts_lower):
            opener_text = re.sub(r"\bin\s+that\s+direction\b", "towards me", opener_text, flags=re.IGNORECASE)

        # Remove "which caused [sounds]" — this incorrectly joins two separate facts (entering a
        # premises + hearing auditory observations) into a single causal clause.  The witness
        # observed the sounds independently; the opener should not invent the causal link.
        # Remove the clause so the two observations remain distinct.
        opener_text = re.sub(
            r",?\s+which\s+caused\s+(?:loud\s+)?(?:screaming|shouting|yelling|noise|glass\s+breaking)[^,\.!?]*",
            "",
            opener_text,
            flags=re.IGNORECASE
        )

        # Generic fabricated uncertainty removal
        # The opener should NEVER claim uncertainty about things that ARE in the facts
        # Common patterns: "I couldn't see", "I didn't notice", "I'm not sure", "unknown", "unclear"
        # Strategy: Remove entire clauses that express uncertainty, as the opener should only state known facts
        # If something isn't in facts, it should be omitted entirely, not mentioned with uncertainty
        uncertainty_patterns = [
            r"\s*[-–—,]\s*I\s+(?:couldn't|could\s+not|didn't|did\s+not|wasn't\s+able\s+to)\s+(?:see|notice|tell|make\s+out|identify|determine)[^\.!?]*",
            r"\s*[-–—,]\s*(?:but\s+)?I\s+(?:don't|do\s+not)\s+(?:know|remember)[^\.!?]*",
            r"\s*\([^)]*(?:couldn't|didn't|unknown|unclear|unsure|not\s+sure)[^)]*\)",
            r"\s*[-–—,]\s*(?:the\s+)?(?:colour|color|make|model|type)\s+(?:was\s+)?(?:unknown|unclear|not\s+visible)[^\.!?]*",
            r"\s*[-–—,]\s*(?:which|whose|that)\s+I\s+(?:couldn't|didn't)\s+(?:see|identify)[^\.!?]*",
        ]
        for pattern in uncertainty_patterns:
            opener_text = re.sub(pattern, "", opener_text, flags=re.IGNORECASE)

        # Clean up whitespace
        opener_text = re.sub(r"\s{2,}", " ", opener_text).strip()
        opener_text = re.sub(r"\s+\.", ".", opener_text)

        opener_text = opener_text.strip()
        if opener_text:
            return opener_text

    # ---------------- Default: delegate to LLM with stream, then guard output ----------------
    # Add FIRST_TURN_RULE or continuation rule depending on conversation state
    final_msgs = base_msgs.copy()
    if not any(m.get("role") == "assistant" for m in message_history):
        final_msgs.append(FIRST_TURN_RULE)
    else:
        final_msgs.append({"role":"system","content":"Continue following the above rules."})

    full_prompt = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "messages": final_msgs + message_history
    }

    try:
        http_resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=full_prompt, timeout=45, stream=True)
        if http_resp.status_code != 200:
            return f"Error: LLM HTTP {http_resp.status_code} — {http_resp.text[:300]}"
        full_content = ""
        for line in http_resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            content = chunk.get("message", {}).get("content", "")
            if content:
                full_content += content
            if chunk.get("done", False):
                break
        response = full_content.strip()
        if not response:
            return "I'm not sure."

        # Log raw LLM response BEFORE post-processing for debugging

        # Suppress unsolicited narrative unless an open-incident question OR a question explicitly asking for actions
        # Don't suppress if the question is asking "what did you do", "what happened next", etc. - those WANT narrative
        question_asks_for_narrative = bool(re.search(
            r"\b(what\s+(did\s+you\s+do|happened(\s+next)?)|then\s+what|after\s+that|describe\s+what|tell\s+me\s+about)\b",
            last_user_text.lower()
        ))

        # Don't suppress location answers — "where" questions require descriptions that naturally use
        # first-person past-tense language ("I was at the front bar", "I saw the argument near the pokies"),
        # which always trip the unsolicited narrative detector.
        # Match any question containing "where" regardless of word order (e.g. "Tell me where you were...")
        question_asks_for_location = bool(re.search(r"\bwhere\b", last_user_text.lower()))

        # If the LLM still returned an affirmative ("Yes", "I did", etc.) despite the unsolicited narrative,
        # do not replace the response — the narrative may have accompanied a valid positive answer.
        response_is_affirmative = bool(re.search(
            r"\b(yes|yeah|I\s+did|I\s+have|I\s+was|I\s+went|I\s+checked|I\s+called)\b",
            response, re.IGNORECASE
        ))

        if not _is_open_incident_question(last_user_text) and not question_asks_for_narrative and not question_asks_for_location and _contains_unsolicited_narrative(response):
            if _looks_like_question(last_user_text):
                # True yes/no questions start with an auxiliary verb and have no interrogative prefix word.
                is_yes_no = (
                    re.search(r"^\s*(is there|are there|was there|were there|did you|could you|would you|can you|will you|have you|had you)\b", last_user_text.lower()) and
                    not re.search(r"^\s*(what|where|when|how|why|which|who|whose)\b", last_user_text.lower())
                )
                if is_yes_no:
                    if not response_is_affirmative:
                        response = "No."
                else:
                    response = "I'm not sure."
            else:
                response = "I'm ready when you are."

        # Remove upload artefacts — Ollama occasionally echoes file-upload metadata from the conversation history.
        response = re.sub(r"(?im)^user uploaded a file:.*$", "", response).strip()
        response = re.sub(r"(?i)\b(i('m| am)|i've|i have)\s+(uploaded|attached)\b[^\.]*\.", "", response).strip()

        # Strip any self-generated questions — the witness answers, does not interrogate.
        # Persona assertions ("I understand you used to work there") are exempt so the witness can correct them.
        if "?" not in last_user_text and not _is_persona_assertion(last_user_text):
            parts = re.split(r'(?<=[\.!\?])\s+', response)
            filtered = [p for p in parts if not p.strip().endswith('?')]
            if filtered:
                response = " ".join(filtered).strip()
            if not response:
                response = "I'm ready when you are."

        # Strip action emotes — the LLM sometimes adds stage directions like *nervous smile* or *sighs*.
        response = re.sub(r"\*[^*]+\*", "", response).strip()
        response = re.sub(r"\s{2,}", " ", response).strip()

        # Role label stripping — the LLM sometimes prefixes responses with "Officer:" or "[Interviewer:]"
        # as if scripting a scene. All such labels are removed before the response is returned.
        response = re.sub(r"(?im)^\s*Officer\b\s*[:,]?\s*", "", response).strip()
        response = re.sub(r"(?im)^(?:interviewer|detective|officer)\s*:\s*.*\n?", "", response).strip()
        response = re.sub(r"\[\s*(?:Interviewer|Detective|Officer)\s*:\s*[^\]]+\]", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\[[^\]]*$", "", response)
        response = re.sub(r"(?im)^[A-Z][a-z]+:\s*", "", response).strip()
        response = re.sub(r"^(?:\.|\s*dot\s+)", "", response, flags=re.IGNORECASE).strip()

        # Remove meta-commentary where the LLM breaks character to comment on interview procedure.
        # Examples: "You didn't ask me that.", "We haven't discussed that yet."
        response = re.sub(r"\b[Yy]ou\s+(didn't|haven't|haven't\s+yet)\s+(ask|asked|mention|mentioned|bring\s+up|brought\s+up)\b[^\.!?]*[\.!?]", "", response)
        response = re.sub(r"\b[Tt]hat\s+wasn't\s+asked\b[^\.!?]*[\.!?]", "", response)
        response = re.sub(r"\b[Ww]e\s+haven't\s+(discussed|talked\s+about|covered)\s+that\s+yet\b[^\.!?]*[\.!?]", "", response)
        response = re.sub(r"\s{2,}", " ", response).strip()

        # Speak dates naturally
        date_pattern = r"(\d{4})[-/](\d{2})[-/](\d{2})(?=[\s\.,;!?]|$)"
        def format_spoken_date(match):
            year, month, day = match.group(1), match.group(2), match.group(3)
            day_int = int(day)
            suffix = "th" if 11 <= day_int <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day_int % 10, "th")
            month_name_str = month_name[int(month)]
            return f"{day_int}{suffix} {month_name_str} {year}"
        response = re.sub(date_pattern, format_spoken_date, response)

        # Tidy spacing and sentence caps
        response = re.sub(r"\s{2,}", " ", response).strip()
        response = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), response)

        # Physical description guardrails — apply after LLM response to prevent hallucinated features.
        # Shaved head overrides any hair colour the LLM may have generated.
        # Beard/goatee/stubble are removed entirely unless the facts explicitly mention them.
        facts_low_for_guard = ("\n".join((f.get("fact", "") for f in filtered_persona.get("facts_to_provide", []) if isinstance(f, dict)))).lower()
        if _facts_have_shaved_head(facts_low_for_guard):
            response = re.sub(r"\b(short|black|brown|blond|blonde|red|grey|gray|dark|light)\s+hair\b", "shaved head", response, flags=re.IGNORECASE)
            response = re.sub(r"\bshaved\s+head\b(\s+and\s+)?\bshaved\s+head\b", "shaved head", response, flags=re.IGNORECASE)
            response = re.sub(r"[^\.!?]*\b(his|her|their)\s+hair\s+(was|is|looked|appeared|seemed)\s+[^\.!?]*[\.!?]", "", response, flags=re.IGNORECASE)
            response = re.sub(r"[^\.!?]*\b(distinctive|visible|noticeable)\s+hair\b[^\.!?]*[\.!?]", "", response, flags=re.IGNORECASE)
            response = re.sub(r"\s{2,}", " ", response).strip()
        if not _facts_have_beard(facts_low_for_guard):
            response = re.sub(r"\b(beard|goatee|stubble)\b", "", response, flags=re.IGNORECASE)
            response = re.sub(r"\s{2,}", " ", response).strip()

        # If facts say "emerged in [vehicle]" (suspect appeared already seated), or if only "drove away"
        # is mentioned, remove any LLM-generated claims about the suspect exiting or entering a vehicle.
        if re.search(r"\bemerged\s+in\s+(a|an|the)\s+", facts_low_for_guard) or \
           (re.search(r"\bdrove\s+away\b", facts_low_for_guard) and not re.search(r"\b(got\s+(out|into)|exited|entered)\s+(of\s+)?(the\s+)?(car|vehicle)\b", facts_low_for_guard)):
            # Remove fabricated exit/entry phrases
            response = re.sub(r"\b(as|when)\s+(he|she|they)\s+(got\s+out|exited|stepped\s+out)\s+(of\s+)?(the\s+)?(car|vehicle)\b[^\.!?]*[,\.]?", "", response, flags=re.IGNORECASE)
            response = re.sub(r"\bI\s+saw\s+(it|him|her|them)\s+up\s+close\s+as\s+[^\.!?]*\b(got\s+out|exited)\b[^\.!?]*[\.!?]", "I saw the car as it drove away.", response, flags=re.IGNORECASE)
            response = re.sub(r"\s{2,}", " ", response).strip()

        # Darkness fabrication — the LLM invents lighting conditions to explain limited visibility
        # when none are mentioned in the facts. Remove these claims entirely.
        if not re.search(r"\b(dark|darkness|night|nighttime|dusk|dawn|evening|lighting|light\s+was)\b", facts_low_for_guard):
            response = re.sub(r"\bbecause\s+it\s+was\s+(dark|nighttime|evening)\b[^\.!?]*", "", response, flags=re.IGNORECASE)
            response = re.sub(r"\b(and|as)\s+it\s+was\s+(dark|nighttime|evening)\b[^\.!?]*", "", response, flags=re.IGNORECASE)
            response = re.sub(r"\s{2,}", " ", response).strip()
            # Clean up orphaned conjunctions
            response = re.sub(r"\s+and\s+\.", ".", response)
            response = re.sub(r"\.\s+[Bb]ut\s+I\s+heard", ". I heard", response)

        # Road-side correction — if facts say "other side of the road", the LLM occasionally misremembers it
        # as "same side", which directly contradicts the witness's stated position.
        if re.search(r"\bother\s+side\s+of\s+the\s+road\b", facts_low_for_guard):
            response = re.sub(r"\bsame\s+side\s+of\s+the\s+road\b", "other side of the road", response, flags=re.IGNORECASE)
            response = re.sub(r"\bon\s+my\s+side\s+of\s+the\s+road\b", "on the other side of the road", response, flags=re.IGNORECASE)

        # Face visibility justification correction — if the reason for not seeing the face is distance or
        # viewing angle (from behind, from the side), remove LLM-generated claims that clothing obscured it.
        for fact_item in filtered_persona.get("facts_to_provide", []):
            if isinstance(fact_item, dict):
                fact_text = fact_item.get("fact", "").lower()
                reason_text = fact_item.get("reason", "").lower()
                # Check if fact mentions not seeing face AND reason mentions distance/angle (not clothing)
                if re.search(r"\bcouldn'?t\s+(see\s+)?(his|her|their)\s+face\b", fact_text):
                    if re.search(r"\b(from\s+behind|from\s+the\s+side|from\s+a\s+distance|too\s+far|not\s+close\s+enough)\b", reason_text):
                        # Remove fabrications that clothing obscured the face
                        response = re.sub(r",?\s*(?:as|because|since)\s+(?:he|she|they)\s+(?:was|were)\s+wearing\s+(?:a\s+)?(?:black\s+)?(?:tracksuit|hoodie|jacket|hat|mask)[^\.!?]*", "", response, flags=re.IGNORECASE)
                        response = re.sub(r"\s{2,}", " ", response).strip()
                        # Clean up trailing commas before period
                        response = re.sub(r",\s*\.", ".", response)
                        break

        # Additional occupants in vehicle — remove if facts only mention one person and no "passenger" or "with" cues.
        vehicle_facts = re.findall(r"[^\.!?]*\b(car|vehicle|mazda|honda|toyota)\b[^\.!?]*[\.!?]", facts_low_for_guard)
        has_multiple_people = any(re.search(r"\b(with\s+(him|her|them|another|someone|a\s+passenger|a\s+driver))\b", vf) for vf in vehicle_facts)
        if not has_multiple_people:
            # Remove fabrications about additional people
            response = re.sub(r"\b[Tt]here\s+was\s+(another|a|an)\s+(person|man|woman)\s+(with\s+him|inside|in\s+the\s+car)\b[^\.!?]*[\.!?]", "", response)
            response = re.sub(r"\b[Hh]e\s+had\s+(a|an)\s+(passenger|driver|person)\s+with\s+him\b[^\.!?]*[\.!?]", "", response)
            response = re.sub(r"\s{2,}", " ", response).strip()

        # Remove invented corroboration — the LLM sometimes fabricates "another witness confirmed"
        # or "I spoke to them on the phone" when the facts only describe a single interaction.
        response = re.sub(r",?\s*(?:which\s+was\s+)?(?:confirmed|verified|corroborated)\s+by\s+(?:another|a\s+second)\s+witness\s+(?:later|afterwards)[^\.!?]*(?:when\s+I\s+spoke\s+to\s+them)?(?:\s+on\s+the\s+phone)?(?:\s+with\s+(?:NZ\s+)?[Pp]olice)?", "", response, flags=re.IGNORECASE)
        response = re.sub(r",?\s*(?:and|which)\s+(?:another|a\s+second)\s+(?:witness|person)\s+(?:later|also)\s+confirmed[^\.!?]*", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\s{2,}", " ", response).strip()
        # Clean up trailing commas before period
        response = re.sub(r",\s*\.", ".", response)

        # Indirect information correction — if the witness only knows a location because a bystander told them
        # ("stopped a pedestrian who said he ran down X"), remove LLM claims of direct observation of that location.
        if re.search(r"\b(stopped|asked|spoke\s+to)\s+(?:a\s+)?(?:pedestrian|witness|person|passer-?by)\s+who\s+said\b", facts_low_for_guard):
            # Extract the location from "who said [person] ran down [location]"
            indirect_location_match = re.search(r"who\s+said\s+(?:the\s+)?(?:teen|man|woman|person|he|she|they)\s+(?:ran|went|headed)\s+(?:down|into|along)\s+(?:an?\s+)?(?:alley|alleyway)\s+(?:into|to|onto)\s+([A-Z][A-Za-z\s]+(?:Street|Road|Lane|Avenue))", facts_low_for_guard, re.IGNORECASE)
            if indirect_location_match:
                indirect_location = indirect_location_match.group(1).strip()
                # Remove claims of seeing them run down that location
                response = re.sub(rf"\b(?:as|when)\s+(?:he|she|they)\s+ran\s+down\s+(?:the\s+|an?\s+)?(?:alley|alleyway)\s+(?:into|to|onto)\s+{re.escape(indirect_location)}\b", "", response, flags=re.IGNORECASE)
                response = re.sub(r"\s{2,}", " ", response).strip()

        # Fabricated carried items — only checked when the question is specifically about what was carried.
        # Removing broadly would risk stripping factual bag/weapon mentions from unrelated answers.
        if re.search(r"\b(carry|carrying|holding|hold|had|have)\s+(?:anything|something|items?)\b", last_user_text.lower()):
            # Check if facts mention any carried items
            has_carried_items = re.search(r"\b(holding|carrying|had|have|with)\s+(?:a|an|the)\s+(?:bag|backpack|purse|wallet|phone|weapon|knife|gun|tool|item|object)\b", facts_low_for_guard)
            if not has_carried_items:
                # Remove fabricated item claims
                response = re.sub(r"\b(?:The\s+)?(?:teen|man|woman|person|he|she|they)\s+had\s+(?:a|an)\s+(backpack|bag|purse|phone|wallet|weapon|knife|tool)\s+(?:with\s+(?:him|her|them))?\b[^\.!?]*[\.!?]?", "", response, flags=re.IGNORECASE)
                response = re.sub(r"\bwas\s+carrying\s+(?:a|an)\s+(backpack|bag|purse|phone|wallet|weapon|knife|tool)\b[^\.!?]*", "", response, flags=re.IGNORECASE)
                response = re.sub(r"\s{2,}", " ", response).strip()
                # If nothing left or just whitespace, return "I'm not sure"
                if not response or response.isspace():
                    response = "I'm not sure."

        # Artefact phrase removal — "the car that came up behind him" is an LLM artefact
        # that implies the witness saw the car approach from behind, which is rarely stated in facts.
        response = re.sub(r"\bcar that came up behind (him|her|them)\b", "a car", response, flags=re.IGNORECASE)
        response = re.sub(r"\b(which|that)\s+came up behind\s+(him|her|them)\b", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\s{2,}", " ", response).strip()

        # Location preposition correction — "ran past me on Lane X" implies the witness was standing on
        # the lane, when they were actually on the street watching someone run towards it.
        response = re.sub(r"\bran\s+past\s+me\s+on\s+([A-Z][A-Za-z''\-]+(?:'s)?\s+(?:Lane|Alley|Alleyway))\b", r"ran past me towards \1", response)

        # Running/vehicle conflation — "ran down X lane in a car" is grammatically incoherent.
        # The facts describe two sequential actions (ran → emerged in vehicle), not simultaneous ones.
        if re.search(r"\b(ran|running)\s+down\s+[^\.!?]+\s+in\s+(?:what\s+(?:appeared|looked|seemed)\s+to\s+be\s+)?(?:a|an|the)\s+(?:light\s+|dark\s+)?(?:grey|gray|white|black|red|blue|green|silver)?\s*(?:car|vehicle|mazda|honda|toyota|ford|holden|nissan|subaru)", response, re.IGNORECASE):
            # Rewrite to separate the actions properly
            response = re.sub(
                r"\b((?:He|She|They)\s+(?:was|were)\s+(?:subsequently\s+)?(?:seen\s+)?)(running|ran)\s+down\s+([A-Z][A-Za-z''\-]+(?:'s)?\s+(?:[Ll]ane|[Aa]lley(?:way)?|[Ss]treet|[Rr]oad))\s+in\s+(what\s+(?:appeared|looked|seemed)\s+to\s+be\s+)?(a|an|the)\s+((?:light\s+|dark\s+)?(?:grey|gray|white|black|red|blue|green|silver)?\s*[A-Za-z0-9]+)(\s+(?:with(?:out)?\s+(?:no\s+)?(?:licence\s+)?plates?)?)?(\s+that\s+drove\s+away[^\.!?]*)?\b",
                r"\1ran down \3, then emerged in \5 \6\7\8",
                response,
                flags=re.IGNORECASE
            )
            # Also fix simpler patterns without "was seen"
            response = re.sub(
                r"\b(running|ran)\s+down\s+([A-Z][A-Za-z''\-]+(?:'s)?\s+(?:[Ll]ane|[Aa]lley(?:way)?|[Ss]treet|[Rr]oad))\s+in\s+(what\s+(?:appeared|looked|seemed)\s+to\s+be\s+)?(a|an|the)\s+((?:light\s+|dark\s+)?(?:grey|gray|white|black|red|blue|green|silver)?\s*[A-Za-z0-9]+)",
                r"ran down \2, then got into \4 \5",
                response,
                flags=re.IGNORECASE
            )
            response = re.sub(r"\s{2,}", " ", response).strip()

        # Temporal logic correction — if the witness left a venue, they cannot have seen events there "later".
        # This removes LLM-generated continuations that extend the witness's observations past their departure.
        if re.search(r"\b(?:at|in)\s+(?:the\s+)?(?:pub|bar|shop|store|building)\s+when\s+I\s+left\b.*\b(?:and\s+)?later\s+I\s+saw\b", response, re.IGNORECASE):
            # Remove the "later I saw" part - witness can't see events after leaving
            response = re.sub(
                r",?\s*(?:and\s+)?later\s+I\s+saw\s+(?:him|her|them|the\s+\w+)\s+being\s+helped.*?(?:\.|$)",
                ".",
                response,
                flags=re.IGNORECASE
            )
            # Clean up any doubled periods or awkward punctuation
            response = re.sub(r'\.\s*\.', '.', response)
            response = re.sub(r',\s*\.', '.', response)

        # "Emerged in" / "got into" de-duplication — if facts say the suspect emerged already in a vehicle,
        # remove any redundant "got into the car" sentence the LLM may have appended.
        resp_low_facts = facts_low_for_guard
        if re.search(r"\bemerged?\s+(?:from\s+\w+\s+)?in\s+(?:a|an|the)\s+", resp_low_facts):
            # Fix: "emerge(d) from a car" should be "emerge(d) in a car" when facts say "emerged in"
            # Handle both present tense "emerge from" and past tense "emerged from"
            response = re.sub(r"\bemerges?\s+from\s+(?:a|an|the)\s+", lambda m: m.group(0).replace(" from ", " in "), response, flags=re.IGNORECASE)
            response = re.sub(r"\bemerged\s+from\s+(?:a|an|the)\s+", "emerged in a ", response, flags=re.IGNORECASE)
            # Remove "got into the car" if we already said "emerged in" the car
            response = re.sub(r"\.\s+He\s+got\s+into\s+the\s+car\s+and\s+", ". He ", response, flags=re.IGNORECASE)
            response = re.sub(r"\.\s+He\s+got\s+into\s+(?:a|an|the)\s+(?:\w+\s+)*(?:car|vehicle)\s*\.", ".", response, flags=re.IGNORECASE)

        # Synonym substitution — enforce the exact wording from the facts to prevent the LLM from
        # paraphrasing key observations (e.g., "screaming" in facts must not become "shouting" in the response).
        synonym_pairs = [
            ("screaming", ["shouting", "yelling", "crying out"]),
            ("shouting", ["yelling", "calling out"]),
            ("yelling", ["calling out"]),
            ("crying", ["weeping", "sobbing"]),
            ("running", ["jogging", "sprinting"]),
            ("walking", ["strolling"]),
        ]

        for fact_word, synonyms in synonym_pairs:
            if re.search(rf"\b{fact_word}\b", resp_low_facts, re.IGNORECASE):
                for synonym in synonyms:
                    response = re.sub(rf"\b{synonym}\b", fact_word, response, flags=re.IGNORECASE)

        # Action attribution correction — if facts say the witness heard/did something, correct the LLM
        # when it incorrectly attributes that action to "he" or "she" (the suspect) instead.
        if re.search(r"\bheard\s+(screaming|shouting|glass\s+breaking|a\s+noise|sounds?)\b", resp_low_facts):
            response = re.sub(r"\b(he|she|they|the\s+(?:man|boy|teen|person|offender|suspect))\s+heard\s+(screaming|shouting|glass\s+breaking|a\s+noise|sounds?)\b",
                              r"I heard \2", response, flags=re.IGNORECASE)
        if re.search(r"\bput\s+(it|the\s+fire)\s+out\b", resp_low_facts):
            response = re.sub(r"\b(he|she|they|the\s+(?:man|boy|teen|person|offender|suspect))\s+put\s+(?:the\s+)?fire\s+out\b",
                              "I put the fire out", response, flags=re.IGNORECASE)
            response = re.sub(r"\b(he|she|they)\s+put\s+it\s+out\b", "I put it out", response, flags=re.IGNORECASE)
        if re.search(r"\bcalled?\s+(111|105)\b|\bcall\s+(111|105)\b", resp_low_facts):
            response = re.sub(r"\b(he|she|they|the\s+(?:man|boy|teen|person|offender|suspect))\s+called\s+(111|105)\b",
                              r"I called \2", response, flags=re.IGNORECASE)
            response = re.sub(r"\b(he|she|they)\s+(dialed|phoned|rang)\s+(111|105)\b",
                              r"I \2 \3", response, flags=re.IGNORECASE)

        # Remove timeline bridge sentences the LLM inserts to connect unrelated events chronologically.
        # Example: "The screaming had just started when they came out" inverts the actual sequence.
        response = re.sub(r"\.\s+The\s+(screaming|shouting|noise|glass\s+breaking|sounds?|fire|smoke).*?(?:had\s+)?(?:probably\s+)?(?:just\s+)?(?:started|begun|happened|occurred).*?when\s+(?:they|he|she|the\s+\w+)\s+(?:came|went|ran|left|exited|emerged).*?\.", ".", response, flags=re.IGNORECASE)

        response = re.sub(r"\s{2,}", " ", response).strip()

        # If the question was preceded by "thank you", acknowledge it before the answer.
        if prepend_thanks_response:
            response = "You're welcome. " + response

        # Log the LLM response with the original question

        return response
    except Exception as e:
        return f"Error: {e}"