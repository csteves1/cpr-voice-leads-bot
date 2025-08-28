import re

def is_hours(s): return re.search(r"\b(what(\s+are)?\s+(your|the)\s+)?hours?\b", s)
def is_location(s): 
    return re.search(r"\b(what(\s+is)?\s+(your|the)\s+)?(address|location)\b", s) or \
           re.search(r"where\s+(are\s+(you|y['’]all|ya['’]ll|yall)|is\s+the\s+store)", s)
def is_phone(s): return re.search(r"\b(what(\s+is)?\s+(your|the)\s+)?(phone(\s+number)?|number)\b", s)
def is_landmarks(s): return re.search(r"(nearby|landmark|close\s+to|around\s+(you|there)|what'?s\s+(near|around)\s+(you|there))", s)
def is_directions(s):
    return re.search(r"\bdirections?\b", s) or \
           re.search(r"\b(i\s+need|looking\s+for|get|send|give|can\s+you\s+send\s+me)\s+directions?\b", s) or \
           re.search(r"how\s+do\s+i\s+get\s+(there|to\s+you|to\s+the\s+store|to\s+your\s+location)", s) or \
           re.search(r"how\s+to\s+get\s+(there|to\s+you|to\s+the\s+store)", s) or \
           re.search(r"where\s+(are\s+(you|y['’]all|ya['’]ll|yall)|ya['’]ll|yall)\s+at", s)