import re

def extract_case_parts(text):
    text = str(text).strip()
    split_patterns = [
        r'(?i)(judgment|held|decision|ruling|court.?held|it.?held)[:.]',
        r'(?i)(facts|background|case.*?facts)[:.]',
        r'(?i)(analysis|reasoning|legal.*?analysis)[:.]',
        r'(?i)(conclusion|outcome|result)[:.]'
    ]
    for pattern in split_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            split_point = matches[0].start()
            facts, judgment = text[:split_point].strip(), text[split_point:].strip()
            if len(facts) > 50 and len(judgment) > 50:
                return facts, judgment
    split_point = len(text) * 3 // 5
    return text[:split_point].strip(), text[split_point:].strip()
