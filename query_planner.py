import re

def parse_search_query(text):
    pattern = r'<search>(.*?)</search>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    content = match.group(1).strip()
    
    mode_match = re.search(r'mode\s*=\s*(local|global)', content)
    hop_match = re.search(r'hop\s*=\s*(\d+)', content)
    query_match = re.search(r'query\s*=\s*["\']?(.*?)["\']?\s*$', content, re.DOTALL)
    
    if mode_match:
        return {
            "mode": mode_match.group(1),
            "hop": int(hop_match.group(1)) if hop_match else 1,
            "query": query_match.group(1).strip() if query_match else content
        }
    else:
        return {"mode": "local", "hop": 1, "query": content}

def parse_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None