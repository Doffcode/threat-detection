"""Parse Firecrawl-scraped Roboflow Universe search pages (JSON with 'markdown') into project cards."""
import json, re, sys
WEAP = re.compile(r'knife|gun|pistol|rifle|sword|sickle|machete|blade|weapon|axe|aruval|katta|dagger|cleaver|celurit|parang|golok', re.I)
cards = {}
for f in sys.argv[1:]:
    try:
        m = json.load(open(f))['markdown']
    except Exception:
        continue
    parts = re.split(r'\n\[\*\*', m)
    for p in parts[1:]:
        t = re.match(r'(.*?)\*\*\]\((https://universe\.roboflow\.com/([^/]+)/([^/ )]+))', p)
        if not t:
            continue
        title, url, ws, proj = t.groups()
        n = re.search(r'\n([\d.,]+k?) images', p)
        cls = re.findall(r'\[([^\]]+)\]\(https://universe\.roboflow\.com/search\?q=class:', p.split('\n[![')[0])
        cards[url] = (title, ws, proj, n.group(1) if n else '?', cls)
for url, (title, ws, proj, n, cls) in cards.items():
    w = [c for c in cls if WEAP.search(c)]
    if w:
        print(f"{n:>6} | {title[:40]:40} | {url} | {', '.join(cls[:14])}")
