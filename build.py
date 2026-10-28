# aice_core.py 내용을 index.html의 <script id="py-core"> 블록에 붙여 넣는다.
import re
from pathlib import Path

here = Path(__file__).parent
html = (here / "index.html").read_text(encoding="utf-8")
core = (here / "aice_core.py").read_text(encoding="utf-8")
html = re.sub(r'(<script type="text/plain" id="py-core">\n).*?(</script>)',
              lambda m: m.group(1) + core + m.group(2), html, count=1, flags=re.S)
(here / "index.html").write_text(html, encoding="utf-8")
print("index.html 갱신 완료")
