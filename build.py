# aice_core.py 내용을 aice selfstudy.html의 <script id="py-core"> 블록에 붙여 넣는다.
import re
from pathlib import Path

here = Path(__file__).parent
html = (here / "aice selfstudy.html").read_text(encoding="utf-8")
core = (here / "aice_core.py").read_text(encoding="utf-8")
html = re.sub(r'(<script type="text/plain" id="py-core">\n).*?(</script>)',
              lambda m: m.group(1) + core + m.group(2), html, count=1, flags=re.S)
(here / "aice selfstudy.html").write_text(html, encoding="utf-8")
print("aice selfstudy.html 갱신 완료")
