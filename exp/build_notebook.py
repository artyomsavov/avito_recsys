"""exp/final_src.py (ячейки через `# %%`) -> final.ipynb; печатает число строк кода."""

import json
import re

src = open("exp/final_src.py").read()
cells, code_lines = [], 0
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    head, _, body = chunk.partition("\n")
    body = body.strip("\n")
    if head.strip() == "[markdown]":
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        continue
    if head.strip():
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"### {head.strip()}"]})
    lines = body.splitlines()
    code_lines += sum(1 for line in lines if line.strip())
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]})
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("final.ipynb", "w"), ensure_ascii=False, indent=1)
print(f"final.ipynb: {len(cells)} ячеек, {code_lines} непустых строк кода (вместе с комментариями, без markdown)")
