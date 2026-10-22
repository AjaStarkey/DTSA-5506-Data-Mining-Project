"""Write requirements.lock.txt (exact versions of everything requirements.txt pulls in)
and metrics/environment.json (versions behind the reported results)."""
import json, platform, re, sys
from importlib import metadata
from pathlib import Path

def norm(n): return re.sub(r"[-_.]+", "-", n).lower()
dists = {norm(d.metadata["Name"]): d for d in metadata.distributions() if d.metadata["Name"]}

roots = []
for line in Path("requirements.txt").read_text().splitlines():
    line = line.split("#")[0].strip()
    if line:
        roots.append(norm(re.split(r"[<>=!~\[; ]", line)[0]))

seen, stack, missing = set(), list(roots), []
while stack:
    n = stack.pop()
    if n in seen: continue
    if n not in dists:
        if n in roots: missing.append(n)
        continue
    seen.add(n)
    for r in dists[n].requires or []:
        if "extra ==" in r: continue          # skip optional extras
        if ";" in r:                          # respect platform/python markers
            try:
                from packaging.requirements import Requirement
                if not Requirement(r).marker.evaluate(): continue
            except Exception: pass
        stack.append(norm(re.split(r"[<>=!~\[; (]", r.strip())[0]))
if missing: sys.exit("Not installed (run pip install -r requirements.txt first): " + ", ".join(missing))

lines = sorted(f'{dists[n].metadata["Name"]}=={dists[n].version}' for n in seen)
py = platform.python_version()
Path("requirements.lock.txt").write_text(
    f"# Exact package versions used to produce the results in metrics/ and the manuscript.\n"
    f"# Python {py}. Install with:  pip install -r requirements.lock.txt\n" + "\n".join(lines) + "\n")
Path(".python-version").write_text(py + "\n")
key = ["scikit-learn", "numpy", "pandas", "scipy", "mlflow", "dvc", "dagster", "matplotlib"]
env = {"python": py, "platform": platform.platform(),
       "packages": {k: dists[k].version for k in key if k in dists},
       "locked_packages": len(lines)}
Path("metrics").mkdir(exist_ok=True)
Path("metrics/environment.json").write_text(json.dumps(env, indent=2) + "\n")
print(f"Locked {len(lines)} packages for Python {py}")
print("  " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()))
