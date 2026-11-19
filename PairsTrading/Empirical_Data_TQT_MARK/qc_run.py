"""
Run the pairs-trading backtest on QuantConnect from your own computer, using the REST API.

WHAT IT DOES
  1. Uploads main.py and factor.py (from the same folder as this script) to a QuantConnect project
  2. Compiles the project
  3. Starts a backtest and waits for it to finish
  4. Prints the statistics and the REFIT / SUMMARY lines from the logs

SETUP (Mac Terminal; the values are NOT stored in this file, so they can't leak into GitHub)
  pip3 install requests
  export QC_USER_ID="your user id"
  export QC_API_TOKEN="your api token"
  python3 qc_run.py                         # creates a NEW project and runs it there (safe)

  Optional: to upload into an existing project instead (this OVERWRITES its main.py and
  factor.py, so it asks you to type yes first):
  export QC_PROJECT_ID="the project id"

The exports only last for that Terminal window. Never paste the token into chat, a file you
commit, or a screenshot. If it leaks, regenerate it in your QuantConnect account settings.
"""
import os
import re
import sys
import time
from base64 import b64encode
from hashlib import sha256
from pathlib import Path

import requests

BASE_URL = "https://www.quantconnect.com/api/v2"
FOLDER = Path(__file__).resolve().parent          # folder holding main.py and factor.py
FILES = ["main.py", "factor.py"]                   # files uploaded to the project

# Project parameters for this run (same names as the Parameters panel in the web IDE).
# If the API rejects the "parameters" field, set PARAMS = {} and use the web IDE panel
# (or edit DEFAULTS in the code) instead.
PARAMS = {
    "START_DATE": "2022-01-01",
    "END_DATE": "2024-12-31",
    "SIGNAL_MODE": "g",          # "g", "ag" or "factor"
    "FUND_METRICS": "operation_margin",
}
LOG_QUERIES = ["REFIT", "SUMMARY", "FACTOR"]      # log lines containing these words are printed
POLL_SECONDS = 10


def need(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing environment variable {name}. See SETUP at the top of this file.")
    return value


USER_ID = need("QC_USER_ID")
API_TOKEN = need("QC_API_TOKEN")
# Optional. If set, the files are uploaded INTO that existing project (overwriting main.py and
# factor.py there, after a confirmation). If not set, a brand-new project is created instead,
# so nobody else's files are touched.
PROJECT_ID = int(os.environ["QC_PROJECT_ID"]) if os.environ.get("QC_PROJECT_ID") else None


def headers():
    """QuantConnect wants a fresh timestamp on every call, and the token hashed with it."""
    timestamp = str(int(time.time()))
    hashed = sha256(f"{API_TOKEN}:{timestamp}".encode("utf-8")).hexdigest()
    auth = b64encode(f"{USER_ID}:{hashed}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {auth}", "Timestamp": timestamp}


def call(path, payload=None, allow_failure=False):
    """POST to the API and return the parsed JSON. Stops with a readable message on errors."""
    response = requests.post(f"{BASE_URL}/{path}", headers=headers(), json=payload or {}, timeout=120)
    try:
        data = response.json()
    except ValueError:
        sys.exit(f"{path}: HTTP {response.status_code}, not JSON: {response.text[:300]}")
    if response.status_code == 401:
        sys.exit(f"{path}: 401 unauthorized. Check the user id and token (or the clock on your computer).")
    if not data.get("success", False) and not allow_failure:
        sys.exit(f"{path} failed: {data.get('errors') or data}")
    return data


def download(path, payload, filename):
    """For endpoints that return a file (text or CSV) instead of JSON. Saves it next to the script."""
    response = requests.post(f"{BASE_URL}/{path}", headers=headers(), json=payload, timeout=300)
    if response.status_code != 200:
        print(f"{path}: HTTP {response.status_code} {response.text[:200]}")
        return None
    target = FOLDER / filename
    target.write_text(response.text, encoding="utf-8")
    return target


def ensure_project():
    """Use QC_PROJECT_ID if given (after a confirmation), otherwise create a new project."""
    if PROJECT_ID:
        print(f"This will OVERWRITE {', '.join(FILES)} in the EXISTING project {PROJECT_ID}.")
        if input("Type yes to continue: ").strip().lower() != "yes":
            sys.exit("Cancelled. Nothing was changed.")
        return PROJECT_ID
    name = f"fundamentals-factor-{int(time.time())}"
    created = call("projects/create", {"name": name, "language": "Py"})
    project_id = created["projects"][0]["projectId"]
    print(f"created new project '{name}' with id {project_id}")
    return project_id


_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_PATCHED = set()


def apply_overrides(name, text):
    """Write the PARAMS values into the DEFAULTS of the uploaded copy of the code (your local
    files are never changed). This way a run does not depend on QuantConnect honoring the
    'parameters' field of the API: the setting is simply inside the code that runs.
    Replaces the first '"KEY": value' found for each key."""
    for key, value in PARAMS.items():
        text_value = str(value)
        if _NUMBER.match(text_value) or text_value in ("True", "False"):
            literal = text_value
        else:
            literal = '"' + text_value.replace('"', '\\"') + '"'
        pattern = re.compile(r'("' + re.escape(key) + r'"\s*:\s*)("(?:[^"\\]|\\.)*"|-?\d[\d_]*\.?\d*|True|False)')
        text, count = pattern.subn(lambda m: m.group(1) + literal, text, count=1)
        if count:
            _PATCHED.add(key)
            print(f"  set {key} = {literal} in {name}")
    return text


def upload(name):
    content = apply_overrides(name, (FOLDER / name).read_text(encoding="utf-8"))
    payload = {"projectId": PROJECT_ID, "name": name, "content": content}
    result = call("files/update", payload, allow_failure=True)     # file already exists
    if not result.get("success"):
        call("files/create", payload)                               # first upload of this file
    print(f"uploaded {name} ({len(content)} characters)")


def compile_project():
    compile_id = call("compile/create", {"projectId": PROJECT_ID})["compileId"]
    while True:
        result = call("compile/read", {"projectId": PROJECT_ID, "compileId": compile_id})
        state = result.get("state")
        if state == "BuildSuccess":
            print("compiled")
            return compile_id
        if state == "BuildError":
            print("BUILD ERROR:")
            for line in result.get("logs", []):
                print("  ", line)
            sys.exit(1)
        time.sleep(3)


def run_backtest(compile_id):
    payload = {"projectId": PROJECT_ID, "compileId": compile_id,
               "backtestName": f"fundamentals-{int(time.time())}"}
    if PARAMS:
        payload["parameters"] = PARAMS
    created = call("backtests/create", payload)
    backtest = created["backtest"]
    backtest = backtest[0] if isinstance(backtest, list) else backtest
    backtest_id = backtest["backtestId"]
    print(f"backtest started: {backtest_id}")
    while True:
        time.sleep(POLL_SECONDS)
        backtest = call("backtests/read", {"projectId": PROJECT_ID, "backtestId": backtest_id})["backtest"]
        if backtest.get("error"):
            print("BACKTEST ERROR:", backtest["error"], backtest.get("stacktrace", ""))
            break
        progress = backtest.get("progress")
        print(f"  progress: {progress}")
        if backtest.get("completed"):
            break
    return backtest_id, backtest


def main():
    global PROJECT_ID
    call("authenticate")
    print("authenticated")
    print("settings for this run:", PARAMS)
    PROJECT_ID = ensure_project()
    for name in FILES:
        upload(name)
    missing = [key for key in PARAMS if key not in _PATCHED]
    if missing:
        print(f"WARNING: these PARAMS were not found in {FILES} and will not take effect: {missing}")
    compile_id = compile_project()
    backtest_id, backtest = run_backtest(compile_id)

    print("\nSTATISTICS")
    for key, value in (backtest.get("statistics") or {}).items():
        print(f"  {key}: {value}")

    print("\nLOG LINES")
    log_file = download("backtests/export/logs",
                        {"projectId": PROJECT_ID, "backtestId": backtest_id},
                        f"logs-{backtest_id}.txt")
    if log_file:
        lines = log_file.read_text(encoding="utf-8").splitlines()
        print(f"({len(lines)} log lines saved to {log_file.name})")
        for line in lines:
            if any(q.lower() in line.lower() for q in LOG_QUERIES):
                print("  ", line)
        for line in lines:
            if "SUMMARY signal_mode=" in line:
                mode = line.split("SUMMARY signal_mode=")[1].split()[0]
                wanted = str(PARAMS.get("SIGNAL_MODE", mode))
                if mode == wanted:
                    print(f"\nCHECK OK: the run used signal_mode={mode}, as requested.")
                else:
                    print(f"\nCHECK FAILED: you asked for signal_mode={wanted} but the run used {mode}.")
                break
    trades_file = download("backtests/export/trades", {"backtestId": backtest_id},
                           f"trades-{backtest_id}.csv")
    if trades_file:
        print(f"\nclosed trades saved to {trades_file.name}")


if __name__ == "__main__":
    main()