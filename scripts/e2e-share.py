#!/usr/bin/env python3
"""End-to-end scenarios for shared best training, against the REAL dino db.

Each scenario asserts an observable outcome, and the negative cases are run
first so a guard that never fires cannot be mistaken for a passing guard.
"""
import json, re, urllib.request, urllib.error, sys

cfg = json.load(open('/var/minis/workspace/dino-redis.json'))
URL = cfg['UPSTASH_DINO_REST_URL']
RW = cfg['UPSTASH_DINO_REST_TOKEN']
RO = cfg['UPSTASH_DINO_READONLY_TOKEN']
KEY = "dino:best:v1"

# Extract the CAS script from the shipped module so the test cannot drift
# from the code under test.
src = open('/var/minis/workspace/dino-viewer/share.js').read()
m = re.search(r'const CAS_LUA =\s*(.*?);\n\nexport async function pushIfBetter', src, re.S)
parts = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
CAS = "".join(p.encode().decode('unicode_escape') for p in parts)
SCHEMA, MIN_RUNS = "1", "1"

def call(args, tok=RW):
    req = urllib.request.Request(URL, data=json.dumps(args).encode(), method="POST",
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=20))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP{e.code}", "body": e.read().decode()[:120]}

def publish(score, runs=2, tok=RW, schema=1):
    payload = {
        "v": schema, "score": score, "best": score + 300, "runs": runs,
        "edition": f"e{score}", "at": 1700000000000,
        "evolve": {"champion": {
            "g": {"loA": 25, "loB": -1.2, "widthA": 31, "widthB": -1.4, "duck": 39},
            "mean": score, "best": score + 300, "runs": runs, "edition": f"e{score}"}},
        "mlp": {"dim": 384}, "knn": {"xs": []},
    }
    return call(["EVAL", CAS, "1", KEY, json.dumps(payload), str(score), SCHEMA, str(MIN_RUNS)], tok)

results = []
def check(name, got, want):
    ok = got == want
    results.append((ok, name, got, want))
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   got={got} want={want}" if not ok else ""))

print("scenario 1: empty key -> first publish wins")
call(["DEL", KEY])
check("publish into empty", publish(1000).get("result"), 1)
cur = json.loads(call(["GET", KEY])["result"])
check("stored score is 1000", cur["score"], 1000)

print("scenario 2: weaker publish is REFUSED")
check("weaker refused", publish(500).get("result"), 0)
check("value unchanged", json.loads(call(["GET", KEY])["result"])["score"], 1000)

print("scenario 3: stronger publish REPLACES")
check("stronger accepted", publish(2000).get("result"), 1)
check("value updated", json.loads(call(["GET", KEY])["result"])["score"], 2000)

print("scenario 4: equal score is refused (no pointless churn)")
check("equal refused", publish(2000).get("result"), 0)

print("scenario 5: CORRUPT value is healed, not blocking")
call(["SET", KEY, "{this is not json"])
check("corrupt -> healed(2)", publish(100).get("result"), 2)
check("healed with client data", json.loads(call(["GET", KEY])["result"])["score"], 100)

print("scenario 6: value of WRONG SCHEMA is treated as corrupt")
call(["SET", KEY, json.dumps({"v": 99, "score": 999999, "runs": 9,
      "evolve": {"champion": {"g": {}}}})])
check("wrong schema -> healed(2)", publish(150).get("result"), 2)
check("replaced despite huge score", json.loads(call(["GET", KEY])["result"])["score"], 150)

print("scenario 7: structurally wrong value (no genome) is healed")
call(["SET", KEY, json.dumps({"v": 1, "score": 999999, "runs": 9, "evolve": {}})])
check("no genome -> healed(2)", publish(160).get("result"), 2)

print("scenario 8: an UNMEASURED incumbent is not allowed to block")
# MIN_RUNS is 1, so runs:1 is now a legitimate entry. runs:0 is the
# unmeasured case that must still be treated as corrupt.
call(["SET", KEY, json.dumps({"v": 1, "score": 999999, "runs": 0,
      "evolve": {"champion": {"g": {"loA": 1}}}})])
check("runs<MIN -> healed(2)", publish(170).get("result"), 2)

print("scenario 9: read-only token CANNOT publish")
r = publish(5000, tok=RO)
check("RO blocked", r.get("error"), "HTTP403")
check("value untouched by RO", json.loads(call(["GET", KEY])["result"])["score"], 170)

print("scenario 10: read path works with the READ-ONLY token")
check("RO can GET", json.loads(call(["GET", KEY], tok=RO)["result"])["score"], 170)

print("scenario 11: realistic payload size is accepted")
big = {"v": 1, "score": 3000, "best": 3200, "runs": 6, "edition": "big", "at": 1,
       "evolve": {"champion": {"g": {"loA": 25, "loB": -1.2, "widthA": 31,
                  "widthB": -1.4, "duck": 39}, "mean": 3000, "runs": 6}},
       "mlp": {"dim": 384, "w": [0.1] * 2000}, "knn": {"xs": [[0.5] * 8] * 400}}
blob = json.dumps(big)
r = call(["EVAL", CAS, "1", KEY, blob, "3000", SCHEMA, str(MIN_RUNS)])
check(f"{len(blob)//1024}KB payload accepted", r.get("result"), 1)

call(["DEL", KEY])
bad = [r for r in results if not r[0]]
print(f"\n{len(results)-len(bad)}/{len(results)} scenarios passed")
sys.exit(1 if bad else 0)
