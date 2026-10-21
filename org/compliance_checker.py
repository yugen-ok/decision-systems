import json

log("Loading OCEL file...")
with open(PATH, encoding="utf8") as f:
    ocel = json.load(f)


from datetime import datetime
from collections import defaultdict

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def parse_time(ts):
    return datetime.fromisoformat(ts)

def log(msg):
    print(msg)

# --------------------------------------------------
# Access events (assumed loaded)
# --------------------------------------------------

events = ocel["ocel:events"]

log(f"Using {len(events)} events already loaded")

# --------------------------------------------------
# Group events by purchase item (cID)
# --------------------------------------------------

events_by_item = defaultdict(list)

for e in events.values():
    vmap = e.get("ocel:vmap", {})
    cid = vmap.get("cID")
    if cid is not None:
        events_by_item[cid].append(e)

log(f"Grouped into {len(events_by_item)} purchase items")

# --------------------------------------------------
# Revised compliance checker
# --------------------------------------------------

def check_item(cID, events):
    """
    Revised rules:
    - Invoice before GR is allowed
    - Invoice must NOT be cleared before GR (for 3-way items)
    """

    # Sort by timestamp (still diagnostic)
    events = sorted(events, key=lambda e: parse_time(e["ocel:timestamp"]))

    log("\n" + "=" * 80)
    log(f"CHECKING ITEM {cID}")
    log(f"Number of events: {len(events)}")

    # Static attributes
    v0 = events[0]["ocel:vmap"]
    item_cat = v0.get("cItemCat")

    log(f"Item category: {item_cat}")

    # State
    goods_received = False
    invoice_cleared_before_gr = False

    violations = []

    # Timeline walk
    for i, e in enumerate(events):
        ts = e["ocel:timestamp"]
        act = e["ocel:activity"]
        res = e["ocel:vmap"].get("resource")

        log(f"\n[{i}] {ts} | {act} | resource={res}")

        if act == "Record Goods Receipt":
            goods_received = True
            log("  -> Goods received")

        elif act == "Clear Invoice":
            log("  -> Invoice cleared")
            if "3-way" in item_cat and not goods_received:
                invoice_cleared_before_gr = True
                violations.append("Invoice cleared before goods receipt")
                log("  !! VIOLATION: cleared before GR")

    # Post rules
    if "2-way" in item_cat:
        if any(e["ocel:activity"] == "Record Goods Receipt" for e in events):
            violations.append("Goods receipt exists for 2-way item")

    if "Consignment" in item_cat:
        if any("Invoice" in e["ocel:activity"] for e in events):
            violations.append("Invoice exists for consignment item")

    # Final result
    if violations:
        log("\nRESULT: ❌ NON-COMPLIANT")
        for v in violations:
            log(" - " + v)
    else:
        log("\nRESULT: ✅ COMPLIANT")

    return violations

# --------------------------------------------------
# Run on sample
# --------------------------------------------------

log("\nRunning revised compliance checker on sample items...\n")

violating_items = []

MAX_ITEMS = 10**5
count = 0

for cID, item_events in events_by_item.items():
    violations = check_item(cID, item_events)
    if violations:
        violating_items.append((cID, violations))
    count += 1
    if count >= MAX_ITEMS:
        break

# --------------------------------------------------
# Summary
# --------------------------------------------------

log("\n" + "=" * 80)
log("SUMMARY")
log(f"Checked {count} items")
log(f"Items with violations: {len(violating_items)}")

for cID, vs in violating_items:
    print(f"\nItem {cID}:")
    for v in vs:
        print(" - " + v)
