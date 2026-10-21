import json

PATH = "data/BPIC19.jsonocel"

# --------------------------------------------------
# Load
# --------------------------------------------------
with open(PATH, encoding="utf8") as f:
    ocel = json.load(f)

print("\n=== TOP-LEVEL KEYS ===")
for k in ocel:
    print(" ", k)

# --------------------------------------------------
# Global metadata
# --------------------------------------------------
print("\n=== GLOBAL METADATA ===")

print("\n-- Global Event --")
for k, v in ocel["ocel:global-event"].items():
    print(f"  {k}: {v}")

print("\n-- Global Object --")
for k, v in ocel["ocel:global-object"].items():
    print(f"  {k}: {v}")

print("\n-- Global Log --")
for k, v in ocel["ocel:global-log"].items():
    if isinstance(v, list):
        print(f"  {k}: list(len={len(v)})")
    else:
        print(f"  {k}: {v}")

# --------------------------------------------------
# Containers
# --------------------------------------------------
events = ocel["ocel:events"]
objects = ocel["ocel:objects"]

print("\n=== BASIC COUNTS ===")
print("Number of events :", len(events))
print("Number of objects:", len(objects))

# --------------------------------------------------
# Collect observed event attributes
# --------------------------------------------------
event_attrs = set()
for e in events.values():
    event_attrs.update(e.get("ocel:vmap", {}).keys())

print("\n=== EVENT ATTRIBUTES (ocel:vmap) ===")
for k in sorted(event_attrs):
    print(" ", k)

# --------------------------------------------------
# Collect observed object attributes
# --------------------------------------------------
object_attrs = set()
for o in objects.values():
    object_attrs.update(o.get("ocel:ovmap", {}).keys())

print("\n=== OBJECT ATTRIBUTES (ocel:ovmap) ===")
for k in sorted(object_attrs):
    print(" ", k)

# --------------------------------------------------
# Show one example event
# --------------------------------------------------
example_event_id = next(iter(events))
example_event = events[example_event_id]

print("\n=== EXAMPLE EVENT ===")
print("Event ID:", example_event_id)
print("Activity :", example_event.get("ocel:activity"))
print("Timestamp:", example_event.get("ocel:timestamp"))
print("Attributes:")
for k, v in example_event.get("ocel:vmap", {}).items():
    print(f"  {k}: {v}")
print("Related objects:", example_event.get("ocel:omap", []))

# --------------------------------------------------
# Show one referenced object
# --------------------------------------------------
obj_id = example_event["ocel:omap"][0]
obj = objects[obj_id]

print("\n=== EXAMPLE OBJECT ===")
print("Object ID:", obj_id)
print("Type     :", obj.get("ocel:type"))
print("Attributes:")
for k, v in obj.get("ocel:ovmap", {}).items():
    print(f"  {k}: {v}")
