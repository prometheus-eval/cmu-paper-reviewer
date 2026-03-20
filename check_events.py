import sys
import json
from backend.services.storage_service import find_trajectory_events

key = sys.argv[1] if len(sys.argv) > 1 else "cle75ft09l1f"
evts = find_trajectory_events(key)
print(f"Events found: {len(evts)}")
if evts:
    print(f"First event keys: {list(evts[0].keys())[:10]}")
    # Show which events have tool_name or thought
    action_count = 0
    for e in evts:
        if "tool_name" in e or "thought" in e or "summary" in e:
            action_count += 1
    print(f"Action events: {action_count}")
    # Print first 3 events summary
    for e in evts[:3]:
        print(f"  idx={e.get('_idx')} tool={e.get('tool_name','N/A')} summary={str(e.get('summary',''))[:80]}")
else:
    print("No events found")
