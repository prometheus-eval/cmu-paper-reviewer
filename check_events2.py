import sys
import json
from backend.services.storage_service import find_trajectory_events

key = sys.argv[1] if len(sys.argv) > 1 else "cle75ft09l1f"
evts = find_trajectory_events(key)

for i, e in enumerate(evts[:5]):
    print(f"=== Event {e.get('_idx', i)} ===")
    print(f"  kind: {e.get('kind')}")
    print(f"  source: {e.get('source')}")
    print(f"  tool_name: {e.get('tool_name', 'N/A')}")
    print(f"  thought: {str(e.get('thought', ''))[:100]}")
    print(f"  summary: {str(e.get('summary', ''))[:100]}")
    # Print all top-level keys to find where the useful info is
    print(f"  all keys: {sorted(e.keys())}")
    # Check for nested content
    for k in ['action', 'content', 'message', 'args', 'observation', 'text']:
        if k in e:
            val = str(e[k])[:150]
            print(f"  {k}: {val}")
    print()
