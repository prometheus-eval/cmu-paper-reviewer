import sys
from backend.worker import SessionLocal
from sqlalchemy import select
from backend.models import Submission

key = sys.argv[1] if len(sys.argv) > 1 else "8d64soonaa4g"
s = SessionLocal()
sub = s.execute(select(Submission).where(Submission.key == key)).scalar_one_or_none()
if sub:
    print(f"Created: {sub.created_at}")
    print(f"Updated: {sub.updated_at}")
    print(f"Status: {sub.status}")
    print(f"Model: {sub.review_model_used}")
else:
    print("Not found")
s.close()
