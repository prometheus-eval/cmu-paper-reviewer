"""Email notification service using aiosmtplib."""

import logging
from email.message import EmailMessage

import aiosmtplib

from backend.config import settings

logger = logging.getLogger(__name__)

REVIEW_BASE_URL = "https://prometheus-eval.github.io/cmu-paper-reviewer/review.html"


def _build_html_email(key: str) -> str:
    review_url = f"{REVIEW_BASE_URL}?key={key}"

    return f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f5f5f5;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f5;padding:40px 0;">
  <tr><td align="center">
    <table width="560" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);">

      <!-- Header -->
      <tr>
        <td style="background:#C41230;padding:28px 32px;text-align:center;">
          <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;letter-spacing:-0.3px;">
            CMU Paper Reviewer
          </h1>
        </td>
      </tr>

      <!-- Body -->
      <tr>
        <td style="padding:32px;">
          <h2 style="margin:0 0 12px 0;color:#171717;font-size:20px;font-weight:700;">
            Your Review is Ready
          </h2>
          <p style="margin:0 0 20px 0;color:#525252;font-size:15px;line-height:1.6;">
            Your AI-generated paper review has been completed. You can view the full review
            using your submission key below.
          </p>

          <!-- Key display -->
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr>
              <td style="background:#fdf2f4;border-radius:8px;padding:16px;text-align:center;">
                <span style="font-family:'Courier New',monospace;font-size:24px;font-weight:700;color:#C41230;letter-spacing:2px;">
                  {key}
                </span>
              </td>
            </tr>
          </table>

          <!-- CTA Button -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:24px;">
            <tr>
              <td align="center">
                <a href="{review_url}"
                   style="display:inline-block;background:#C41230;color:#ffffff;text-decoration:none;
                          padding:14px 32px;border-radius:8px;font-size:15px;font-weight:600;
                          letter-spacing:0.01em;">
                  View Your Review
                </a>
              </td>
            </tr>
          </table>

          <p style="margin:24px 0 0 0;color:#737373;font-size:13px;line-height:1.5;">
            You can also access your review directly at:<br>
            <a href="{review_url}" style="color:#C41230;text-decoration:underline;">{review_url}</a>
          </p>
        </td>
      </tr>

      <!-- Footer -->
      <tr>
        <td style="background:#fafafa;padding:20px 32px;border-top:1px solid #e5e5e5;text-align:center;">
          <p style="margin:0;color:#a3a3a3;font-size:12px;line-height:1.5;">
            Built at Carnegie Mellon University &middot; Backed by OpenHands SDK<br>
            This is an automated message. Please do not reply.
          </p>
        </td>
      </tr>

    </table>
  </td></tr>
</table>
</body>
</html>"""


def _build_plain_email(key: str) -> str:
    review_url = f"{REVIEW_BASE_URL}?key={key}"
    return (
        f"Hello,\n\n"
        f"Your AI-generated paper review is ready.\n\n"
        f"Submission Key: {key}\n\n"
        f"View your review at:\n{review_url}\n\n"
        f"Or retrieve it via the API: GET /api/review/{key}\n\n"
        f"Thank you for using CMU Paper Reviewer.\n\n"
        f"---\n"
        f"Built at Carnegie Mellon University. Backed by OpenHands SDK.\n"
        f"This is an automated message. Please do not reply.\n"
    )


async def send_review_ready_email(to_email: str, key: str) -> bool:
    """Send an email notifying the user that their review is ready.

    Returns True on success, False on failure.
    """
    if not settings.smtp_user or not settings.smtp_password:
        logger.warning("SMTP not configured — skipping email for key=%s", key)
        return False

    msg = EmailMessage()
    msg["Subject"] = f"Your paper review is ready — {key}"
    msg["From"] = settings.email_from
    msg["To"] = to_email

    # Plain text first (fallback)
    msg.set_content(_build_plain_email(key))

    # HTML alternative (preferred by email clients)
    msg.add_alternative(_build_html_email(key), subtype="html")

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_user,
            password=settings.smtp_password,
            start_tls=True,
        )
        logger.info("Email sent to %s for key=%s", to_email, key)
        return True
    except Exception:
        logger.exception("Failed to send email to %s for key=%s", to_email, key)
        return False
