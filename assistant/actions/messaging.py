"""Mail, Messages and Contacts through AppleScript. Sending only ever happens after the user said yes
(assistant.task asks first)."""

import re

from .errors import Failed
from .macos import quoted, tell_app

MAIL_LIMIT = 20   # newest inbox messages looked at
SNIPPET = 160     # characters of each message body shown to the LLM
SEP = "\x1f"      # field separator in AppleScript output (never in names or subjects)

INBOX_SCRIPT = f'''tell application "Mail"
  if (count of accounts) is 0 then return "NO ACCOUNTS"
  set out to ""
  set limit to {MAIL_LIMIT}
  if (count of messages of inbox) < limit then set limit to (count of messages of inbox)
  repeat with i from 1 to limit
    set m to message i of inbox
    set body to content of m
    if length of body > {SNIPPET} then set body to text 1 thru {SNIPPET} of body
    set out to out & (sender of m) & "{SEP}" & (subject of m) & "{SEP}" & (read status of m) & "{SEP}" & body & "{SEP}{SEP}"
  end repeat
  return out
end tell'''


def read_mail(sender: str = "") -> str:
    """The newest emails (from `sender`, else the unread ones), one line each, for the LLM to answer from."""
    out = tell_app("Mail", INBOX_SCRIPT)
    if out == "NO ACCOUNTS":
        raise Failed("Mail has no account yet. Add yours in the Mail app's settings")
    mails = [m.split(SEP) for m in out.split(SEP * 2) if m.count(SEP) == 3]
    if sender:
        mails = [m for m in mails if sender.lower() in m[0].lower()]
    else:
        mails = [m for m in mails if m[2] == "false"] or mails[:5]
    if not mails:
        return f"No emails from {sender}" if sender else "No emails"
    return "\n".join(f"From {who}: {subject} - {' '.join(body.split())}" for who, subject, _, body in mails[:8])


def send_email(to: str, subject: str, body: str) -> str:
    address = contact(to, "email")
    script = f'''tell application "Mail"
  set m to make new outgoing message with properties {{subject:{quoted(subject)}, content:{quoted(body)}, visible:false}}
  tell m to make new to recipient at end of to recipients with properties {{address:{quoted(address)}}}
  send m
end tell'''
    tell_app("Mail", script)
    return f"Sent the email to {to}"


def send_message(to: str, text: str) -> str:
    """An iMessage to a contact's phone number or email."""
    handle = contact(to, "phone")
    script = f'''tell application "Messages"
  set theBuddy to participant {quoted(handle)} of (1st account whose service type = iMessage)
  send {quoted(text)} to theBuddy
end tell'''
    tell_app("Messages", script)
    return f"Sent the message to {to}"


def contact(name: str, kind: str) -> str:
    """A person's email address or phone number from Contacts ("Mom", "Rahul"); an address or number as is."""
    if "@" in name or re.fullmatch(r"\+?[\d\s()-]{7,}", name.strip()):
        return name.strip()
    field = "emails" if kind == "email" else "phones"
    script = f'''tell application "Contacts"
  set found to (people whose name contains {quoted(name)} or nickname contains {quoted(name)})
  if found is {{}} then return ""
  set values to value of {field} of item 1 of found
  if values is {{}} then return ""
  return item 1 of values
end tell'''
    value = tell_app("Contacts", script)
    if not value:
        raise Failed(f"I couldn't find {'an email address' if kind == 'email' else 'a phone number'} for {name} "
                     f"in your contacts")
    return value
