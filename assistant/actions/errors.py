"""The two ways an action can fail on purpose. Their message is spoken to the user as-is."""


class Failed(Exception):
    """An action couldn't be done; the message says why ("I couldn't find an app called Foo")."""


class NeedsPermission(Failed):
    """macOS blocked the action; the message says which switch to turn on."""
