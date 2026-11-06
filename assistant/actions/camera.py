"""Looking through the Mac's camera when the user asks: "what am I holding?", "how do I look?".

One picture, looked at by the vision model and deleted right away; nothing is recorded or kept.
Needs the Camera permission for the app Laya runs in.
"""

from assistant.mac.permissions import camera_message

from .errors import NeedsPermission

PHOTO = "a photo just taken by the webcam of the user's Mac; the person in it is the user, talking to you"


def look_through_camera(question: str = "What do you see?") -> str:
    """Take one picture with the camera and answer the question about it."""
    from assistant.mac import vision
    try:
        picture = vision.photo()
    except vision.NoCameraAccess:
        raise NeedsPermission(camera_message()) from None
    try:
        return vision.ask(picture, question, PHOTO) or "I couldn't make out the picture"
    finally:
        vision.discard(picture)
