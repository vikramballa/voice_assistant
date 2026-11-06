"""The face: a Dynamic-Island-style pill at the top of the screen (Siri look).

It runs as its own small process so a UI problem can never stop the assistant.
The assistant talks to it with one JSON line per update on the process's stdin:
    {"state": "listening" | "heard" | "thinking" | "reply" | "error" | "hide", "text": "..."}
This file is both sides: start()/show() run in the assistant, _run() is the window process.
It imports nothing from the package, so it can be started as a plain script.

Try it:
    python assistant/mac/island.py --demo
"""

import json
import subprocess
import sys

_proc = None


def start() -> None:
    """Launch the island window (called once at startup)."""
    global _proc
    try:
        _proc = subprocess.Popen([sys.executable, __file__], stdin=subprocess.PIPE, text=True)
    except OSError:
        _proc = None


def stop() -> None:
    """Close the island window (it quits when its stdin closes)."""
    global _proc
    if _proc is not None:
        try:
            _proc.stdin.close()
        except (OSError, ValueError):
            pass
        _proc = None


def show(state: str, text: str = "") -> None:
    """Update the island. Does nothing if it isn't running."""
    global _proc
    if _proc is None:
        return
    try:
        _proc.stdin.write(json.dumps({"state": state, "text": text}) + "\n")
        _proc.stdin.flush()
    except (OSError, ValueError):  # the window process died: carry on without it
        _proc = None


# ---------------------------------------------------------------- the window (own process)

ACTIVE = {"listening", "heard", "thinking"}  # dots pulse and the Siri glow is on
PLACEHOLDER = {"listening": "Listening…", "thinking": "Thinking…"}
HIDE_AFTER = {"reply": 2.5, "error": 4.0, "listening": 8.0}  # seconds; listening = nobody spoke
HEIGHT, MIN_W, MAX_W, PAD = 48, 190, 560, 22
SIRI = [(1.0, 0.36, 0.72), (0.66, 0.36, 1.0), (0.25, 0.55, 1.0), (0.2, 0.85, 1.0), (1.0, 0.62, 0.3)]


def _run(demo: bool = False) -> None:
    import threading

    import objc
    from AppKit import (NSApp, NSApplication, NSAnimationContext, NSBackingStoreBuffered, NSColor, NSFont,
                        NSLineBreakByTruncatingTail, NSMakeRect, NSPanel, NSScreen, NSTextField, NSView,
                        NSVisualEffectView)
    from Foundation import NSObject
    from PyObjCTools import AppHelper
    from Quartz import (CABasicAnimation, CAGradientLayer, CAKeyframeAnimation, CALayer, CAMediaTimingFunction,
                        CAShapeLayer, CATransaction, CGPathCreateWithRoundedRect, CGColorCreateSRGB)

    def rgb(r, g, b, a=1.0):
        return CGColorCreateSRGB(r, g, b, a)

    spring = CAMediaTimingFunction.functionWithControlPoints____(0.3, 1.3, 0.5, 1.0)  # slight overshoot

    class Island(NSObject):
        def init(self):
            self = objc.super(Island, self).init()
            screen = NSScreen.mainScreen()
            full, vis = screen.frame(), screen.visibleFrame()
            self.win_w, win_h = MAX_W + 60, HEIGHT + 40
            top = vis.origin.y + vis.size.height  # just under the menu bar / notch
            rect = NSMakeRect(full.origin.x + (full.size.width - self.win_w) / 2, top - win_h - 4, self.win_w, win_h)
            self.win = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, 1 << 7, NSBackingStoreBuffered, False)  # borderless, non-activating
            self.win.setOpaque_(False)
            self.win.setBackgroundColor_(NSColor.clearColor())
            self.win.setHasShadow_(False)
            self.win.setIgnoresMouseEvents_(True)
            self.win.setHidesOnDeactivate_(False)  # panels hide when their app is inactive - ours always is
            self.win.setLevel_(25 + 1)  # above the menu bar
            self.win.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))  # all spaces, stationary, full screen
            root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, self.win_w, win_h))
            root.setWantsLayer_(True)
            self.win.setContentView_(root)
            self.cy = win_h - HEIGHT / 2 - 6  # pill's vertical centre, near the top of the window

            # Siri glow: a colour gradient seen only through a rounded outline; the holder's shadow blurs it.
            self.glow_box = CALayer.layer()
            self.glow = CAGradientLayer.layer()
            self.glow.setType_("conic")
            self.glow.setStartPoint_((0.5, 0.5))
            self.glow.setEndPoint_((0.5, 0.0))
            self.glow.setColors_([rgb(*c) for c in SIRI + SIRI[:1]])
            self.glow_mask = CAShapeLayer.layer()
            self.glow_mask.setFillColor_(None)
            self.glow_mask.setStrokeColor_(rgb(0, 0, 0))
            self.glow_mask.setLineWidth_(3)
            self.glow.setMask_(self.glow_mask)
            self.glow_box.setShadowColor_(rgb(0.6, 0.4, 1.0))
            self.glow_box.setShadowRadius_(9)
            self.glow_box.setShadowOpacity_(1.0)
            self.glow_box.setShadowOffset_((0, 0))
            self.glow_box.setOpacity_(0)
            self.glow_box.addSublayer_(self.glow)
            root.layer().addSublayer_(self.glow_box)
            spin = CAKeyframeAnimation.animationWithKeyPath_("colors")
            spin.setValues_([[rgb(*c) for c in (SIRI[i:] + SIRI[:i] + SIRI[i:i + 1])] for i in range(len(SIRI) + 1)])
            spin.setDuration_(3.0)
            spin.setRepeatCount_(1e9)
            self.glow.addAnimation_forKey_(spin, "spin")

            # The pill: dark blurred glass with a hairline edge.
            self.pill = NSVisualEffectView.alloc().initWithFrame_(self._pill_rect(MIN_W * 0.4, 0.6))
            self.pill.setMaterial_(13)       # HUD window
            self.pill.setBlendingMode_(0)    # blur what's behind the window
            self.pill.setState_(1)           # always active
            self.pill.setWantsLayer_(True)
            layer = self.pill.layer()
            layer.setCornerRadius_(HEIGHT * 0.35)
            layer.setMasksToBounds_(True)
            layer.setBorderWidth_(0.5)
            layer.setBorderColor_(rgb(1, 1, 1, 0.18))
            tint = CALayer.layer()           # darker than the stock HUD, like the Siri pill
            tint.setBackgroundColor_(rgb(0, 0, 0, 0.7))
            tint.setFrame_(((0, 0), (MAX_W, HEIGHT)))
            layer.addSublayer_(tint)
            root.addSubview_(self.pill)

            # Three dots that breathe while listening/thinking.
            self.dots = []
            for i, (x, y, size) in enumerate([(16, 25, 7), (27, 15, 5), (26, 31, 4)]):
                dot = CALayer.layer()
                dot.setFrame_(((x, y), (size, size)))
                dot.setCornerRadius_(size / 2)
                dot.setBackgroundColor_(rgb(1, 1, 1))
                dot.setShadowColor_(rgb(1, 1, 1))
                dot.setShadowRadius_(4)
                dot.setShadowOpacity_(0.9)
                dot.setShadowOffset_((0, 0))
                pulse = CABasicAnimation.animationWithKeyPath_("transform.scale")
                pulse.setFromValue_(0.55)
                pulse.setToValue_(1.15)
                pulse.setDuration_(0.6)
                pulse.setAutoreverses_(True)
                pulse.setRepeatCount_(1e9)
                pulse.setTimeOffset_(i * 0.25)
                dot.addAnimation_forKey_(pulse, "pulse")
                layer.addSublayer_(dot)
                self.dots.append(dot)

            self.label = NSTextField.labelWithString_("")
            self.label.setFont_(NSFont.systemFontOfSize_weight_(15, 0.23))  # medium
            self.label.setTextColor_(NSColor.whiteColor())
            self.label.setLineBreakMode_(NSLineBreakByTruncatingTail)
            self.pill.addSubview_(self.label)
            self.visible = False
            return self

        def _pill_rect(self, width, scale=1.0):
            w, h = width * scale, HEIGHT * scale
            return NSMakeRect((self.win_w - w) / 2, self.cy - h / 2, w, h)

        def apply_(self, msg):
            state, text = msg.get("state", "hide"), msg.get("text") or PLACEHOLDER.get(msg.get("state"), "")
            NSObject.cancelPreviousPerformRequestsWithTarget_(self)
            if state == "hide":
                return self.hide()
            active = state in ACTIVE
            self.label.setStringValue_(text)
            self.label.setTextColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(1, 0.55, 0.55, 1) if state == "error"
                                     else NSColor.whiteColor())
            self.label.setAlphaValue_(0.75 if state in PLACEHOLDER else 1.0)
            text_w = self.label.intrinsicContentSize().width
            width = min(max(text_w + 54 + PAD, MIN_W), MAX_W)  # +6: the text field's own inset
            self.label.setFrame_(NSMakeRect(46, (HEIGHT - 20) / 2, width - 46 - PAD, 20))

            if not self.visible:  # grow out of a small blob, like the island
                self.pill.setFrame_(self._pill_rect(width * 0.35, 0.7))
                self.pill.setAlphaValue_(0)
                self.visible = True
            self.win.orderFrontRegardless()  # every time: an order-front during app launch can be lost
            final = self._pill_rect(width)
            def change(ctx):
                ctx.setDuration_(0.45)
                ctx.setTimingFunction_(spring)
                self.pill.animator().setFrame_(final)
                self.pill.animator().setAlphaValue_(1)
            NSAnimationContext.runAnimationGroup_(change)
            self.pill.layer().setCornerRadius_(HEIGHT / 2)

            CATransaction.begin()
            CATransaction.setAnimationDuration_(0.45)
            CATransaction.setAnimationTimingFunction_(spring)
            w, h = final.size.width + 4, final.size.height + 4
            self.glow_box.setFrame_(((final.origin.x - 2, final.origin.y - 2), (w, h)))
            self.glow.setFrame_(((0, 0), (w, h)))
            r = (h - 3) / 2 - 0.01  # CoreGraphics rejects a radius over half the height
            self.glow_mask.setPath_(CGPathCreateWithRoundedRect(((1.5, 1.5), (w - 3, h - 3)), r, r, None))
            self.glow_box.setOpacity_(1.0 if active else 0.0)
            for dot in self.dots:
                dot.setOpacity_(1.0 if active else 0.35)
                dot.setSpeed_(1.0 if active else 0.0)
            CATransaction.commit()

            if state in HIDE_AFTER:
                self.performSelector_withObject_afterDelay_("hide", None, HIDE_AFTER[state])

        def hide(self):
            if not self.visible:
                return
            self.visible = False
            small = self._pill_rect(self.pill.frame().size.width * 0.35, 0.7)
            def change(ctx):
                ctx.setDuration_(0.3)
                self.pill.animator().setFrame_(small)
                self.pill.animator().setAlphaValue_(0)
            NSAnimationContext.runAnimationGroup_completionHandler_(
                change, lambda: None if self.visible else self.win.orderOut_(None))
            self.glow_box.setOpacity_(0)

    NSApplication.sharedApplication().setActivationPolicy_(2)  # no Dock icon, never steals focus
    island = Island.alloc().init()

    def read_stdin():
        for line in sys.stdin:
            try:
                AppHelper.callAfter(island.apply_, json.loads(line))
            except ValueError:
                pass
        AppHelper.callAfter(NSApp().terminate_, None)  # stdin closed: the assistant quit
        # That request was sometimes lost and the window outlived the assistant (2 found 2026-10-03): make sure.
        import os
        import time
        time.sleep(2)
        os._exit(0)

    def play_demo():
        import time
        time.sleep(0.5)  # let the app finish launching
        for state, text, wait in [("listening", "", 1.5), ("heard", "open Chrome and search YouTube for lo-fi", 1.2),
                                  ("thinking", "", 1.5), ("reply", "Searching YouTube for lo-fi", 3.5),
                                  ("listening", "", 1.5), ("error", "Sorry, that didn't work", 5)]:
            AppHelper.callAfter(island.apply_, {"state": state, "text": text})
            time.sleep(wait)
        AppHelper.callAfter(NSApp().terminate_, None)

    threading.Thread(target=play_demo if demo else read_stdin, daemon=True).start()
    AppHelper.runEventLoop(installInterrupt=True)


if __name__ == "__main__":
    _run(demo="--demo" in sys.argv)
