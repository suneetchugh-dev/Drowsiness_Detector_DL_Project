"""
Pitch-robust eye/mouth crop localisation shared by the live detector and the
calibration collector.

The old approach took fixed slices of the Haar face box (eyes 15-55%,
mouth 55-97% of face height). Those slices are wrong whenever the head is
tilted up/down, because the mouth slides toward the nose inside an
axis-aligned bounding box - so a tilted-down neutral face reads like a yawn.

These helpers localise the eyes (lefteye/righteye cascades) and the mouth
(smile cascade) *inside* the face box, and fall back to the geometric slice
when a cascade misses. Keeping one code path here guarantees that the crops
used at calibration time and at live-inference time are identical.
"""

import cv2

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
SMILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml")
LEFT_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
RIGHT_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")

MIN_FACE = 90


def detect_face(frame_gray):
    """Largest frontal face (same cascade settings everywhere)."""
    faces = FACE_CASCADE.detectMultiScale(
        frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE, MIN_FACE))
    return tuple(faces[0]) if len(faces) else None


def _slice_bboxes(face):
    """Fallback geometric geometry (identical to the historic crop logic)."""
    fx, fy, fw, fh = face
    ey1, ey2 = fy + int(fh * 0.15), fy + int(fh * 0.55)
    lx, lw = fx + int(fw * 0.05), int(fw * 0.42)
    rx = fx + int(fw * 0.53)
    left_eye = (lx, ey1, lw, ey2 - ey1)
    right_eye = (rx, ey1, lw, ey2 - ey1)
    my1, my2 = fy + int(fh * 0.55), fy + int(fh * 0.97)
    mx, mw = fx + int(fw * 0.22), int(fw * 0.56)
    mouth = (mx, my1, mw, my2 - my1)
    return left_eye, right_eye, mouth


def _clamp_box(bbox, frame_w, frame_h):
    x, y, w, h = bbox
    x, w = max(0, min(int(x), frame_w - 1)), max(1, int(w))
    y, h = max(0, min(int(y), frame_h - 1)), max(1, int(h))
    x = min(x, frame_w - w if frame_w - w >= 0 else x)
    y = min(y, frame_h - h if frame_h - h >= 0 else y)
    return x, y, w, h


def _localize_eyes(frame_bgr, face):
    """Cascade-localised eye boxes with geometric fallback.

    Returns ((lx, ly, lw, lh), (rx, ry, rw, rh))."""
    fx, fy, fw, fh = face
    le, re, _ = _slice_bboxes(face)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    roi_y1, roi_y2 = fy + int(fh * 0.08), fy + int(fh * 0.62)

    # left/right cascade each scan their own half of the upper face
    lroi = gray[roi_y1:roi_y2, fx:int(fx + fw * 0.52)]
    rroi = gray[roi_y1:roi_y2, int(fx + fw * 0.48):fx + fw]
    hits = {}

    for side, roi, xoff, cascade in (
            ("L", lroi, fx, LEFT_EYE_CASCADE),
            ("R", rroi, int(fx + fw * 0.48), RIGHT_EYE_CASCADE)):
        dets = cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3,
                                        minSize=(int(fw * 0.12), int(fh * 0.05)))
        good = []
        for (dx, dy, dw, dh) in dets:
            w, h = dw, dh
            ar = w / max(h, 1)
            if 0.8 <= ar <= 3.5 and 0.12 * fh <= h <= 0.35 * fh:
                good.append((dx + xoff, dy + roi_y1, dw, dh))
        if good:
            # biggest, most central candidate wins
            good.sort(key=lambda b: -b[2] * b[3])
            cx = good[0][0] + good[0][2] / 2.0
            hits[side] = min(good, key=lambda b: abs((b[0] + b[2] / 2.0) - cx))

    le = _pad_box(hits.get("L") or le, face, "eye", frame_bgr)
    re = _pad_box(hits.get("R") or re, face, "eye", frame_bgr)
    return le, re


def _localize_mouth(frame_bgr, face):
    """Smile-cascade mouth box (pitch-tolerant) with geometric fallback.

    The smile cascade scans a lower-face ROI for the actual mouth, so a
    downward-tilted head no longer shoves upper-lip/nostril shadow into the
    crop. Neutral mouths often produce a small detection too; when none fires
    we fall back and nudge the geometric slice using the detected eyes."""
    fx, fy, fw, fh = face
    _, _, mouth = _slice_bboxes(face)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    roi_y1, roi_y2 = fy + int(fh * 0.42), fy + int(fh * 1.00)
    roi = gray[roi_y1:roi_y2, max(fx - 5, 0):fx + fw + 5]
    xoff, yoff = max(fx - 5, 0), roi_y1

    dets = SMILE_CASCADE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=8,
                                          minSize=(int(fw * 0.15), int(fh * 0.06)))
    good = []
    for (dx, dy, dw, dh) in dets:
        w, h, ar = dw, dh, dw / max(dh, 1)
        if 1.0 <= ar <= 5.0 and 0.08 * fh <= h <= 0.45 * fh \
                and 0.15 * fw <= w <= 0.8 * fw:
            good.append((dx + xoff, dy + yoff, dw, dh))
    if good:
        # prefer the widest candidate, i.e. the widest mouth band
        good.sort(key=lambda b: -b[2])
        mouth = good[0]

    mouth = _pad_box(mouth, face, "mouth", frame_bgr)
    return mouth


def _pad_box(bbox, face, kind, frame_bgr):
    """Expand a localised box so it covers the full feature + a little margin.

    The eye/smile cascades return tight boxes; a yawn can lip out of a tight
    mouth box, and droopy eyes differ between looks. The padding factors are
    shared constant geometry so detector and collector stay in sync."""
    fx, fy, fw, fh = face
    x, y, w, h = bbox
    hh, ww = frame_bgr.shape[:2]
    if kind == "eye":
        px, py = int(w * 0.25), int(h * 0.30)
    else:  # mouth
        px, py = int(w * 0.10), int(h * 0.35)
    x2, y2 = x + w, y + h
    return _clamp_box((x - px, y - py, (x2 - x) + 2 * px, (y2 - y) + 2 * py),
                      ww, hh)


def localize_regions(frame_bgr, face):
    """(left_eye, right_eye, mouth) boxes aligned to the actual features."""
    fx, fy, fw, fh = face
    le, re = _localize_eyes(frame_bgr, face)
    mouth = _localize_mouth(frame_bgr, face)
    return le, re, mouth


def crop_regions(frame_bgr, face):
    """Returns the three crops as BGR numpy arrays: (left_eye, right_eye, mouth).

    Same signature/order used by detector.py, collect_calibration.py and the
    diagnostic scripts, so every consumer sees identical crops."""
    fx, fy, fw, fh = (int(v) for v in face)
    face = (fx, fy, fw, fh)
    le, re, mouth = localize_regions(frame_bgr, face)
    h, w = frame_bgr.shape[:2]
    if le[2] <= 0 or le[3] <= 0 or re[2] <= 0 or re[3] <= 0 \
            or mouth[2] <= 0 or mouth[3] <= 0:
        le, re, mouth = _slice_bboxes(face)
    def _clip(b):
        x, y, bw, bh = b
        x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
        bw = min(bw, w - x); bh = min(bh, h - y)
        return (x, y, bw, bh)
    le, re, mouth = map(_clip, (le, re, mouth))
    left_eye = frame_bgr[le[1]:le[1] + le[3], le[0]:le[0] + le[2]]
    right_eye = frame_bgr[re[1]:re[1] + re[3], re[0]:re[0] + re[2]]
    mouth = frame_bgr[mouth[1]:mouth[1] + mouth[3], mouth[0]:mouth[0] + mouth[2]]
    return left_eye, right_eye, mouth