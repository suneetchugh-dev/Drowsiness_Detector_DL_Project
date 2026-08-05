"""
Synthetic dataset generator for Driver Drowsiness Detection (v2).

Mimics the *geometric crops* the detector actually feeds the CNNs:
  - eye crops  : eyebrow above + eye below, on textured skin  (48x48)
  - mouth crops: lips / mouth with surrounding skin           (48x48)

Only numpy + opencv-python (cv2) are used.
"""

import os
import random
import numpy as np
import cv2

random.seed(42)
np.random.seed(42)

IMG_SIZE = 48
N_PER_CLASS = 1200


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------

def skin_color():
    base = random.randint(60, 110)
    r = min(255, base + random.randint(15, 55))
    g = min(255, base + random.randint(8, 30))
    b = max(0, base - random.randint(0, 20))
    return (b, g, r)


def eyebrow_color(skin):
    return tuple(int(max(0, c - random.randint(30, 60))) for c in skin)


def textured_skin(img, skin):
    """flat skin + subtle tone gradient + pores."""
    h, w = img.shape[:2]
    grad = np.linspace(0, random.randint(4, 12), w, dtype=np.float32)
    img[:, :, 0] = np.clip(img[:, :, 0] + grad, 0, 255)
    noise = np.random.normal(0, 3.5, (h, w)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    return img


def apply_variation(img):
    img = cv2.GaussianBlur(img, (5, 5), random.uniform(0.3, 1.1))
    alpha = random.uniform(0.78, 1.22)
    beta = random.randint(-16, 16)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if random.random() < 0.6:
        noise = np.random.normal(0, random.uniform(2, 7), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        img = cv2.resize(img, (IMG_SIZE - 4, IMG_SIZE - 4))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def rotate(img, angle=None, maxa=10):
    if angle is None:
        angle = random.uniform(-maxa, maxa)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------
# eye generator (eye sits in the LOWER part of the crop, eyebrow above)
# ----------------------------------------------------------------------------

def draw_eyebrow(img, cx, y, skin):
    col = eyebrow_color(skin)
    w = random.randint(16, 22)
    for i in range(3):
        cv2.ellipse(img, (cx + random.randint(-2, 2), y + i), (w - i * 2, 2), 0,
                    190, 350, col, 2, lineType=cv2.LINE_AA)


def draw_open_eye(img, center):
    cx, cy = center
    skin = skin_color()
    img[:, :, :] = skin
    img = textured_skin(img, skin)

    draw_eyebrow(img, cx, cy - 14, skin)

    sclera_w, sclera_h = random.randint(16, 19), random.randint(10, 13)
    # eye region box (almond) on skin
    cv2.ellipse(img, (cx, cy), (sclera_w, sclera_h), 0, 0, 360, (235, 238, 242), -1)
    cv2.ellipse(img, (cx, cy), (sclera_w, sclera_h), 0, 0, 360, (0, 0, 0), 1)

    # iris: large, dark, prominent (mimics the strong iris/pupil signal of real open eyes)
    iris_r = random.randint(6, 8)
    iris = random.choice([(15, 45, 100), (20, 60, 130), (12, 40, 70),
                          (40, 75, 30), (60, 100, 150), (40, 85, 110)])
    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = (dist <= iris_r).astype(np.float32)
    iris_layer = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for c in range(3):
        iris_layer[..., c] = iris[c] * ring
    pupil_r = random.uniform(2.4, 3.4)
    pupil = (dist <= pupil_r).astype(np.float32)
    for c in range(3):
        iris_layer[..., c] += (8 - iris[c]) * pupil
    # highlight
    hl_r = 1.3
    hl = ((xx - (cx - 2)) ** 2 + (yy - (cy - 2)) ** 2) <= hl_r ** 2
    for c in range(3):
        iris_layer[..., c] += (255 - iris_layer[..., c]) * hl.astype(np.float32) * 0.9
    mask = ring > 0
    img[mask] = np.clip(iris_layer[mask], 0, 255).astype(np.uint8)

    # upper eyelid (skin) over the top of the iris + outline
    cv2.ellipse(img, (cx, cy - sclera_h // 2), (sclera_w + 2, sclera_h // 2 + 1), 0, 0, 180, skin, -1)
    cv2.ellipse(img, (cx, cy - sclera_h // 2), (sclera_w + 2, sclera_h // 2 + 1), 0, 0, 180, (20, 20, 20), 1)
    cv2.ellipse(img, (cx, cy + sclera_h // 2), (sclera_w + 2, sclera_h // 2), 0, 180, 360, (0, 0, 0), 1)
    cv2.ellipse(img, (cx, cy), (sclera_w, sclera_h), 0, 0, 360, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    # subtle lower-lid shadow
    cv2.ellipse(img, (cx, cy + sclera_h // 2 + 1), (sclera_w - 2, 2), 0, 180, 360,
                tuple(int(max(0, c - 25)) for c in skin), 1)

    # eyelashes (few, thin)
    for _ in range(random.randint(4, 7)):
        side = random.choice([-1, 1])
        x = int(cx + side * random.randint(sclera_w - 4, sclera_w + 1))
        y = int(cy - sclera_h + random.randint(0, 3))
        cv2.line(img, (x, y), (x + random.randint(-1, 1), y - random.randint(4, 6)), (0, 0, 0), 1)
    return img


def draw_closed_eye(img, center):
    cx, cy = center
    skin = skin_color()
    img[:, :, :] = skin
    img = textured_skin(img, skin)

    draw_eyebrow(img, cx, cy - 14, skin)

    # closed lid: gentle skin crease - LOW dark content, like real closed eyes
    w = random.randint(19, 23)
    crease = tuple(int(max(0, c - random.randint(20, 35))) for c in skin)
    cv2.ellipse(img, (cx, cy), (w, 4), 0, 0, 360, crease, 1, lineType=cv2.LINE_AA)
    # barely-there shadow above the crease
    shadow = np.zeros_like(img)
    cv2.ellipse(shadow, (cx, cy - 1), (w, 4), 0, 190, 350, (25, 25, 25), 2)
    img = cv2.addWeighted(img, 1.0, shadow, 0.18, 0)
    # crow's feet hint
    cv2.line(img, (cx - w + 4, cy - 2), (cx - w - 3, cy - 4), crease, 1)
    cv2.line(img, (cx + w - 4, cy - 2), (cx + w + 3, cy - 4), crease, 1)
    # fine lashes (thin, low contrast)
    for _ in range(random.randint(4, 7)):
        side = random.choice([-1, 1])
        x = int(cx + side * random.randint(w - 5, w))
        y = int(cy + random.randint(-1, 2))
        cv2.line(img, (x, y), (x + random.randint(-1, 1), y + random.randint(3, 5)), crease, 1)
    return img


def generate_eye(state):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    cx = random.randint(22, 26)
    cy = random.randint(29, 33)
    if state == "open":
        img = draw_open_eye(img, (cx, cy))
    else:
        img = draw_closed_eye(img, (cx, cy))
    img = rotate(img, maxa=9)
    img = apply_variation(img)
    return img


# ----------------------------------------------------------------------------
# mouth generator
# ----------------------------------------------------------------------------

def draw_no_yawn(img, center):
    cx, cy = center
    skin = skin_color()
    img[:, :, :] = skin
    img = textured_skin(img, skin)
    lip = random.choice([(90, 65, 150), (110, 75, 165), (85, 60, 135)])
    w = random.randint(14, 19)
    cv2.ellipse(img, (cx, cy - 1), (w, 4), 0, 0, 180, lip, -1)
    cv2.ellipse(img, (cx, cy + 1), (w, 3), 0, 180, 360, (65, 45, 110), -1)
    cv2.ellipse(img, (cx, cy), (w, 4), 0, 0, 360, (30, 25, 25), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (cx - w // 2 + 2, cy), (cx + w // 2 - 2, cy), (15, 10, 10), 1)
    cv2.ellipse(img, (cx, cy + 4), (w - 2, 2), 0, 180, 360, tuple(int(max(0, c - 30)) for c in skin), 1)
    return img


def draw_yawn(img, center):
    cx, cy = center
    skin = skin_color()
    img[:, :, :] = skin
    img = textured_skin(img, skin)
    w = random.randint(12, 17)
    h = random.randint(10, 14)
    # cavity
    cv2.ellipse(img, (cx, cy), (w, h), 0, 0, 360, (28, 12, 12), -1)
    # tongue (soft gradient)
    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    d = np.sqrt(((xx - (cx + random.randint(-2, 2))) / (w * 0.8)) ** 2 +
                ((yy - (cy + 2)) / (h * 0.55)) ** 2)
    tongue = (d <= 1.0).astype(np.float32)
    tcol = (random.randint(95, 130), random.randint(65, 95), random.randint(120, 160))
    for c in range(3):
        img[..., c] = np.where(tongue > 0,
                               img[..., c] * 0.25 + tcol[c] * tongue * 0.75,
                               img[..., c]).astype(np.uint8)
    # teeth
    cv2.ellipse(img, (cx, cy - h // 2 + 1), (w - 3, 3), 0, 180, 360, (230, 236, 242), -1)
    # lips border
    cv2.ellipse(img, (cx, cy), (w + 2, h + 2), 0, 0, 360, (80, 55, 140), 2, lineType=cv2.LINE_AA)
    # tension lines around the mouth
    for i in range(2):
        cv2.line(img, (cx - w - 6, cy - 2 + i * 2), (cx - w - 1, cy - 1 + i * 2), (40, 35, 30), 1)
        cv2.line(img, (cx + w + 6, cy - 2 + i * 2), (cx + w + 1, cy - 1 + i * 2), (40, 35, 30), 1)
    return img


def generate_mouth(state):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    cx = random.randint(22, 26)
    cy = random.randint(26, 30)
    if state == "yawn":
        img = draw_yawn(img, (cx, cy))
    else:
        img = draw_no_yawn(img, (cx, cy))
    img = rotate(img, maxa=7)
    img = apply_variation(img)
    return img


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def make_dataset():
    jobs = [
        ("data/eyes/open", generate_eye, "open", N_PER_CLASS),
        ("data/eyes/closed", generate_eye, "closed", N_PER_CLASS),
        ("data/mouth/yawn", generate_mouth, "yawn", N_PER_CLASS),
        ("data/mouth/no_yawn", generate_mouth, "no_yawn", N_PER_CLASS),
    ]
    for folder, fn, state, n in jobs:
        os.makedirs(folder, exist_ok=True)
        # clear old images from previous runs
        for old in os.listdir(folder):
            os.remove(os.path.join(folder, old))
        for i in range(n):
            img = fn(state=state)
            cv2.imwrite(os.path.join(folder, f"{state}_{i:05d}.png"), img)
        print(f"generated {n} -> {folder}")


if __name__ == "__main__":
    make_dataset()
    print("done")
