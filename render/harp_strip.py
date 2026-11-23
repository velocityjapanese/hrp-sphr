"""Per-frame harp view: a real CC0 pedal harp, strings plucked.

The instrument is a CC0 photograph of a French pedal harp ca. 1800 from The
Metropolitan Museum of Art (object 501663), background removed and placed
upright on the stage. The photographed strings are replaced with drawn
strings that snap to the measured fan between the neck and the soundboard
ridge, so each pluck can flash, vibrate and decay on its own. C strings are
tinted red and F strings blue, following harp convention.
"""
import math
import queue
import random
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import FPS, H, W
from render.scene import load_font, make_spark_sprite, palette
from render.score_layer import PANEL_H, PANEL_W, PANEL_X, PANEL_Y

# ---------------------------------------------------------------------------
# String frame geometry, measured in photo pixels (1233 x 1861).
# ---------------------------------------------------------------------------
PHOTO_W, PHOTO_H = 1233, 1861

NECK_CURVE = np.array([
    (452, 252), (480, 275), (520, 308), (580, 348), (640, 392), (700, 444),
    (745, 492), (772, 508), (795, 522), (812, 548), (828, 572), (840, 582),
    (852, 594),
], dtype=np.float64)
SB_CURVE = np.array([
    (452, 1452), (470, 1408), (500, 1350), (523, 1300), (568, 1200),
    (616, 1100), (662, 1000), (710, 900), (756, 800), (790, 720),
    (806, 682), (826, 650), (838, 622), (848, 606), (852, 598),
], dtype=np.float64)

N_STRINGS = 40
STRING_X = np.linspace(468.0, 848.0, N_STRINGS)
Y_TOP = np.interp(STRING_X, NECK_CURVE[:, 0], NECK_CURVE[:, 1])
Y_BOT = np.interp(STRING_X, SB_CURVE[:, 0], SB_CURVE[:, 1])

PLUCK_S = 0.55          # pluck point as a fraction of the sounding length

# Whole-layer placement on screen (no tilt: the harp stands on the stage).
# The harp sits clear of the brand row on the left and the score panel on the
# right; the movement title is placed in the empty stage below the panel.
SCALE = 0.55
TX, TY = 220.0, 30.0

BODY_PHOTO = Path(__file__).resolve().parents[1] / "assets" / "harp_body.jpg"

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
_LETTER_OF_PC = (0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6)


def _string_base_pitch(i):
    return 24 + 12 * (i // 7) + _MAJOR_STEPS[i % 7]


def _string_index(pitch):
    """String carrying the pitch: pedal harps retune the same string."""
    letter = _LETTER_OF_PC[int(pitch) % 12]
    octave = int(pitch) // 12 - 1
    return max(0, min(N_STRINGS - 1, (octave - 1) * 7 + letter))


def _string_color(i):
    f = i / (N_STRINGS - 1)
    col = (np.array((222, 204, 172), np.float32) * (1 - f)
           + np.array((240, 244, 252), np.float32) * f)
    if i % 7 == 0:
        col = col * 0.62 + np.array((238, 112, 104), np.float32) * 0.38
    elif i % 7 == 3:
        col = col * 0.62 + np.array((126, 166, 242), np.float32) * 0.38
    return col


# ---------------------------------------------------------------------------
# Cut-out
# ---------------------------------------------------------------------------
def _photo_mask(img):
    """Alpha for the instrument: the backdrop is a low-saturation studio blue."""
    R = img[:, :, 0].astype(np.int16)
    G = img[:, :, 1].astype(np.int16)
    B = img[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    bg = (B >= R - 2) & (mx > 105) & ((mx - mn) < 42)
    fg = (~bg).astype(np.uint8)

    # Small enclosed backdrop-coloured speckles are bluish paint on the
    # soundboard, not background: fill holes that never touch the border.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bg.astype(np.uint8), 8)
    h, w = bg.shape
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area >= 4000:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        fg[lab == k] = 1

    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (lab == keep).astype(np.uint8)

    # The photographed strings are replaced by drawn ones, so clear the band
    # between the neck and the soundboard ridge.
    poly = np.concatenate([
        np.array([(452.0, 256.0)]),
        np.stack([STRING_X, Y_TOP + 3], axis=1),
        np.stack([STRING_X[::-1], Y_BOT[::-1] - 1], axis=1),
        np.array([(452.0, 1448.0)]),
    ]).astype(np.int32)
    clear = np.zeros_like(fg)
    cv2.fillPoly(clear, [poly], 1)
    fg[clear > 0] = 0

    return cv2.GaussianBlur(fg.astype(np.float32) * 255, (0, 0), 1.4)


_BODY_CACHE = {}


def _body_photo():
    cached = _BODY_CACHE.get("alpha")
    if cached is not None:
        return cached
    img = cv2.imread(str(BODY_PHOTO), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"harp body photo missing: {BODY_PHOTO}")
    rgb = img[:, :, ::-1]
    alpha = _photo_mask(rgb)
    _BODY_CACHE["alpha"] = (rgb, alpha)
    return rgb, alpha


def _draw_static_strings(layer):
    """Rest-position strings, so the harp reads even before it is played."""
    for i in range(N_STRINGS - 1, -1, -1):
        x0 = STRING_X[i]
        ys = np.arange(int(Y_TOP[i]) + 1, max(int(Y_TOP[i]) + 5, int(Y_BOT[i]) - 1))
        xc = int(round(x0))
        f = i / (N_STRINGS - 1)
        half = 1.25 - 0.75 * f
        col = _string_color(i)
        for dx in range(-2, 3):
            wgt = 1.0 - abs(dx) / (half + 0.7)
            if wgt <= 0.08:
                continue
            x = xc + dx
            if 0 <= x < PHOTO_W:
                shade = np.clip(col * (0.55 + 0.45 * wgt), 0, 255).astype(np.uint8)
                layer[ys, x, :3] = shade
                layer[ys, x, 3] = 255


def make_harp_base(palette_name):
    pal = palette(palette_name)
    accent = np.array(pal["accent"], dtype=np.float32)
    rgb, alpha = _body_photo()

    canvas = Image.new("RGBA", (PHOTO_W, PHOTO_H), (0, 0, 0, 0))
    sh = Image.new("RGBA", (PHOTO_W, PHOTO_H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse((300, 1806, 770, 1861), fill=(0, 0, 0, 150))
    sh = sh.filter(ImageFilter.GaussianBlur(20))
    canvas.alpha_composite(sh)

    a8 = np.clip(alpha, 0, 255).astype(np.uint8)
    halo = Image.fromarray(np.dstack([np.zeros((PHOTO_H, PHOTO_W, 3), np.uint8),
                                      (a8 * 0.45).astype(np.uint8)]), "RGBA")
    halo = halo.transform((PHOTO_W, PHOTO_H), Image.AFFINE,
                          (1, 0, -10, 0, 1, -8), resample=Image.BILINEAR)
    halo = halo.filter(ImageFilter.GaussianBlur(20))
    canvas.alpha_composite(halo)

    # The museum photograph is lit for the archive, not for a stage: lift the
    # wood out of the shadows, add a little saturation, then tint the whole
    # instrument with the palette so it sits inside the scene.
    body_rgb = rgb.astype(np.float32) / 255.0
    body_rgb = np.clip(body_rgb, 0.0, 1.0) ** 0.90
    gray = body_rgb.mean(axis=2, keepdims=True)
    body_rgb = np.clip(gray + (body_rgb - gray) * 1.12, 0.0, 1.0)
    body = np.dstack([np.clip(body_rgb * 255.0 + accent[None, None, :] * 0.032,
                              0, 255).astype(np.uint8), a8])
    canvas.alpha_composite(Image.fromarray(body, "RGBA"))

    layer = np.array(canvas, dtype=np.uint8)
    _draw_static_strings(layer)
    return layer


# ---------------------------------------------------------------------------
# Glow helpers
# ---------------------------------------------------------------------------
def _glow_sprite(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    return np.clip(1.0 - r, 0, 1) ** 2.6


def _clip_add(roi, add):
    roi[:] = np.minimum(roi.astype(np.uint16) + add.astype(np.uint16), 255).astype(np.uint8)


class HarpStripGenerator:
    def __init__(self, section, notes, seed=0):
        pal = palette(section["palette"])
        self.key_color = np.array(pal["key_glow"], dtype=np.float32)
        self.accent = np.array(pal["accent"], dtype=np.float32)
        self.base = make_harp_base(section["palette"])
        self.pluck_glow = _glow_sprite(56)
        self.spark = make_spark_sprite(14)
        glow_img = Image.fromarray((_glow_sprite(96) * 255).astype(np.uint8), "L")
        self.board_glow = (np.asarray(glow_img.resize((420, 880), Image.Resampling.BICUBIC),
                                      dtype=np.float32) / 255.0)
        self.M = np.array([[SCALE, 0.0, TX], [0.0, SCALE, TY]])

        self.y0 = np.clip(np.round(Y_TOP).astype(int) + 1, 2, PHOTO_H - 6)
        self.y1 = np.clip(np.round(Y_BOT).astype(int) - 1, self.y0 + 4, PHOTO_H - 3)
        self.len = (self.y1 - self.y0).astype(np.float64)
        self.ys = [np.arange(self.y0[i], self.y1[i]) for i in range(N_STRINGS)]
        self.pluck_y = Y_TOP + PLUCK_S * (Y_BOT - Y_TOP) + 3
        f = np.arange(N_STRINGS) / (N_STRINGS - 1.0)
        self.amp_max = np.clip(self.len / 120.0, 1.4, 11.0)
        self.wl = np.clip(self.len * 0.18, 40.0, 230.0)
        self.omega = 2.0 * np.pi * (2.2 + 2.4 * f)
        self.tau = 2.8 - 1.9 * f
        self.colors = [_string_color(i) for i in range(N_STRINGS)]

        self.energy = np.zeros(N_STRINGS, dtype=np.float32)
        self.flash = np.zeros(N_STRINGS, dtype=np.float32)
        self.phase = np.zeros(N_STRINGS, dtype=np.float32)
        self.last_meta = [(36, False)] * N_STRINGS

        self.badge_sprites = {}
        self.badge_key = None
        self.badge_si = 0
        self.badge_alpha = 0.0
        self.badge_since = -10.0
        self.badge_hold = 0.3

        self.section = section
        self.rng = random.Random(seed * 7919 + section["index"] * 104729)

        sec_start, sec_end = section["start"], section["end"]
        self.notes = [n for n in notes if n["start"] <= sec_end and n["end"] >= sec_start - 0.4]
        self.harp_notes = [n for n in self.notes if n["hand"] == "R"]
        self.piano_notes = [n for n in self.notes if n["hand"] == "L"]
        self.frame_events = {}
        for n in self.harp_notes:
            fr = int(round((n["start"] - sec_start) * FPS))
            if 0 <= fr < section["frames"]:
                self.frame_events.setdefault(fr, []).append(n)

        self.active_piano = []
        self.next_piano = 0
        self.particles = []

    # ------------------------------------------------------------------
    def frames(self):
        sec = self.section
        frames = sec["frames"]
        dt = 1.0 / FPS
        for i in range(frames):
            t = sec["start"] + i * dt
            layer = self.base.copy()
            piano = self._piano_level(t)
            self._on_onsets(i, t)
            self._decay(dt)
            self._draw_strings(layer, t)
            self._draw_board(layer, piano)
            self._update_particles(layer, dt)
            frame = cv2.warpAffine(layer, self.M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
            self._update_badge(t)
            self._draw_badge(frame)
            yield frame.tobytes()

    def _piano_level(self, t):
        while (self.next_piano < len(self.piano_notes)
               and self.piano_notes[self.next_piano]["start"] <= t):
            self.active_piano.append(self.piano_notes[self.next_piano])
            self.next_piano += 1
        self.active_piano = [n for n in self.active_piano if n["end"] + 0.25 >= t]
        level = 0.0
        for n in self.active_piano:
            if t < n["start"]:
                continue
            rel = t - n["start"]
            value = min(1.0, rel / 0.012) * (0.5 + 0.5 * math.exp(-rel / 0.55)) \
                * (n["vel"] / 127.0)
            if t > n["end"]:
                value *= math.exp(-(t - n["end"]) / 0.13)
            level = max(level, value)
        return level

    def _on_onsets(self, frame_index, t):
        for n in self.frame_events.get(frame_index, ()):
            si = _string_index(n["pitch"])
            self.energy[si] = min(1.2, self.energy[si] + 0.55 + n["vel"] / 190.0)
            self.flash[si] = 1.0
            self.phase[si] = self.rng.uniform(0.0, 2.0 * math.pi)
            self.last_meta[si] = (int(n["pitch"]), bool(n.get("arp", False)))
            self._spawn(si, n)

    def _decay(self, dt):
        self.energy *= np.exp(-dt / self.tau).astype(np.float32)
        self.flash *= math.exp(-dt / 0.11)
        self.energy[self.energy < 0.004] = 0.0
        self.flash[self.flash < 0.02] = 0.0

    # ------------------------------------------------------------------
    def _draw_strings(self, layer, t):
        for si in range(N_STRINGS):
            e = float(self.energy[si])
            if e <= 0.02:
                continue
            ys = self.ys[si]
            y0 = self.y0[si]
            L = max(1.0, self.len[si])
            s = (ys - y0) / L
            env = np.sin(np.pi * s) ** 0.85
            amp = self.amp_max[si] * e * 1.15
            xw = (STRING_X[si] + amp * env
                  * np.sin(2.0 * np.pi * (ys - self.pluck_y[si]) / self.wl[si]
                           - t * self.omega[si] + self.phase[si]))
            xi = np.clip(np.round(xw).astype(np.int32), 2, PHOTO_W - 3)

            strength = (0.5 + 0.85 * e) * env
            strength *= 1.0 + 0.7 * np.exp(-((ys - self.pluck_y[si]) / 46.0) ** 2)
            strength = np.clip(strength, 0.0, 1.15)

            core = (self.key_color[None, :] * (255.0 * strength)[:, None]
                    + self.colors[si][None, :] * (60.0 * strength)[:, None])
            core = np.clip(core, 0, 255).astype(np.uint16)
            cur = layer[ys, xi, :3].astype(np.uint16)
            layer[ys, xi, :3] = np.minimum(cur + core, 255).astype(np.uint8)
            layer[ys, xi, 3] = 255
            for dx, wgt in ((-3, 0.09), (-2, 0.18), (-1, 0.40),
                            (1, 0.40), (2, 0.18), (3, 0.09)):
                xh = np.clip(xi + dx, 0, PHOTO_W - 1)
                add = (self.key_color[None, :]
                       * (255.0 * strength * wgt)[:, None]).astype(np.uint16)
                cur = layer[ys, xh, :3].astype(np.uint16)
                layer[ys, xh, :3] = np.minimum(cur + add, 255).astype(np.uint8)
                a = np.clip(strength * wgt * 330.0, 0, 255).astype(np.uint8)
                layer[ys, xh, 3] = np.maximum(layer[ys, xh, 3], a)

            fl = float(self.flash[si])
            if fl > 0.03:
                px, py = STRING_X[si], self.pluck_y[si]
                self._stamp(layer, px, py, self.pluck_glow, self.key_color, 0.9 * fl)
                self._stamp(layer, px, py, self.spark, self.accent, 0.8 * fl)

    def _draw_board(self, layer, piano):
        if piano <= 0.01:
            return
        sh, sw = self.board_glow.shape
        cx, cy = 720, 1090
        x0, y0 = cx - sw // 2, cy - sh // 2
        cy0, cy1 = max(0, y0), min(PHOTO_H, y0 + sh)
        cx0, cx1 = max(0, x0), min(PHOTO_W, x0 + sw)
        if cy1 <= cy0 or cx1 <= cx0:
            return
        sub = self.board_glow[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][:, :, None]
        add = (sub * self.accent[None, None, :] * (0.15 * piano)).astype(np.uint16)
        _clip_add(layer[cy0:cy1, cx0:cx1, :3], add)

    def _stamp(self, strip, cx, cy, sprite, color, intensity):
        sh = sprite.shape[0]
        x0, y0 = int(round(cx - sh / 2)), int(round(cy - sh / 2))
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(PHOTO_W, x0 + sh), min(PHOTO_H, y0 + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][:, :, None]
        add = (sub * color[None, None, :] * (255.0 * min(1.0, intensity))).astype(np.uint16)
        _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)

    # ------------------------------------------------------------------
    def _spawn(self, si, note):
        count = 2 + int(note["vel"] / 60)
        x = STRING_X[si]
        y = self.pluck_y[si]
        for _ in range(count):
            self.particles.append({
                "x": x + self.rng.uniform(-5, 5),
                "y": y + self.rng.uniform(-4, 8),
                "vx": self.rng.uniform(-30, 30),
                "vy": self.rng.uniform(-18, 34),
                "life": self.rng.uniform(0.35, 0.8),
                "age": 0.0,
            })

    def _update_particles(self, strip, dt):
        alive = []
        sprite = self.spark
        sh = sprite.shape[0]
        for p in self.particles:
            p["age"] += dt
            if p["age"] >= p["life"]:
                continue
            p["x"] += p["vx"] * dt
            p["vy"] += 26 * dt
            p["y"] += p["vy"] * dt
            life = 1.0 - p["age"] / p["life"]
            x, y = int(p["x"]), int(p["y"])
            x0, y0 = x - sh // 2, y - sh // 2
            x1, y1 = x0 + sh, y0 + sh
            if x1 <= 0 or y1 <= 0 or x0 >= PHOTO_W or y0 >= PHOTO_H:
                continue
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(PHOTO_W, x1), min(PHOTO_H, y1)
            sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            add = (sub[:, :, None] * self.accent[None, None, :]
                   * (life ** 1.4) * 190).astype(np.uint16)
            _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)
            alive.append(p)
        self.particles = alive

    # ------------------------------------------------------------------
    def _badge_sprite(self, pitch, si, arp):
        key = (pitch, si, arp)
        sprite = self.badge_sprites.get(key)
        if sprite is not None:
            return sprite
        note = f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"
        pc = pitch % 12
        if arp:
            detail = "rolled chord"
        elif pc == 0:
            detail = "C string"
        elif pc == 5:
            detail = "F string"
        else:
            detail = "plucked"
        f_note = load_font("sans", 14)
        f_detail = load_font("sans", 11)
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        nw = int(probe.textlength(note, font=f_note))
        dw = int(probe.textlength(detail, font=f_detail))
        pad_x, gap, h = 10, 8, 29
        w = pad_x * 2 + nw + gap + dw
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill=(7, 10, 17, 214),
                            outline=tuple(int(c) for c in self.accent) + (105,), width=1)
        d.text((pad_x, 6), note, font=f_note, fill=(236, 243, 255, 240))
        d.text((pad_x + nw + gap, 9), detail, font=f_detail,
               fill=(204, 214, 232, 225))
        arr = np.asarray(img, dtype=np.float32)
        sprite = (arr[:, :, :3], arr[:, :, 3] / 255.0, w, h)
        self.badge_sprites[key] = sprite
        return sprite

    def _update_badge(self, t):
        loudest = int(np.argmax(self.energy))
        if self.energy[loudest] > 0.05:
            pitch, arp = self.last_meta[loudest]
            key = (pitch, loudest, arp)
            if key != self.badge_key and t - self.badge_since >= self.badge_hold:
                self.badge_key = key
                self.badge_si = loudest
                self.badge_since = t
            target = 1.0
        else:
            target = 0.0
        self.badge_alpha += (target - self.badge_alpha) * 0.18

    def _draw_badge(self, frame):
        if self.badge_key is None or self.badge_alpha < 0.03:
            return
        pitch, si, arp = self.badge_key
        rgb, alpha, w, h = self._badge_sprite(pitch, si, arp)
        sx, sy = self._project(STRING_X[si], self.pluck_y[si])
        x = int(round(sx + 20))
        y = int(round(sy + 14))
        if (x + w > PANEL_X - 6 and x < PANEL_X + PANEL_W + 6
                and y + h > PANEL_Y - 6 and y < PANEL_Y + PANEL_H + 6):
            x = int(round(sx - w - 20))
        x = max(16, min(W - w - 16, x))
        y = max(16, min(H - h - 16, y))
        self._blend(frame, x, y, rgb, alpha, min(1.0, self.badge_alpha))

    def _project(self, x, y):
        return (self.M[0, 0] * x + self.M[0, 2],
                self.M[1, 1] * y + self.M[1, 2])

    def _blend(self, frame, x, y, rgb, alpha, mul=1.0):
        sh, sw = alpha.shape
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(W, x + sw), min(H, y + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        a = (alpha[cy0 - y:cy1 - y, cx0 - x:cx1 - x] * mul)[:, :, None]
        roi = frame[cy0:cy1, cx0:cx1]
        roi[:, :, :3] = np.clip(roi[:, :, :3] * (1.0 - a)
                                + rgb[cy0 - y:cy1 - y, cx0 - x:cx1 - x] * a,
                                0, 255).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], (a[:, :, 0] * 255).astype(np.uint8))


class PipeWriter:
    """Writes frames to a subprocess stdin on a background thread."""

    def __init__(self, stream, maxsize=4):
        self.stream = stream
        self.q = queue.Queue(maxsize=maxsize)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, data):
        while not self.error:
            try:
                self.q.put(data, timeout=0.5)
                return
            except queue.Full:
                continue

    def close(self):
        while True:
            try:
                self.q.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        self.thread.join()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            if self.error is not None:
                continue
            try:
                self.stream.write(item)
            except (BrokenPipeError, ValueError, OSError) as exc:
                self.error = exc


def stream_section(section, notes, stream, seed=0):
    gen = HarpStripGenerator(section, notes, seed)
    writer = PipeWriter(stream)
    try:
        for frame in gen.frames():
            writer.write(frame)
    finally:
        writer.close()
        try:
            stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    out = Path(__file__).resolve().parents[1] / "_src"
    section = {"index": 0, "name": "I. Awakening", "palette": "midnight",
               "start": 0.0, "end": 6.0, "frames": 180, "bpm": 64}
    notes = []
    for k, pitch in enumerate((48, 52, 55, 60, 64, 67)):
        notes.append({"start": 0.6 + k * 0.045, "end": 3.6, "pitch": pitch,
                      "vel": 84, "hand": "R", "arp": True})
    notes.append({"start": 3.1, "end": 4.4, "pitch": 72, "vel": 92, "hand": "R", "arp": False})
    notes.append({"start": 3.4, "end": 4.6, "pitch": 74, "vel": 92, "hand": "R", "arp": False})
    notes.append({"start": 0.0, "end": 3.0, "pitch": 40, "vel": 66, "hand": "L", "arp": False})

    t0 = time.time()
    gen = HarpStripGenerator(section, notes, 7)
    picks = {12: "frame_02.png", 40: "frame_14.png", 70: "frame_24.png", 120: "frame_40.png"}
    bg = Image.new("RGBA", (W, H), (8, 12, 26, 255))
    for i, raw in enumerate(gen.frames()):
        if i in picks:
            layer = Image.frombytes("RGBA", (W, H), raw)
            img = Image.alpha_composite(bg, layer)
            img.save(out / picks[i])
    print("rendered 180 frames in %.1fs" % (time.time() - t0))
