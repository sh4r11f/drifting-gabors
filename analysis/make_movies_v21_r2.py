"""make_movies_v21_r2.py: Movies 1 and 2 for the single-drift / double-drift paper, revision 2 (2026-10-20).

Revision 2 (Sharif, 20 Oct): carrier spatial frequency 0.55 cycles/dva (was 0.44) and internal drift 4 Hz
(7.3 dva/s, was 5 Hz / 11.25 dva/s), so the movies show slightly finer stripes drifting slightly slower than
in the experiment; labels larger (40 px) and white with a dark outline. Everything else as below.


Re-renders the stimuli of the experiment (Submission/experiment/src/double_drift.py, config/parameters.json)
frame by frame with numpy, because the original PsychoPy + EyeLink program needs the lab hardware.
Parameters are the paper's: Gabor envelope sigma 0.3 dva drawn to +/-3 sigma (1.8 dva), carrier
0.44 cycles/dva (0.8 cycles per 1.8-dva patch, the value in the manuscript), 80% Michelson contrast,
internal drift 5 Hz (11.25 dva/s), external drift 2.5 dva/s, patch 10.8 dva from fixation at 25 deg
above the horizontal (upper right quadrant), black fixation disc 0.2 dva, mid-grey background, black
response line rising from the fixation point (7 px at 45.2 px/dva on the VG279QM = 0.155 dva).
Each pass lasts the path duration and then reverses, as in _move_gabor(); one stabilisation cycle
precedes the response period. The adjusted values shown are the group means at 800 ms (v16 data):
illusory angle 68 deg from vertical; matched length 1.11 dva (single-drift) and 1.26 dva (double-drift,
= 1.17 dva offset / sin 68 deg).

Usage: python make_movies_v21.py <out_dir>
"""
import os, sys, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
FPS = 60; PPD = 80.0                      # video frame rate, pixels per degree in the movie
W, H = 1280, 720
X0, Y0 = -3.0, 7.6                        # dva at the top-left corner (y up)
SIG, HALF = 0.3, 0.9                      # envelope sd and half-size (dva)
SF, TF, CONTRAST = 0.55, 4.0, 0.8           # rev 2: experiment used 0.8/1.8 c/dva and 5 Hz
V_ENV = 2.5
ECC, POLAR = 10.8, 25.0
CENTER = np.array([ECC * np.cos(np.radians(POLAR)), ECC * np.sin(np.radians(POLAR))])
BG = 0.5
LINE_W = 0.155
FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)

def to_px(x, y): return (x - X0) * PPD, (Y0 - y) * PPD

# a background frame with the fixation disc
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
XD, YD = X0 + (xx + 0.5) / PPD, Y0 - (yy + 0.5) / PPD           # dva coordinates of every pixel
base = np.full((H, W), BG, np.float32)
fix = np.hypot(XD, YD) <= 0.1
r = np.hypot(XD, YD); base = np.where(r <= 0.1, 0.0, base)
# antialias the fixation edge
edge = (r > 0.1) & (r < 0.1 + 1.5 / PPD); base[edge] = BG * ((r[edge] - 0.1) * PPD / 1.5)

def gabor_into(img, pos, bars_dir, phase):
    """bars_dir: unit vector along the bars; phase in cycles (carrier moves along the normal)."""
    n = np.array([-bars_dir[1], bars_dir[0]])
    c0, r0 = to_px(pos[0] - HALF, pos[1] + HALF)
    c1, r1 = to_px(pos[0] + HALF, pos[1] - HALF)
    c0, r0, c1, r1 = int(np.floor(c0)), int(np.floor(r0)), int(np.ceil(c1)), int(np.ceil(r1))
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, W), min(r1, H)
    dx, dy = XD[r0:r1, c0:c1] - pos[0], YD[r0:r1, c0:c1] - pos[1]
    env = np.exp(-(dx ** 2 + dy ** 2) / (2 * SIG ** 2)) * ((np.abs(dx) <= HALF) & (np.abs(dy) <= HALF))
    car = np.sin(2 * np.pi * (SF * (dx * n[0] + dy * n[1]) - phase))
    img[r0:r1, c0:c1] = BG * (1 + CONTRAST * env * car)

def line_into(img, length):
    m = (np.abs(XD) <= LINE_W / 2) & (YD >= 0) & (YD <= length)
    img[m] = 0.0

class Writer:
    def __init__(self, path):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                                   "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        self.stills = {}
    def put(self, img, label=None, still=None):
        g = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        im = Image.fromarray(np.stack([g] * 3, -1))
        if label:
            ImageDraw.Draw(im).text((28, H - 68), label, font=FONT, fill=(255, 255, 255), stroke_width=3, stroke_fill=(30, 30, 30))
        if still: self.stills[still] = im.copy()
        self.p.stdin.write(im.tobytes())
    def close(self):
        self.p.stdin.close(); assert self.p.wait() == 0

def ease(u): u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)

def run(path, dur_ms, kind, segments):
    """segments: list of (label, seconds, angle_fn(t), line_fn(t) or None). Motion state continues across segments."""
    n_pass = int(round(dur_ms / 1000 * FPS))
    w = Writer(path); phase = 0.0; s = 0.0; k = 0
    for label, secs, ang_fn, line_fn, still in segments:
        nf = int(round(secs * FPS))
        for i in range(nf):
            t = i / FPS
            img = base.copy()
            if ang_fn is not None:
                direction = 1 if (k % (2 * n_pass)) < n_pass else -1
                theta = ang_fn(t)                                   # path angle from vertical (deg); None for single-drift
                if kind == "dd":
                    a = np.radians(90 - theta)                       # from horizontal
                    u = np.array([np.cos(a), np.sin(a)])
                    pos = CENTER + u * s
                    gabor_into(img, pos, u, phase)
                    s += direction * V_ENV / FPS
                else:
                    gabor_into(img, CENTER, np.array([1.0, 0.0]), phase)
                phase += direction * TF / FPS
                k += 1
            if line_fn is not None: line_into(img, line_fn(t))
            w.put(img, label, still if (still and i == nf - 1) else None)
    w.close(); return w.stills

THETA, SD_LEN, DD_LEN = 68.0, 1.11, 1.17 / np.sin(np.radians(68.0))
stills = {}
# Movie 1: single-drift, length matching, 800-ms passes
stills.update(run(os.path.join(OUT, "Movie1_single_drift.mp4"), 800, "sd", [
    (None, 0.6, None, None, None),
    ("Single-drift: 800-ms passes", 1.6, lambda t: 0, None, None),
    ("Length matching", 6.4, lambda t: 0, lambda t: 0.3 + (SD_LEN - 0.3) * ease((t - 0.8) / 4.4), None),
    ("Length matching", 3.2, lambda t: 0, lambda t: SD_LEN, "movie1_still"),
]))
# Movie 2: double-drift, orientation adjustment then length matching
stills.update(run(os.path.join(OUT, "Movie2_double_drift.mp4"), 800, "dd", [
    (None, 0.6, None, None, None),
    ("Double-drift: 800-ms passes", 1.6, lambda t: 45.0, None, "movie2_start"),
    ("Orientation adjustment", 6.4, lambda t: 45.0 + (THETA - 45.0) * ease((t - 0.8) / 4.8), None, None),
    ("Orientation adjustment", 3.2, lambda t: THETA, None, "movie2_angle"),
    (None, 0.6, None, None, None),
    ("Length matching", 1.6, lambda t: THETA, None, None),
    ("Length matching", 6.4, lambda t: THETA, lambda t: 0.3 + (DD_LEN - 0.3) * ease((t - 0.8) / 4.4), None),
    ("Length matching", 3.2, lambda t: THETA, lambda t: DD_LEN, "movie2_length"),
]))
for k, im in stills.items(): im.save(os.path.join(OUT, k + ".png"))
print("done", sorted(os.listdir(OUT)))
