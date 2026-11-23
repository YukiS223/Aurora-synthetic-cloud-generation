"""Video I/O through ffmpeg pipes (raw RGB24)."""
import json
import subprocess
import numpy as np


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return dict(width=int(s["width"]), height=int(s["height"]),
                nb_frames=int(s["nb_frames"]), fps=float(num) / float(den))


def count_frames(path):
    """Frames that actually decode (container headers can overstate by a few)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return int(out.strip().split(",")[0])


def read_frames(path, step=1, start=0, count=None):
    """Yield (index, HxWx3 uint8 RGB) for frames start, start+step, ..."""
    info = probe(path)
    w, h = info["width"], info["height"]
    vf = []
    if step > 1 or start > 0:
        vf = ["-vf", f"select='gte(n\\,{start})*not(mod(n-{start}\\,{step}))'", "-vsync", "0"]
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)] + vf + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    # with a frame limit we close the pipe early; ffmpeg's broken-pipe complaint is then expected
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4,
                         stderr=subprocess.DEVNULL if count is not None else None)
    n = 0
    try:
        while True:
            if count is not None and n >= count:
                break
            buf = p.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            yield start + n * step, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


class Writer:
    """H.264 writer fed with raw RGB frames."""

    def __init__(self, path, w, h, fps=25, crf=14, preset="medium"):
        self.w, self.h = w, h
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
               "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        assert frame.shape == (self.h, self.w, 3) and frame.dtype == np.uint8
        self.p.stdin.write(frame.tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder failed with code {rc}")
