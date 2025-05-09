import time
import threading
import heapq

import numpy as np
import pygame
import soundfile as sf  # for reading HRTF WAV

from hrtf import get_HRTF_params, apply_hrtf
from my_constants import HRTF_DIR

# ——————————————————————————————————————————————
# CONFIG
# ——————————————————————————————————————————————
COOLDOWN       = 2.0     # secs before same obj can re-announce
DISTANCE_STEP  = 0.5     # announce again if object moves > this much closer
DISPATCH_HZ    = 20      # checks per second
SINE_LOW       = 400     # Hz for non-dangerous
SINE_HIGH      = 1500    # Hz for dangerous
DURATION       = 0.1     # secs tone length
SAMPLE_RATE    = 44100

# dangerous if label contains any of:
URGENT_KEYWORDS = {
    "car horn", "horn",
    "siren", "alarm",
    "train", "vehicle", "engine", "motor"
}

# ——————————————————————————————————————————————
# STATE
# ——————————————————————————————————————————————
_last_seen = {}    # obj_id -> (last_time, last_distance)
_queue     = []    # heap of (enqueue_time, obj_id, label, distance, x, y)
_lock      = threading.Lock()


def schedule_sound(obj_id: str,
                   label: str,
                   distance: float,
                   x_angle: float,
                   y_angle: float):
    """
    Called by your detection loop. Decides whether to enqueue a tone for this object.
    """
    now = time.time()
    prev = _last_seen.get(obj_id, (0.0, float("inf")))
    last_t, last_d = prev

    is_urgent = any(k in label.lower() for k in URGENT_KEYWORDS)
    dt = now - last_t
    dd = (last_d - distance)

    # Only re-announce if:
    #  - first time, or
    #  - it's been >= COOLDOWN secs OR it moved significantly closer
    if last_t == 0 or dt >= COOLDOWN or dd >= DISTANCE_STEP:
        _last_seen[obj_id] = (now, distance)
        with _lock:
            heapq.heappush(_queue, (now, obj_id, label, distance, x_angle, y_angle))


def _dispatcher():
    """Background thread: pops and plays one tone at a time."""
    pygame.mixer.init(frequency=SAMPLE_RATE, channels=2)
    while True:
        evt = None
        with _lock:
            if _queue:
                evt = heapq.heappop(_queue)

        if evt:
            _, obj_id, label, dist, x_ang, y_ang = evt

            # choose pitch
            freq = SINE_HIGH if any(k in label.lower() for k in URGENT_KEYWORDS) else SINE_LOW

            # generate sine
            t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
            wave = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)

            # apply HRTF
            hrtf_file, flipped = get_HRTF_params(y_ang, x_ang, HRTF_DIR)
            hrtf_data, hrtf_sr = sf.read(hrtf_file)
            proc = apply_hrtf(wave, SAMPLE_RATE, hrtf_data, hrtf_sr, flipped, dist)

            # play
            snd = pygame.sndarray.make_sound((proc * 32767).astype(np.int16))
            snd.play()
            time.sleep(DURATION)  # block until tone finishes
        else:
            # idle
            time.sleep(1.0 / DISPATCH_HZ)


# start dispatcher
threading.Thread(target=_dispatcher, daemon=True).start()
