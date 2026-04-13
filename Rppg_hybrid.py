"""
rPPG Hybrid Pipeline
====================
Combines classical rPPG signal extraction methods (POS, CHROM, G)
with deep learning models from open-rppg.

Filter strength: 0.0 = classical methods do nothing (pure DL output)
                 1.0 = classical methods fully control the final signal

Usage:
    pipeline = HybridRPPGPipeline(
        model_name='ME-chunk.rlap',
        method='POS',           # 'POS', 'CHROM', 'G', 'BLEND', or 'NONE'
        filter_strength=0.3,    # ← THE ONLY KNOB. Change 0.0 to 1.0
    )
    result = pipeline.process_video("your_video.mp4")
"""

import numpy as np
import av
import rppg
import mediapipe as mp
import pkg_resources
from scipy.signal import butter, filtfilt, welch


# ─────────────────────────────────────────────
#  CLASSICAL rPPG SIGNAL EXTRACTION METHODS
# ─────────────────────────────────────────────

def extract_rgb_trace(frames):
    """
    Extract mean R, G, B from a list of face crop frames.
    frames : list of np.ndarray (H, W, 3) uint8 RGB
    Returns: (N, 3) float64 array
    """
    rgb = []
    for f in frames:
        if f is None or f.size == 0:
            rgb.append([0., 0., 0.])
        else:
            rgb.append(f.mean(axis=(0, 1)).astype(float))
    return np.array(rgb)


def method_G(rgb):
    """Green channel — simplest rPPG baseline."""
    return rgb[:, 1].copy()


def method_POS(rgb, fps=30):
    """
    Plane-Orthogonal-to-Skin (POS).
    Wang et al. 2017.
    """
    mean  = rgb.mean(axis=0, keepdims=True) + 1e-8
    cn    = rgb / mean
    S1    = cn[:, 0] - cn[:, 1]
    S2    = cn[:, 0] + cn[:, 1] - 2 * cn[:, 2]
    alpha = (np.std(S1) + 1e-8) / (np.std(S2) + 1e-8)
    return S1 + alpha * S2


def method_CHROM(rgb, fps=30):
    """
    Chrominance-based method (CHROM).
    de Haan & Jeanne 2013.
    """
    mean  = rgb.mean(axis=0, keepdims=True) + 1e-8
    cn    = rgb / mean
    Xs    = 3 * cn[:, 0] - 2 * cn[:, 1]
    Ys    = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    alpha = (np.std(Xs) + 1e-8) / (np.std(Ys) + 1e-8)
    return Xs - alpha * Ys


def blend_classical(rgb, fps=30):
    """Equal-weight blend of G + POS + CHROM, each normalised first."""
    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    return (norm(method_G(rgb)) +
            norm(method_POS(rgb, fps)) +
            norm(method_CHROM(rgb, fps))) / 3.0


# ─────────────────────────────────────────────
#  LIGHTWEIGHT BANDPASS FILTER
# ─────────────────────────────────────────────

def light_bandpass(signal, fps=30, lowcut=0.5, highcut=3.0, order=2):
    """
    Gentle order-2 Butterworth bandpass.
    Removes DC drift and very high-frequency noise only.
    order=2 is intentionally weak — won't destroy the signal.
    """
    nyq  = fps / 2.0
    low  = lowcut  / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def normalize(signal):
    """Zero-mean, unit-variance normalisation."""
    mu, sigma = signal.mean(), signal.std()
    if sigma < 1e-8:
        return signal - mu
    return (signal - mu) / sigma


def get_hr_from_bvp(bvp, fps):
    """Estimate HR in BPM from a BVP signal using Welch's method."""
    p, q = welch(
        bvp, fps,
        nfft=1e5 / fps,
        nperseg=min(len(bvp) - 1, int(256 / 30 * fps))
    )
    mask = (p > 0.5) & (p < 3.0)   # 30-180 BPM range
    if not mask.any():
        return None
    return float(p[mask][np.argmax(q[mask])] * 60)


def _sqi(signal, sr=30, min_freq=0.5, max_freq=3.0):
    """Signal Quality Index — 0.0 (bad) to 1.0 (good)."""
    n = len(signal)
    if n < 2:
        return 0.0
    signal   = (signal - signal.mean()) / (signal.std() + 1e-8)
    autocorr = np.correlate(signal, signal, mode='full')[n-1:]
    autocorr = autocorr / (autocorr[0] + 1e-8)
    min_lag  = max(1, int(sr / max_freq))
    max_lag  = min(len(autocorr) - 1, int(sr / min_freq))
    if min_lag >= max_lag:
        return 0.0
    return float(np.clip(np.max(autocorr[min_lag:max_lag + 1]), 0.0, 1.0))


# ─────────────────────────────────────────────
#  HYBRID PIPELINE
# ─────────────────────────────────────────────

class HybridRPPGPipeline:
    """
    Parameters
    ----------
    model_name : str
        Any model from rppg.supported_models.
        Default: 'ME-chunk.rlap'

    method : str
        Classical signal extraction method.
        'POS'   - Plane-Orthogonal-to-Skin
        'CHROM' - Chrominance method
        'G'     - Green channel only
        'BLEND' - Equal mix of POS + CHROM + G
        'NONE'  - Skip classical entirely, pure DL

    filter_strength : float  [0.0 to 1.0]
        0.0  ->  pure DL  (classical does nothing)
        0.5  ->  50% DL + 50% classical
        1.0  ->  pure classical
    """

    SUPPORTED_METHODS = ('POS', 'CHROM', 'G', 'BLEND', 'NONE')

    def __init__(
        self,
        model_name:      str   = 'ME-chunk.rlap',
        method:          str   = 'POS',
        filter_strength: float = 0.3,
    ):
        if model_name not in rppg.supported_models:
            raise ValueError(
                f"Unknown model '{model_name}'.\n"
                f"Supported: {rppg.supported_models}"
            )
        if method.upper() not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'.\n"
                f"Supported: {self.SUPPORTED_METHODS}"
            )
        if not (0.0 <= filter_strength <= 1.0):
            raise ValueError("filter_strength must be between 0.0 and 1.0")

        self.model_name      = model_name
        self.method          = method.upper()
        self.filter_strength = filter_strength  # <- THE THRESHOLD
        self.model           = rppg.Model(model_name)

        print(f"[HybridRPPG] Model           : {model_name}")
        print(f"[HybridRPPG] Classical method : {self.method}")
        print(f"[HybridRPPG] Filter strength  : {filter_strength}  "
              f"(0.0=pure DL, 1.0=pure classical)")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _decode_video(self, video_path):
        """
        Decode video ONCE and return frames + fps + timestamps.
        Both the DL path and classical path share these frames,
        so the video file is only read a single time.

        Returns
        -------
        frames     : list of np.ndarray (H, W, 3) uint8 RGB
        timestamps : list of float (seconds)
        fps        : float — actual fps read from the container
        """
        container  = av.open(video_path)
        stream     = container.streams.video[0]
        stream.thread_type = 'AUTO'

        # Read actual fps from container (fixes hardcoded 30fps bug)
        fps = float(stream.average_rate or 30.0)

        frames, timestamps = [], []
        for frame in container.decode(stream):
            rotation = -frame.rotation % 360
            img      = frame.to_ndarray(format='rgb24')
            # Handle rotated videos (portrait phone recordings)
            if rotation == 90:
                img = img.swapaxes(0, 1)[:, ::-1, :]
            elif rotation == 180:
                img = img[::-1, ::-1, :]
            elif rotation == 270:
                img = img.swapaxes(0, 1)[::-1, :, :]
            frames.append(img)
            timestamps.append(frame.time)

        container.close()
        return frames, timestamps, fps

    def _detect_faces(self, frames, timestamps):
        """
        Run MediaPipe face detection on pre-decoded frames.
        Returns list of face crops (RGB np.ndarray or None).
        Reuses the same tflite model bundled with open-rppg.
        """
        BaseOptions      = mp.tasks.BaseOptions
        FaceDetector     = mp.tasks.vision.FaceDetector
        FaceDetectorOpts = mp.tasks.vision.FaceDetectorOptions
        VisionMode       = mp.tasks.vision.RunningMode

        model_asset = pkg_resources.resource_filename(
            'rppg', 'weights/blaze_face_short_range.tflite'
        )
        options = FaceDetectorOpts(
            base_options=BaseOptions(model_asset_path=model_asset),
            running_mode=VisionMode.VIDEO,
        )

        faces    = []
        last_box = None

        with FaceDetector.create_from_options(options) as detector:
            for img, ts in zip(frames, timestamps):
                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(img)
                )
                result = detector.detect_for_video(mp_img, round(ts * 1e6))

                if result.detections:
                    b        = result.detections[0].bounding_box
                    y0       = max(0, b.origin_y - round(b.height * 0.2))
                    y1       = b.origin_y + round(b.height * 0.9)
                    x0       = b.origin_x
                    x1       = b.width + b.origin_x
                    last_box = (y0, y1, x0, x1)

                if last_box:
                    y0, y1, x0, x1 = last_box
                    face = img[y0:y1, x0:x1]
                    faces.append(face if face.size > 0 else None)
                else:
                    faces.append(None)

        return faces

    def _classical_signal(self, faces, fps):
        """Run the chosen classical method on face crops."""
        rgb = extract_rgb_trace(faces)
        if self.method == 'POS':
            return method_POS(rgb, fps)
        elif self.method == 'CHROM':
            return method_CHROM(rgb, fps)
        elif self.method == 'G':
            return method_G(rgb)
        elif self.method == 'BLEND':
            return blend_classical(rgb, fps)
        else:  # 'NONE'
            return None

    def _blend(self, dl_bvp, classical_bvp):
        """
        Blend DL and classical BVP signals.

        filter_strength = 0.0 -> 100% DL,   0% classical
        filter_strength = 0.5 ->  50% DL,  50% classical
        filter_strength = 1.0 ->   0% DL, 100% classical
        """
        if classical_bvp is None or self.filter_strength == 0.0:
            return dl_bvp

        n   = min(len(dl_bvp), len(classical_bvp))
        dl  = normalize(dl_bvp[:n])
        cls = normalize(classical_bvp[:n])

        # ── THE THRESHOLD LINE ──────────────────────────────────
        alpha   = self.filter_strength        # classical weight
        beta    = 1.0 - self.filter_strength  # DL weight
        blended = beta * dl + alpha * cls
        # ────────────────────────────────────────────────────────

        return normalize(blended)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process_video(self, video_path: str) -> dict:
        """
        Process a video file and return HR + HRV metrics.

        Returns the standard open-rppg dict PLUS:
            'dl_hr'           - HR from DL model before blending
            'classical_hr'    - HR from classical method alone
            'filter_strength' - the threshold value used
            'method'          - which classical method was used
            'model'           - which DL model was used
        """
        print(f"[HybridRPPG] Processing: {video_path}")

        # ── Step 1: Decode video ONCE ──────────────────────────────────
        print("[HybridRPPG] Decoding video...")
        frames, timestamps, fps = self._decode_video(video_path)
        print(f"[HybridRPPG] {len(frames)} frames at {fps:.1f} fps")

        # ── Step 2: Run DL model on decoded frames ─────────────────────
        print("[HybridRPPG] Running DL model...")
        dl_bvp = np.array([])

        with self.model:
            for img, ts in zip(frames, timestamps):
                self.model.update_frame(img, ts)
            # Collect BVP while context manager is still alive
            if self.model.has_signal:
                dl_bvp, _ = self.model.bvp()

        dl_hr_raw = get_hr_from_bvp(dl_bvp, fps) if len(dl_bvp) > 10 else None

        # ── Step 3: Run classical method on same frames ────────────────
        classical_bvp = None
        classical_hr  = None

        if self.method != 'NONE' and self.filter_strength > 0.0:
            print(f"[HybridRPPG] Running classical method: {self.method}")
            faces = self._detect_faces(frames, timestamps)

            if any(f is not None for f in faces):
                classical_bvp = self._classical_signal(faces, fps)

                # Very gentle bandpass on classical signal only
                if len(classical_bvp) > 10:
                    classical_bvp = light_bandpass(
                        classical_bvp, fps=fps,
                        lowcut=0.5, highcut=3.0, order=2
                    )
                classical_hr = get_hr_from_bvp(classical_bvp, fps)

        # ── Step 4: Blend ──────────────────────────────────────────────
        if len(dl_bvp) > 0 and classical_bvp is not None:
            final_bvp = self._blend(dl_bvp, classical_bvp)
            final_hr  = get_hr_from_bvp(final_bvp, fps) or dl_hr_raw
        else:
            final_bvp = dl_bvp
            final_hr  = dl_hr_raw

        # ── Step 5: Compute HRV and SQI on final BVP ──────────────────
        hrv = {}
        sqi = 0.0
        if len(final_bvp) > 10:
            sqi = _sqi(final_bvp, fps)
            try:
                import heartpy as hp
                m, n = hp.process(final_bvp, fps,
                                  high_precision=True, clean_rr=True)
                hrv  = n
            except Exception:
                pass

        return {
            'hr':              final_hr,
            'SQI':             sqi,
            'hrv':             hrv,
            'latency':         0.0,
            'dl_hr':           dl_hr_raw,
            'classical_hr':    classical_hr,
            'filter_strength': self.filter_strength,
            'method':          self.method,
            'model':           self.model_name,
        }

    # ------------------------------------------------------------------
    #  Runtime controls
    # ------------------------------------------------------------------

    def set_filter_strength(self, value: float):
        """
        Change the blend threshold on the fly.
        0.0 = pure DL output
        1.0 = pure classical output
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError("filter_strength must be between 0.0 and 1.0")
        self.filter_strength = value
        print(f"[HybridRPPG] filter_strength -> {value}")

    def set_method(self, method: str):
        """Switch classical method without rebuilding."""
        if method.upper() not in self.SUPPORTED_METHODS:
            raise ValueError(f"Supported: {self.SUPPORTED_METHODS}")
        self.method = method.upper()
        print(f"[HybridRPPG] method -> {self.method}")

    def set_model(self, model_name: str):
        """Hot-swap the DL model."""
        if model_name not in rppg.supported_models:
            raise ValueError(f"Supported: {rppg.supported_models}")
        self.model_name = model_name
        self.model      = rppg.Model(model_name)
        print(f"[HybridRPPG] model -> {self.model_name}")


# ─────────────────────────────────────────────
#  USAGE EXAMPLES
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # Basic usage
    pipeline = HybridRPPGPipeline(
        model_name='ME-chunk.rlap',
        method='POS',
        filter_strength=0.3,    # <- CHANGE THIS: 0.0 to 1.0
    )
    result = pipeline.process_video('your_video.mp4')
    print(result)

    # Change threshold without rebuilding
    pipeline.set_filter_strength(0.1)
    result = pipeline.process_video('your_video.mp4')

    # Sweep to find best threshold for your video
    for strength in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        pipeline.set_filter_strength(strength)
        r = pipeline.process_video('your_video.mp4')
        print(f"strength={strength:.1f}  "
              f"final_HR={r['hr']:.1f}  "
              f"dl_HR={r['dl_hr']:.1f}  "
              f"classical_HR={r['classical_hr']}  "
              f"SQI={r['SQI']:.3f}")

    # Try every model, pick best SQI
    best, best_sqi = None, -1
    for name in rppg.supported_models:
        try:
            p = HybridRPPGPipeline(name, method='BLEND', filter_strength=0.2)
            r = p.process_video('your_video.mp4')
            print(f"{name:25s}  HR={r['hr']:.1f}  SQI={r['SQI']:.3f}")
            if r['SQI'] > best_sqi:
                best_sqi, best = r['SQI'], r
        except Exception as e:
            print(f"{name}: failed - {e}")
    print("\nBest:", best)

    # Real-time webcam (DL only recommended for live mode)
    import time
    pipeline = HybridRPPGPipeline('ME-flow.rlap', method='NONE',
                                   filter_strength=0.0)
    with pipeline.model.video_capture(0):
        while True:
            result = pipeline.model.hr(start=-15)
            if result:
                print(f"HR: {result['hr']:.1f} BPM  SQI: {result['SQI']:.2f}")
            time.sleep(1)
