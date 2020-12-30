from scandir import scandir, walk
import os
import click
import regex as re
import ffmpeg
from PIL import Image
import pandas as pd
import json
import subprocess
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from collections.abc import Callable
import traceback

PHOTO_FORMATS = set([".png", ".jpg", ".jpeg"])
VIDEO_FORMATS = set([".mov", ".m4a", ".mp4", ".mkv", ".gif", ".m4v", ".mts", ".3gp"])
MEDIA_FORMATS = PHOTO_FORMATS.union(VIDEO_FORMATS)


def scan_recursive(path):
    for entry in scandir(path):
        if entry.is_file():
            yield entry
        else:
            yield from scan_recursive(entry.path)


def is_media(path):
    return path.suffix.lower() in MEDIA_FORMATS


def is_video(path):
    return path.suffix.lower() in VIDEO_FORMATS


def is_media_json(path):
    if len(path.suffixes) == 2:
        suf1, suf2 = path.suffixes
        return suf2.lower() == ".json" and suf1.lower() in MEDIA_FORMATS
    else:
        return False


def duration(filename):
    if not is_video(filename):
        return 0.0

    vidcapture = cv2.VideoCapture(filename.as_posix())
    fps = vidcapture.get(cv2.CAP_PROP_FPS)
    if not fps:
        return 0.0
    total_frames = vidcapture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = float(total_frames) / float(fps)
    return duration


def media_data(json_entry):
    """
    Takes a DirEntry of a .json and returns the location and timestamp of the
    corresponding file.
    """
    with open(json_entry.path) as f:
        metadata = json.load(f)

    timestamp = int(metadata["photoTakenTime"]["timestamp"])
    path = Path(json_entry.path)
    path = path.parent / path.stem
    return path, timestamp, duration(path)


def index_media(path):
    """
    Takes a path to a Google Photos archive and returns a dataframe with
    all media locations and their timestamps.
    """
    media = []
    for entry in tqdm(scan_recursive(path), desc="Indexing media"):
        if is_media_json(Path(entry.path)):
            try:
                location, timestamp, duration = media_data(entry)
            except Exception as e:
                continue
            media.append(
                {
                    "timestamp": timestamp,
                    "duration": duration,
                    "location": location,
                    "video": is_video(location),
                }
            )

    df = pd.DataFrame(media)
    return df.sort_values("timestamp").reset_index()


def set_target_durations(
    df: pd.DataFrame, target_length: int, scale_method: Callable = np.log
):
    t = df["timestamp"].values
    durations = t[1:] - t[:-1]
    durations = np.append(durations, durations[-1])
    durations += (durations == 0).astype(int)
    durations = scale_method(durations)
    durations *= target_length / sum(durations)
    df["target_length"] = durations
    return df


# https://github.com/qqwweee/keras-yolo3/issues/330#issuecomment-472462125
def letterbox_image(image, expected_size):
    ih, iw, _ = image.shape
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_img = np.full((eh, ew, 3), 0, dtype="uint8")
    new_img[
        (eh - nh) // 2 : (eh - nh) // 2 + nh, (ew - nw) // 2 : (ew - nw) // 2 + nw, :
    ] = image.copy()
    return new_img


def frame_gen(video_path):
    vidcap = cv2.VideoCapture(str(video_path))
    sucess, image = vidcap.read()
    while sucess:
        yield image
        sucess, image = vidcap.read()


@click.command()
@click.argument("source", type=click.Path())
@click.option(
    "--length",
    "-l",
    type=int,
    default=600,
    help="Target length in seconds for output video",
)
@click.option("--fps", "-r", type=int, default=60)
@click.option("--framedrop", "-f", type=int, default=2)
@click.option("--videoweight", "-v", type=int, default=10)
@click.option(
    "--scale_method", "-s", type=click.Choice(["log", "sqrt"]), default="sqrt"
)
@click.option(
    "--resolution", nargs=2, type=click.Tuple([int, int]), default=(1920, 1080)
)
def make_video(**kwargs):
    fps = kwargs["fps"]
    size = kwargs["resolution"]

    scale_method = kwargs["scale_method"]
    if scale_method == "log":
        sm = np.log
    elif scale_method == "sqrt":
        sm = np.sqrt

    df = index_media(kwargs["source"])
    media = set_target_durations(df, kwargs["length"], sm)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output.avi", fourcc, fps, size)
    media = media.dropna()
    try:
        for n, row in tqdm(media.iterrows(), total=len(media), desc="Making video"):
            target_length = row["target_length"]
            n_frames = max(1, int(fps * target_length))
            if row["video"]:
                for n, frame in enumerate(frame_gen(row["location"])):
                    if n % kwargs["framedrop"] == 0:
                        continue
                    if n > n_frames * kwargs["framedrop"] * kwargs["videoweight"]:
                        break
                    else:
                        frame = letterbox_image(frame, size)
                        out.write(frame)
            else:
                frame = cv2.imread(row["location"].as_posix())
                frame = letterbox_image(frame, size)
                [out.write(frame) for n in range(n_frames)]
    except KeyboardInterrupt:
        pass
    out.release()


def date_taken(path):
    try:
        return Image.open(path)._getexif()[36867]
    except:
        return False


if __name__ == "__main__":
    make_video()
