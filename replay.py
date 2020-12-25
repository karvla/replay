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


PHOTO_FORMATS = set([".png", ".jpg", ".jpeg"])
VIDEO_FORMATS = set([".mov", ".mp4", ".mkv", ".gif"])
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

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


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
    Takes a path to a Google Photos archive and returns a data_frame with
    all media locations and their timestamps.
    """
    media = []
    for entry in scan_recursive(path):
        if is_media_json(Path(entry.path)):
            try:
                location, timestamp, duration = media_data(entry)
            except Exception as e:
                print(e)
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
    return df.sort_values("timestamp")


def set_target_durations(
    df: pd.DataFrame, target_length: int, rule: Callable = np.sqrt
):
    t = df["timestamp"].values
    target_durations = np.append((t[1:] - t[:-1]), [100])
    target_durations = rule(target_durations)
    target_durations = target_length / sum(target_durations) * target_durations
    df["target_length"] = target_durations
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


def make_video(frames: pd.DataFrame, size=(1920, 1080), fps=60):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output.avi", fourcc, 20.0, size)
    for n, row in tqdm(frames.iterrows(), total=len(frames)):
        target_length = row["target_length"]
        n_frames = max(1, int(fps * target_length))
        if row["video"]:
            frame_skip = int(row["duration"] * fps / (n_frames * 10))
            for n, frame in enumerate(frame_gen(row["location"])):
                if n % frame_skip != 0:
                    continue
                else:
                    frame = letterbox_image(frame, size)
                    out.write(frame)
        else:
            frame = cv2.imread(str(row["location"]))
            frame = letterbox_image(frame, size)
            [out.write(frame) for n in range(n_frames)]
    out.release()


def date_taken(path):
    try:
        return Image.open(path)._getexif()[36867]
    except:
        return False


@click.command()
@click.argument("source", type=click.Path())
@click.argument("length", type=int)
def main(**kwargs):
    df = index_media(kwargs["source"])
    df = set_target_durations(df, kwargs["length"])
    make_video(df)


if __name__ == "__main__":
    main()
