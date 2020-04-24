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


def _super_path(dir_entry):
    return dir_entry.path[: -len(dir_entry.name)]


def scan_recursive(path):
    for entry in scandir(path):
        if entry.is_file():
            yield entry
        else:
            yield from scan_recursive(entry.path)


def is_media(path):
    return re.findall(r"\.(png|jpg|mp4|mkv|mov|gif)$", path.lower()) != []

def is_video(path):
    return re.findall(r"\.(mp4|mkv|mov)$", path.lower()) != []

def is_media_json(path):
    return re.findall(r"\.(png|jpg|mp4|mkv|mov|gif|mts).json$", path.lower()) != []


def duration(filename):
    if not is_video(filename):
        return 0

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
    path = _super_path(json_entry) + metadata["title"]
    path = re.sub(" ", "\\ ", path)
    path = re.sub("#", "\\#", path)
    path = re.sub("'", "\\'", path)
    path = re.sub("(\&|')", "_", path)

    return path, timestamp, duration(path)


def index_media(path):
    """ 
    Takes a path to a Google Photos archive and returns a data_frame with
    all media locations and their timestamps.
    """
    media = []
    for entry in scan_recursive(path):
        if is_media_json(entry.path):
            location, timestamp, duration = media_data(entry)
            media.append(
                {"timestamp": timestamp, "duration": duration, "location": location}
            )

    df = pd.DataFrame(media)
    df.sort_values("timestamp")
    return df

def set_target_durations(df, total_target_duration):
    t = df["timestamp"]
    target_durations = np.array(([t2-t1 for t1, t2 in zip(t[:-1], t[1:])] + [100]))
    scale_factor = np.divide(target_durations.sum(), total_target_duration)
    target_durations = np.divide(target_durations, scale_factor)

    df["target_durations"] = target_durations
    return df


def render_video(location, title, duration):
    command = """
    ffmpeg -i {} -r {} -an -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"  {}.mp4
    """.format(
        location, duration, title
    )
    os.system(command)

def date_taken(path):
    try:
        return Image.open(path)._getexif()[36867]
    except:
        return False


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    df = index_media(path)
    df = set_target_durations(df, 10)
    print(df["location"][0])
    index = 5
    render_video(df["location"][index],  "test", df["target_durations"][index])

#    main()
