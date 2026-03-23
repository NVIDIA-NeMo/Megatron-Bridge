import argparse
import json
import os

import pandas as pd


def main():
    """Preprocess OpenVid1M dataset by filtering valid videos and generating metadata JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="OpenVidHD.csv")
    parser.add_argument("--video_dir", default="OpenVidHD_part_1")
    args = parser.parse_args()

    csv_path = args.csv_path
    video_dir = args.video_dir

    # 1. Load the CSV into a dictionary for O(1) lookup speed
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Create a mapping of filename -> caption
    # We strip any whitespace just in case
    caption_map = dict(zip(df["video"].str.strip(), df["caption"]))

    # 2. List all mp4 files in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} videos in {video_dir}. Starting conversion...")

    count = 0
    for video_name in video_files:
        if video_name in caption_map:
            # Construct the JSON content
            content = {"video": video_name, "caption": caption_map[video_name]}

            # Create the output filename (replace .mp4 with .json)
            json_name = os.path.splitext(video_name)[0] + ".json"
            json_path = os.path.join(video_dir, json_name)

            # Write the individual JSON file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4, ensure_ascii=False)

            count += 1
        else:
            print(f"Warning: No caption found for {video_name}")

    print(f"Finished! Created {count} JSON files in {video_dir}.")


if __name__ == "__main__":
    main()
