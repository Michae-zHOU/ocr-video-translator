import os
import cv2
import argparse
import logging
import numpy as np
import easyocr
import keras_ocr
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def initialize_ocr_pipeline(ocr_method):
    if ocr_method == "easyocr":
        return easyocr.Reader(["en"])
    elif ocr_method == "keras":
        return keras_ocr.pipeline.Pipeline()
    elif ocr_method == "macocr":
        # Placeholder for macOCR initialization if needed
        pass
    return None


def extract_frames_ffmpeg(video_path, frames_folder, fps, duration):
    os.makedirs(frames_folder, exist_ok=True)
    command = f"ffmpeg -i {video_path} -vf fps={fps} -t {duration} {frames_folder}/frame-%04d.png"
    subprocess.run(command, shell=True)


def remove_subtitles(frame, ocr_method, ocr_pipeline, debug=False):
    mask = detect_text_ocr(frame, ocr_method, ocr_pipeline)
    inpainted_frame = cv2.inpaint(frame, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return inpainted_frame, mask


def detect_text_ocr(frame, ocr_method, ocr_pipeline):
    if ocr_method == "easyocr":
        result = ocr_pipeline.readtext(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for bbox, text, prob in result:
            points = np.array(bbox).astype(np.int32)
            cv2.fillPoly(mask, [points], (255))
        return mask
    elif ocr_method == "keras":
        predictions = ocr_pipeline.recognize([frame])
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for prediction in predictions[0]:
            points = prediction[1].astype(np.int32)
            cv2.fillPoly(mask, [points], (255))
        return mask


def process_frame(args):
    frame_file, output_file, ocr_method, debug = args
    frame = cv2.imread(frame_file)
    ocr_pipeline = initialize_ocr_pipeline(ocr_method)
    inpainted_frame, mask = remove_subtitles(
        frame, ocr_method, ocr_pipeline, debug=debug
    )
    if debug:
        os.makedirs("debug_frames", exist_ok=True)
        debug_frame_file = os.path.join("debug_frames", os.path.basename(frame_file))
        cv2.imwrite(debug_frame_file, inpainted_frame)
        debug_mask_file = os.path.join(
            "debug_frames", os.path.basename(frame_file).replace(".png", "_mask.png")
        )
        cv2.imwrite(debug_mask_file, mask)
    cv2.imwrite(output_file, inpainted_frame)
    logging.debug(f"Processed frame: {frame_file}")


def inpaint_frames(frames_folder, ocr_method, debug, parallel):
    frame_files = sorted(
        [
            os.path.join(frames_folder, frame)
            for frame in os.listdir(frames_folder)
            if frame.endswith(".png") and not frame.startswith("inpainted_")
        ]
    )
    output_files = [
        os.path.join(frames_folder, f"inpainted_{os.path.basename(frame)}")
        for frame in frame_files
    ]
    args = [
        (frame_files[i], output_files[i], ocr_method, debug)
        for i in range(len(frame_files))
    ]

    if parallel:
        with Pool(max(1, cpu_count() // 2)) as pool:
            for _ in tqdm(pool.imap_unordered(process_frame, args), total=len(args)):
                pass
    else:
        for arg in tqdm(args):
            process_frame(arg)


def reassemble_video(output_video_path, frames_folder, fps):
    inpainted_files = sorted(
        [
            os.path.join(frames_folder, frame)
            for frame in os.listdir(frames_folder)
            if frame.startswith("inpainted_")
        ]
    )
    if not inpainted_files:
        raise FileNotFoundError("No inpainted frames found to reassemble the video.")
    frame = cv2.imread(inpainted_files[0])
    height, width, layers = frame.shape
    logging.info(f"Video dimensions: width={width}, height={height}")

    # Try different codecs
    codecs = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
    for codec in codecs:
        video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if video.isOpened():
            break
        else:
            logging.error(f"Failed to open video writer with codec: {codec}")

    if not video.isOpened():
        logging.error("Failed to open video writer with all tested codecs.")
        return

    for frame_file in inpainted_files:
        try:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video.write(frame)
                logging.info(f"Writing frame: {frame_file}")
            else:
                logging.error(f"Frame is None: {frame_file}")
        except Exception as e:
            logging.error(f"Error writing frame: {frame_file}")
            logging.error(e)
    video.release()

    # Verify that the video was created
    if os.path.exists(output_video_path):
        logging.info(f"Video successfully saved to: {output_video_path}")
    else:
        logging.error(f"Failed to create the video: {output_video_path}")


def main(args):
    steps = ["extract", "inpaint", "reassemble"]
    start_index = steps.index(args.start_step)

    if "extract" in steps[start_index:]:
        extract_frames_ffmpeg(
            args.input_video, args.frames_folder, args.fps, args.duration
        )
    if "inpaint" in steps[start_index:]:
        inpaint_frames(args.frames_folder, args.ocr, args.debug, not args.no_parallel)
    if "reassemble" in steps[start_index:]:
        reassemble_video(args.output_video, args.frames_folder, args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove subtitles from video")
    parser.add_argument("input_video", type=str, help="Path to the input video")
    parser.add_argument("output_video", type=str, help="Path to the output video")
    parser.add_argument(
        "--ocr",
        type=str,
        choices=["easyocr", "keras", "macocr"],
        help="OCR method to use",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--duration", type=int, default=10, help="Duration in seconds to process"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--no_parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--frames_folder",
        type=str,
        default="frames",
        help="Folder to store extracted frames",
    )
    parser.add_argument(
        "--debug_folder",
        type=str,
        default="debug_frames",
        help="Folder to store debug frames",
    )
    parser.add_argument(
        "--start_step",
        type=str,
        choices=["extract", "inpaint", "reassemble"],
        default="extract",
        help="Step to start processing from",
    )

    args = parser.parse_args()
    main(args)
