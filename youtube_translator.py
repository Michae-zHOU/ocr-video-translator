import sys
import os
from pytube import YouTube
import ffmpeg
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from video_subtitle_remover import (
    extract_frames_ffmpeg,
    inpaint_frames,
    reassemble_video,
)
import time


def translate_to_chinese(text):
    base_url = "https://api.mymemory.translated.net/get"
    params = {"q": text, "langpair": "en|zh"}

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()["responseData"]["translatedText"]
    else:
        raise Exception(f"Translation failed: {response.text}")


def create_srt_file(transcript, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, entry in enumerate(transcript, 1):
            start = format_time(entry["start"])
            end = format_time(entry["start"] + entry["duration"])
            text = entry["text"]
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def download_video_audio(url, download_video=True, download_audio=True):
    try:
        yt = YouTube(url)

        video_file = None
        audio_file = None

        if download_video:
            video_stream = (
                yt.streams.filter(adaptive=True, file_extension="mp4", type="video")
                .order_by("resolution")
                .desc()
                .first()
            )
            print(f"Downloading video... Resolution: {video_stream.resolution}")
            video_file = video_stream.download(filename_prefix="video_")

        if download_audio:
            audio_stream = (
                yt.streams.filter(only_audio=True).order_by("abr").desc().first()
            )
            print(f"Downloading audio... Bitrate: {audio_stream.abr}")
            audio_file = audio_stream.download(filename_prefix="audio_")

        return yt, video_file, audio_file
    except Exception as e:
        print(f"An error occurred during download: {str(e)}")
        return None, None, None


def fetch_transcript(video_id):
    try:
        print("Fetching transcript...")
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        print(f"An error occurred while fetching transcript: {str(e)}")
        return None


def translate_transcript(transcript, translate=True):
    if not translate:
        return transcript

    chinese_transcript = []
    try:
        print("Translating transcript to Chinese...")
        for entry in transcript:
            chinese_text = translate_to_chinese(entry["text"])
            chinese_transcript.append({**entry, "text": chinese_text})
            time.sleep(1)  # To avoid hitting API rate limits
    except Exception as e:
        print(f"An error occurred during translation: {str(e)}")
        return None
    return chinese_transcript


def merge_media(video_file, audio_file, srt_file, output_file):
    try:
        input_video = ffmpeg.input(video_file)
        input_audio = ffmpeg.input(audio_file)
        input_subtitles = ffmpeg.input(srt_file)

        (
            ffmpeg.output(
                input_video,
                input_audio,
                input_subtitles,
                output_file,
                vcodec="libx264",
                acodec="aac",
                **{"c:s": "mov_text", "metadata:s:s:0": "language=chi"},
                strict="experimental",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Merging complete! File saved as: {output_file}")
    except Exception as e:
        print(f"An error occurred during merging: {str(e)}")


def clean_up(*files):
    for file in files:
        if file and os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python youtube_downloader.py <YouTube URL>")
    else:
        url = sys.argv[1]

        # Flags to control each step
        DOWNLOAD_VIDEO = True
        DOWNLOAD_AUDIO = True
        REMOVE_SUBTITLES = True
        FETCH_TRANSCRIPT = True
        TRANSLATE_TRANSCRIPT = True
        CREATE_SRT = True
        MERGE_MEDIA = True

        # Existing file paths
        existing_video_file = "downloaded_video.mp4"
        existing_audio_file = "downloaded_video_audio.webm"
        existing_srt_file = "chinese_subtitles.srt"

        yt = None
        video_file = existing_video_file if not DOWNLOAD_VIDEO else None
        audio_file = existing_audio_file if not DOWNLOAD_AUDIO else None
        transcript = None

        if DOWNLOAD_VIDEO or DOWNLOAD_AUDIO:
            yt, video_file, audio_file = download_video_audio(
                url, download_video=DOWNLOAD_VIDEO, download_audio=DOWNLOAD_AUDIO
            )

        if REMOVE_SUBTITLES:
            extract_frames_ffmpeg(video_file, "frames", fps=1, duration=10)
            inpaint_frames("frames", ocr=True, debug=False, parallel=False)
            reassemble_video(video_file, "frames", fps=30)

        if FETCH_TRANSCRIPT and yt:
            transcript = fetch_transcript(yt.video_id)

        if TRANSLATE_TRANSCRIPT and transcript:
            transcript = translate_transcript(
                transcript, translate=TRANSLATE_TRANSCRIPT
            )

        srt_file = existing_srt_file if not CREATE_SRT else "chinese_subtitles.srt"
        if CREATE_SRT and transcript:
            create_srt_file(transcript, srt_file)

        output_file = f"{YouTube(url).title.replace('/', '_')}_with_subtitles.mp4"
        if MERGE_MEDIA:
            merge_media(video_file, audio_file, srt_file, output_file)

        clean_up(
            video_file if DOWNLOAD_VIDEO else None,
            audio_file if DOWNLOAD_AUDIO else None,
            srt_file if CREATE_SRT else None,
        )
