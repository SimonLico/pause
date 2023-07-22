import whisper
from pydub import AudioSegment
import os
import json

FILE = "/home/al/temp/spanish1.m4a"
OUTPUT_FILE_TYPE = 'mp3'
USE_CACHED_WHISPER = True


def do_whisper(file: str) -> dict:
    def intermediate_save(whisper_result: dict):
        with open('whisper.json', 'w') as f:
            f.write(json.dumps(whisper_result))

    def intermediate_load() -> dict:
        with open('whisper.json', 'r') as f:
            return json.loads(f.read())

    if USE_CACHED_WHISPER:
        result = intermediate_load()
    else:
        model = whisper.load_model("base")
        result = model.transcribe(file, word_timestamps=True)
    intermediate_save(result)
    return result


def make_audio_segment(full_audio: AudioSegment, whisp_segment: dict):
    start = whisp_segment['start']
    end = whisp_segment['end']
    return full_audio[start * 1000:end * 1000]


def make_meta(whisp_segment: dict) -> dict:
    return whisp_segment


if __name__ == '__main__':
    whisper_result = do_whisper(FILE)
    output_dir = os.path.join(
        os.path.dirname(FILE),
        os.path.basename(FILE).split(".")[0] + "_segments"
    )
    os.makedirs(output_dir)
    audio = AudioSegment.from_file(FILE)
    for i, segment in enumerate(whisper_result['segments']):
        segment_files_dir = os.path.join(output_dir, f"segment_{i}")
        os.makedirs(segment_files_dir)
        clip_path = os.path.join(segment_files_dir, f"clip.{OUTPUT_FILE_TYPE}")
        meta_path = os.path.join(segment_files_dir, "meta.json")

        audio_segment = make_audio_segment(audio, segment)
        audio_segment.export(clip_path, format=OUTPUT_FILE_TYPE)
        with open(meta_path, 'w') as f:
            f.write(json.dumps(make_meta(segment)))


