import os

import openai
import ffmpeg
import subprocess

os.environ['OPENAI_API_KEY'] = ''
openai.api_key = os.environ['OPENAI_API_KEY']


def mp4_to_mp3(mp4_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    mp3_path = os.path.join(output_path, "audio.mp3")

    command = f'ffmpeg -i "{mp4_path}" -vn -ar 44100 -ac 2 -b:a 192k "{mp3_path}"'
    subprocess.call(command)

    try:
        subprocess.run(command, check=True)
        print(f"Audio extracted and saved to {mp3_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def summary(path):
    # todo: feed visual_summary and subtitle_summary to Llama3
    subtitle: str = subtitle_summary(path)
    visual: str = visual_summary(path)

    pass


def visual_summary(path):
    # todo: video to text (visual)
    text: str = ''
    return text


def subtitle_summary(path):
    # todo: speech to text (audio)
    client = openai.OpenAI()

    audio_file = open(path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text


if __name__ == '__main__':
    # mp4_to_mp3('test_sources/test_video/test01.mp4', 'test_sources/mp3files')
    S = subtitle_summary('test_sources/mp3files/audio.mp3')
    print(S)