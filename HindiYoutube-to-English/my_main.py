#!/usr/bin/env python3

# https://www.thepythoncode.com/article/using-speech-recognition-to-convert-speech-to-text-python
from __future__ import unicode_literals
import speech_recognition as sr
import os, sys
from pydub import AudioSegment
from pydub.silence import split_on_silence
from my_translator import hindiToEnglish

audioname = "voice.wav"
r = sr.Recognizer()

try:
    link = sys.argv[1]
except:
    link = ''
sys.stdout = open('output.txt', 'a')


def transcript(audioname=audioname):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    if not audioname:
        return
    # open the audio file using pydub
    print('started to transcript')
    sound = AudioSegment.from_file(audioname)
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened, language='hi-In')
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
            except sr.UnknownValueError as e:
                print("Error:", str(e))
    os.system("rm -rf voice.* audio-chunks")
    return whole_text


# https://stackoverflow.com/questions/27473526/download-only-audio-from-youtube-video-using-youtube-dl-in-python-script
# from __future__ import unicode_literals  # written at first
import youtube_dl
ydl_opts = {
    'format': 'bestaudio',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
	    'preferredquality': '192',
    }],
    'outtmpl': audioname,
}

def download_video(link=""):
    if not link:
        return False
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    # link = link.lstrip.('https://www.youtube.com/watch?v=')
    # if len(link) != 11:
        # return False
    # os.system('youtube-dl -f "bestaudio[ext=m4a]" '+link)
    return True


def findFilename():
    for file in os.listdir("./"):
        if file.startswith("voice"):
            return file
    return ''


def videoToEnglish(link=''):
    print(link + '\n\n', file=open('last_translated.txt', 'w'))
    download_video(link)
    audioname = findFilename()
    whole_text = transcript(audioname)
    # whole_text = hindiToEnglish(whole_text)
    print(whole_text, file=open('last_translated.txt', 'a'))
    return whole_text


if __name__ == '__main__':
    videoToEnglish(link)
