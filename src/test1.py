import logging
import os
import sys
import traceback
from contextlib import contextmanager
from typing import Optional

import diart.operators as dops
import numpy as np
import rich
import rx.operators as ops
import sounddevice as sd
import whisper_timestamped as whisper
from diart import OnlineSpeakerDiarization, PipelineConfig
from diart.sources import MicrophoneAudioSource
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment

Model = 'tiny'  # Whisper model size (tiny, base, small, medium, large)
def concat(chunks, collar=0.05):
    """
    Concatenate predictions and audio
    given a list of `(diarization, waveform)` pairs
    and merge contiguous single-speaker regions
    with pauses shorter than `collar` seconds.
    """
    first_annotation = chunks[0][0]
    first_waveform = chunks[0][1]
    annotation = Annotation(uri=first_annotation.uri)
    data = []
    for ann, wav in chunks:
        annotation.update(ann)
        data.append(wav.data)
    annotation = annotation.support(collar)
    window = SlidingWindow(
        first_waveform.sliding_window.duration,
        first_waveform.sliding_window.step,
        first_waveform.sliding_window.start,
    )
    data = np.concatenate(data, axis=0)
    return annotation, SlidingWindowFeature(data, window)


def colorize_transcription(transcription):
    """
    Unify a speaker-aware transcription represented as
    a list of `(speaker: int, text: str)` pairs
    into a single text colored by speakers.
    """
    colors = 2 * [
        "bright_red", "bright_blue", "bright_green", "orange3", "deep_pink1",
        "yellow2", "magenta", "cyan", "bright_magenta", "dodger_blue2"
    ]
    result = []
    for speaker, text in transcription:
        if speaker == -1:
            # No speakerfound for this text, use default terminal color
            result.append(text)
        else:
            result.append(f"[{colors[speaker]}]{text}")
    return "\n".join(result)

@contextmanager
def suppress_stdout():
    # Auxiliary function to suppress Whisper logs (it is quite verbose)
    # All credit goes to: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class WhisperTranscriber:
    def __init__(self, model="small", device=None):
        self.model = whisper.load_model(model, device=device)
        self._buffer = ""

    def transcribe(self, waveform):
        """Transcribe audio using Whisper"""
        # Pad/trim audio to fit 30 seconds as required by Whisper
        audio = waveform.data.astype("float32").reshape(-1)
        audio = whisper.pad_or_trim(audio)

        # Transcribe the given audio while suppressing logs
        with suppress_stdout():
            transcription = whisper.transcribe(
                self.model,
                audio,
                # We use past transcriptions to condition the model
                initial_prompt=self._buffer,
                verbose=True  # to avoid progress bar
            )

        return transcription

    def identify_speakers(self, transcription, diarization, time_shift):
        """Iterate over transcription segments to assign speakers"""
        speaker_captions = []
        for segment in transcription["segments"]:

            # Crop diarization to the segment timestamps
            start = time_shift + segment["words"][0]["start"]
            end = time_shift + segment["words"][-1]["end"]
            dia = diarization.crop(Segment(start, end))

            # Assign a speaker to the segment based on diarization
            speakers = dia.labels()
            num_speakers = len(speakers)
            if num_speakers == 0:
                # No speakers were detected
                caption = (-1, segment["text"])
            elif num_speakers == 1:
                # Only one speaker is active in this segment
                spk_id = int(speakers[0].split("speaker")[1])
                caption = (spk_id, segment["text"])
            else:
                # Multiple speakers, select the one that speaks the most
                max_speaker = int(np.argmax([
                    dia.label_duration(spk) for spk in speakers
                ]))
                caption = (max_speaker, segment["text"])
            speaker_captions.append(caption)

        return speaker_captions

    def __call__(self, diarization, waveform):
        # Step 1: Transcribe
        transcription = self.transcribe(waveform)
        # Update transcription buffer
        self._buffer += transcription["text"]
        # The audio may not be the beginning of the conversation
        time_shift = waveform.sliding_window.start
        # Step 2: Assign speakers
        speaker_transcriptions = self.identify_speakers(transcription, diarization, time_shift)
        return speaker_transcriptions


def start(self, input_device_index: Optional[int], sample_rate: int) -> None:
    # Suppress whisper-timestamped warnings for a clean output
    # logging.getLogger("whisper_timestamped").setLevel(logging.ERROR)
    # If you have a GPU, you can also set device=torch.device("cuda")
    config = PipelineConfig(
        device=input_device_index,
        sample_rate=sample_rate,
        duration=5,
        step=0.5,
        latency="min",
        tau_active=0.5,# 只识别说话概率高于50%的发言者。
        rho_update=0.1,# Diart自动收集演讲者的信息来改进自己（别担心，这是本地完成的，不会与任何人分享）。在这里，我们只使用每个演讲者超过100ms的语音进行自我改进。
        delta_new=0.57 # 这是一个介于0和2之间的内部阈值，用于调节新的扬声器检测。该值越低，系统对声音的差异就越敏感
    )
    dia = OnlineSpeakerDiarization(config)
    source = MicrophoneAudioSource(sample_rate=config.sample_rate,device=config.device)
    # If you have a GPU, you can also set device="cuda"
    asr = WhisperTranscriber(model=Model)
    # Split the stream into 2s chunks for transcription
    transcription_duration = 2
    # Apply models in batches for better efficiency
    batch_size = int(transcription_duration // config.step)
    # Chain of operations to apply on the stream of microphone audio
    source.stream.pipe(
        # Format audio stream to sliding windows of 5s with a step of 500ms
        dops.rearrange_audio_stream(
            config.duration, config.step, config.sample_rate
        ),
        # Wait until a batch is full
        # The output is a list of audio chunks
        ops.buffer_with_count(count=batch_size),
        # Obtain diarization prediction
        # The output is a list of pairs `(diarization, audio chunk)`
        ops.map(dia),
        # Concatenate 500ms predictions/chunks to form a single 2s chunk
        ops.map(concat),
        # Ignore this chunk if it does not contain speech
        ops.filter(lambda ann_wav: ann_wav[0].get_timeline().duration() > 0),
        # Obtain speaker-aware transcriptions
        # The output is a list of pairs `(speaker: int, caption: str)`
        ops.starmap(asr),
        # Color transcriptions according to the speaker
        # The output is plain text with color references for rich
        ops.map(colorize_transcription),
    ).subscribe(
        on_next=rich.print,  # print colored text
        on_error=lambda _: traceback.print_exc()  # print stacktrace if error
    )
    print("Listening...")
    source.read()


def get_device_sample_rate(device_id: Optional[int]) -> int:
    """Returns the sample rate to be used for recording. It uses the default sample rate
    provided by Whisper if the microphone supports it, or else it uses the device's default
    sample rate.
    """
    whisper_sample_rate = whisper.audio.SAMPLE_RATE
    try:
        sd.check_input_settings(
            device=device_id, samplerate=whisper_sample_rate)
        return whisper_sample_rate
    except sd.PortAudioError:
        device_info = sd.query_devices(device=device_id)
        if isinstance(device_info, dict):
            return int(device_info.get('default_samplerate', whisper_sample_rate))
        return whisper_sample_rate


def checkDevice():
    # 获取所有可用的音频设备信息
    print(sd.query_devices())
    # 获取当前默认输入设备
    print("当前默认【输入，输出】设备为："+sd.default.device)



def main():
    try:
        device_id = sd.default.device[0]
        # 接受用户输入的设备序号
        device_id = int(input("请输入设备序号："))
        device_sample_rate = get_device_sample_rate(device_id)
        start(input_device_index=device_id, sample_rate=device_sample_rate)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n\033[93mQuitting..\033[0m")
        if os.path.exists('dictate.wav'): os.remove('dictate.wav')


if __name__ == '__main__':
    checkDevice()
    main()  # by Nik
