#!/usr/bin/env python3

import argparse
import queue
import sys
import sounddevice as sd
import json

from src.k9_loop import K9_Loop
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-l", "--list-devices", action="store_true",        help="show list of audio devices and exit")
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])

    parser.add_argument("-d", "--device",       type=int_or_str,                     help="input device (numeric ID or substring)")
    parser.add_argument("-r", "--samplerate",   type=int, default=22050,             help="sampling rate")
    parser.add_argument("-m", "--model",        type=str, default="en-us",           help="speech rec model; e.g. en-us, fr, nl; default is en-us")
    parser.add_argument("-b", "--blocksize",    type=int, default=2000,              help="blocksize")
    parser.add_argument("-t", "--tts_model",    type=str, default="./k9_model.onnx", help="tts model")
    parser.add_argument("-o", "--ollama_model", type=str, default="llama3.1:8b",     help="ollama model")
    args = parser.parse_args(remaining)
    
    print("Using samplerate: " + str(args.samplerate))
    print("Using blocksize: "  + str(args.blocksize))
    
    try:
    
        istream = sd.RawInputStream(samplerate=args.samplerate, blocksize=args.blocksize, device=args.device, dtype="int16", channels=1, callback=callback)

        # TODO: the sample rate actually comes from the tts model and is currently 22050
        ostream = sd.RawOutputStream(samplerate=args.samplerate, channels=1, dtype='int16')

        k9_loop = K9_Loop(q, istream, ostream)
        k9_loop.load_ollama_model(args.ollama_model)
    
        k9_loop.load_rec_model(args.model, args.samplerate)
        fs = k9_loop.load_tts_model(args.tts_model, args.blocksize)

        k9_loop.say("hello, master")

        with istream:
            print("#" * 80)
            print("Press Ctrl+C to stop the recording")
            print("#" * 80)

            k9_loop.run()

    except KeyboardInterrupt:
        print("\nDone")
        parser.exit(0)

    except Exception as e:
        parser.exit(type(e).__name__ + ": " + str(e))
