
import sounddevice as sd
import numpy as np

import json
import queue
import sys
from piper.voice import PiperVoice
from vosk import Model, KaldiRecognizer

import ollama


class K9_Loop():

    def __init__(self, q, istream, ostream):
        self.state = "wait"
        self.q     = q
        self.istream = istream
        self.ostream = ostream
    
    def load_rec_model(self, model_name, samplerate):
        print("Loading recognizer model: " + model_name)
        model = Model(lang=model_name)
        self.rec   = KaldiRecognizer(model, samplerate)
        
    def load_tts_model(self, model_name, blocksize):
        print("Loading tts model: " + model_name)

        self.voice = PiperVoice.load(model_name)
        fs = self.voice.config.sample_rate
        return fs

    def load_ollama_model(self, model_name):
        self.ollama_model = model_name

    def say(self, txt):
        print(txt)
        self.istream.stop()
        self.ostream.start()
        for audio_bytes in self.voice.synthesize_stream_raw(txt):
            print(".", end="")
            self.ostream.write(audio_bytes)
        self.ostream.stop()
        self.istream.start()
        print("")        

    def gen_response(self, txt):
        query = "answer in one sentence " + txt
        messages = [
           {'role': 'user', 'content': query},
        ]
        try:
           response = ollama.chat(model=self.ollama_model, messages=messages)
           rsp = response['message']['content']
           self.say(rsp)
        except Exception as e:
           print(type(e).__name__ + ": " + str(e))

    def run(self):

        #TODO add a timeout

        cnt = 0
        while True:
            data = self.q.get()
            if self.rec.AcceptWaveform(data):
                txt = json.loads(self.rec.Result())["text"]
                print("-- " + self.state + " -- " + txt)

                # TODO: Filter any dumb commands here
                if self.state == "cmd":
                    if txt != "" and txt != "huh":
                        self.gen_response(txt)
                        cnt = 0
                    else:
                        cnt = cnt + 1
                        print(".", end="")

                    if txt == "huh":
                        self.rec.Reset()

                    if cnt > 3:
                        self.state = "wait"
            else:
                print(self.rec.PartialResult())
                pt = json.loads(self.rec.PartialResult())["partial"]
                if self.state == "wait" and (pt == "canine" or pt == "k nine" or pt == "hey nine"):
                   self.say("yes master")
                   self.rec.Reset()
                   self.state = "cmd"



