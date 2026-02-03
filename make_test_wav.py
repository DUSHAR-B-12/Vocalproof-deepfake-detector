import wave, struct
sr = 16000
dur = 1
nframes = sr * dur
path = r"d:\MINI project\test_silence.wav"
with wave.open(path, 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    vals = struct.pack('<' + 'h'*nframes, *([0]*nframes))
    f.writeframes(vals)
print(path)
