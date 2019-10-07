from scipy.io import wavfile

rate,snd = wavfile.read(filename='/home/gabriel/Documentos/github/udemy/arquivos/sms.wav')

from IPython.display import Audio
import matplotlib.pyplot as plt

print(Audio(data=snd,rate=rate))
print(len(snd))
print(snd)

plt.plot(snd)

_ = plt.specgram(snd, NFFT=1024,Fs=44100,noverlap=900)
plt.ylabel('Frequency')
plt.xlabel('Time')

plt.show()