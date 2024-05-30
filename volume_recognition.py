import numpy as np
import wave
import os

#  pip install PyQt5


'''def draw(s_e, s_i, min_e, min_i):
    for i in s_e:
        plt.plot(i[0:min_e])
    plt.title("exhale")
    plt.show()

    for i in s_i:
        plt.plot(i[0:min_i])
    plt.title("inhale")
    plt.show()'''


# get dirs
def a():
    dir_exhale = os.listdir('data/exhale')
    dir_inhale = os.listdir('data/inhale')
    dir_silence = os.listdir('data/silence')

    # init values
    soundwaves_exhale = []
    soundwaves_inhale = []
    soundwaves_silence = []
    min_exh_len = 1000000
    min_inh_len = 1000000
    min_sil_len = 1000000

    # get waves characteristics and get min length of a sample
    for file in dir_exhale:
        w = wave.open("data/exhale/" + file, 'r')
        signal = w.readframes(-1)
        p = np.frombuffer(signal, dtype='int16')
        soundwaves_exhale.append(p)
        if len(p) < min_exh_len:
            min_exh_len = len(p)

    for file in dir_inhale:
        w = wave.open("data/inhale/" + file, 'r')
        signal = w.readframes(-1)
        p = np.frombuffer(signal, dtype='int16')
        soundwaves_inhale.append(p)
        if len(p) < min_inh_len:
            min_inh_len = len(p)

    for file in dir_silence:
        w = wave.open("data/silence/" + file, 'r')
        signal = w.readframes(-1)
        p = np.frombuffer(signal, dtype='int16')
        soundwaves_silence.append(p)
        if len(p) < min_sil_len:
            min_sil_len = len(p)

    a = []
    b = []
    c = []
    for i in soundwaves_exhale[0:min_exh_len]:
        a.append(np.mean(abs(i)))
    for i in soundwaves_inhale[0:min_inh_len]:
        b.append(np.mean(abs(i)))
    for i in soundwaves_silence[0:min_sil_len]:
        c.append(np.mean(abs(i)))

    avg_exh = (int)(np.ceil(np.mean(a)))
    avg_inh = (int)(np.ceil(np.mean(b)))
    avg_sil = (int)(np.ceil(np.mean(c)))

    return avg_exh, avg_inh, avg_sil

class Volume_Recognition:
    # exhale = 1
    # inhale = 2
    # silence = 0

    def calc(self, sig):
        new = 0
        if sig == 0:
            new = 0
            if self.prev != 0:
                self.prev_to_0 = self.prev
        else:
            if self.prev == 0:
                new = 1 if self.prev_to_0 == 2 else 2
            else:
                new = self.prev
        self.prev = new
        return new

    def volume_update(self, frames, confirm):
        check_frames = abs(frames[::1])

        loud_frames = []
        for i in check_frames:
            if i > self.avg_exh + self.avg_inh:
                loud_frames.append(i)

        l = len(loud_frames)
        output = 0
        if(l > 0):
            output = self.calc(1)
        else:
            output = self.calc(0)

        print(output)



    def __init__(self):
        self.avg_exh, self.avg_inh, self.avg_sil = a()
        self.prev = 0
        self.prev_to_0 = 2
        #print(self.avg_exh, self.avg_inh, self.avg_sil)
        #print(self.avg_exh + self.avg_sil)