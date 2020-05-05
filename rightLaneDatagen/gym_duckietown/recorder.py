from os import path, getcwd, mkdir
from queue import SimpleQueue
from threading import Thread

import cv2


class Recorder():
    def __init__(self):
        # default recordings directory
        self.rec_dir = path.join(getcwd(), "recordings")
        if not path.exists(self.rec_dir):
            mkdir(self.rec_dir)

        # starting sequence numbers
        self.seq_num = 0
        self.format = 'avi'
        self.fourcc = cv2.VideoWriter_fourcc(*'FFV1')

        self.buffer = SimpleQueue()
        self.saveThread = None
        self.isRecording = False

    def startRecording(self, filename, fps=30, framesize=(640, 480), isColor=True,
                       starting_seq_num=0, overwrite=False) -> bool:
        if self.saveThread is not None:
            return False

        # sanitize input parameters
        filename = filename.split(".")[0]
        if len(filename) == 0:
            raise TypeError("Please supply a valid filename.")
        max_seq_num = 99999
        if starting_seq_num < 0 or starting_seq_num > max_seq_num:
            raise TypeError("Be reasonable! Please chose a sequence number between 0-{}.".format(max_seq_num))

        # find a suitable filename
        self.seq_num = starting_seq_num if starting_seq_num > self.seq_num else self.seq_num
        full_filename = f'{self.seq_num:05d}_{filename}.{self.format}'
        if not overwrite:
            for seq_num in range(self.seq_num, max_seq_num + 1):
                full_filename = f'{self.seq_num:05d}_{filename}.{self.format}'
                if path.exists(path.join(self.rec_dir, full_filename)):
                    self.seq_num = seq_num + 1  # prepare for the next file
                else:
                    break

        writer = cv2.VideoWriter(path.join(self.rec_dir, full_filename), self.fourcc, fps, framesize, isColor)
        self.saveThread = Thread(target=saveVideoFromQueue, args=[writer, self.buffer, full_filename])
        self.saveThread.start()

        self.isRecording = True
        return True

    def record(self, image):
        self.buffer.put(image)

    def stopRecording(self):
        if not self.isRecording:
            return

        self.buffer.put(None)
        self.saveThread.join()
        self.saveThread = None
        self.isRecording = False


def saveVideoFromQueue(writer, queue, full_filename='FILENAME_NOT_AVAILABLE'):
    # do the actual saving
    print(f"Saving {full_filename} ...")
    while True:
        img = queue.get()

        if img is None:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.write(img)
    writer.release()
    print(f"Saved {full_filename}")
