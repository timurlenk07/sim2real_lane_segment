import cv2
from os import path, getcwd, mkdir
from queue import SimpleQueue
from threading import Thread


class Recorder():
    def __init__(self):
        # default recordings directory
        self.rec_dir = path.join(getcwd(), "recordings")
        if not path.exists(self.rec_dir):
            mkdir(self.rec_dir)        

        # starting sequence numbers
        self.seq_num = 0
        self.format = 'avi'
        self.buffer = SimpleQueue()

        self.saveThread = None
        self.isRecording = False

    def startRecording(self, filename, fourcc=cv2.VideoWriter_fourcc(*'FFV1'), fps=20, framesize=(640,480), isColor=True, starting_seq_num = 0, overwrite = False):
        """
        :param filename: the desired filename without extension
        :param overwrite: uses the given filename and starting_seq_num to save the video regardless of its existence
         """
        
        if self.saveThread is not None:
            return
        
        # sanitize input parameters
        filename = filename.split(".")[0]
        if len(filename) == 0:
            raise TypeError("Please supply a valid filename.")
        max_seq_num = 99999
        if starting_seq_num < 0 or starting_seq_num > max_seq_num:
            raise TypeError("Be reasonable! Please chose a sequence number between 0-{}.".format(max_seq_num))

        # find a suitable filename
        self.seq_num = starting_seq_num if starting_seq_num > self.seq_num else self.seq_num
        full_filename = "{:05d}_{}.{}".format(self.seq_num, filename, self.format)
        if not overwrite:
            for seq_num in range(self.seq_num, max_seq_num+1):
                full_filename = "{:05d}_{}.{}".format(seq_num, filename, self.format)
                if not path.exists(path.join(self.rec_dir, full_filename)):
                    self.seq_num = seq_num + 1  # prepare for the next file
                    break
        
        writer = cv2.VideoWriter(path.join(self.rec_dir, full_filename), fourcc, fps, framesize, isColor)
        self.saveThread = Thread(target=self._saveImage, args=[writer, self.buffer, full_filename])
        self.saveThread.start()
        
        self.isRecording = True

    def record(self, image):
        self.buffer.put(image)
    
    def stopRecording(self):
        if not self.isRecording:
            return
        
        self.buffer.put(None)
        self.saveThread.join()
        self.saveThread = None
        self.isRecording = False

    def _saveImage(self, writer, queue, full_filename):
        # do the actual saving
        print("Saving {} ...".format(full_filename))
        while True:
            img = queue.get()
            
            if img is None:
                break
            
            # TODO shift the pixels from RGB to BGR
            writer.write(img)
        writer.release()
        print("Saved {}".format(full_filename))
