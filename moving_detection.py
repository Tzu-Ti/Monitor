import cv2
import numpy as np
import time
import datetime

class Moving_detection():
    def __init__(self):
        self.width = 1280
        self.height = 960
        self.recording = False
        self.startTime = None
        self.endTime = None

    def detection(self):
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        avg = cv2.blur(frame, (4, 4))
        avg_float = np.float32(avg)

        while True:
            _, frame = cap.read()
            origin_frame = frame.copy()
            blur = cv2.blur(frame, (4, 4))
            diff = cv2.absdiff(avg, blur)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            cntImg, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 2500 or self.recording:
                    continue
                self.recording = True
                self.startTime = time.time()
                videoName = datetime.datetime.now().strftime("%H:%M:%S")
                self.out = cv2.VideoWriter('{}.avi'.format(videoName), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25.0, (weight, height))

                print("Start record")
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)
            cv2.imshow('frame', frame)
            
            if self.recording:
                self.endTime = time.time()
                self.out.write(origin_frame)

                if (self.endTime - self.startTime) > 20:
                    print("Record over")
                    self.recording = False
                    self.out.release()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cv2.accumulateWeighted(blur, avg_float, 0.01)
            avg = cv2.convertScaleAbs(avg_float)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    moving_detection = Moving_detection()
    moving_detection.detection()
