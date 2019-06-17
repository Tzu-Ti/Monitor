import cv2

class Record_video():
    def __init__(self):
        self.name = "titi"

    def record(self):
        cap = cv2.VideoCapture(0)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        print(weight, height)

        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25.0, (weight, height))

        while True:
            _, frame = cap.read()
            cv2.imshow("frame", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    record_video = Record_video()
    record_video.record()