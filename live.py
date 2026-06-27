from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-5/weights/best.pt")   # your trained model
cap = cv2.VideoCapture(0)                              # 0 = default webcam

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, device=0, verbose=False)    # run model on the frame, on GPU

    for r in results:
        annotated = r.plot()                           # draw the detection boxes
        classes = r.boxes.cls.tolist()                 # class ids found this frame

        violation = 0 in [int(c) for c in classes]     # is a bare head (class 0) present?
        if violation:                                  # if so, flash the warning
            cv2.putText(annotated, "NO HELMET", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("PPE", annotated)                    # show the frame

    if cv2.waitKey(1) == ord("q"):                      # press q to quit
        break

cap.release()
cv2.destroyAllWindows()