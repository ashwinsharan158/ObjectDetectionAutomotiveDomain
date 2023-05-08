import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib

from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

detection_threshold = 0.8

cap = cv2.VideoCapture('./video.mp4')

out = cv2.VideoWriter('inference_outputs/videos/output_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color, 2)
                cv2.putText(frame, class_name,
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                            2, lineType=cv2.LINE_AA)
                
        cv2.imshow('image', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()