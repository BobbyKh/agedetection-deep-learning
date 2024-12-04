import os
from django.shortcuts import render
import cv2
import numpy as np

def index(request):
    return render(request, 'pages/index.html')

def age_detection(request):
    if request.method == 'POST':
        image = request.FILES['image']
        
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces) == 0:
            return render(request, 'pages/index.html', {
                'error_message': 'No faces found in the image.'
            })
        print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        age_net = cv2.dnn.readNetFromCaffe(os.path.join(os.path.dirname(__file__), 'dataset/age_deploy.prototxt'), os.path.join(os.path.dirname(__file__), 'dataset/age_net.caffemodel'))
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = ageList[age_preds[0].argmax()]
            print("Predicted age: " + age)
            cv2.putText(img, age, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            if age:
                return render(request, 'pages/index.html', {'age': age})
        return render(request, 'pages/index.html', {'image': image})
    else:
        return render(request, 'pages/age_detection.html')

