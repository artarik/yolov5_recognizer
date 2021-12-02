# -*- coding: utf-8 -*-
import cv2 
from flask import Flask, render_template, Response
import torch
import warnings

warnings.filterwarnings("ignore")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')

app = Flask(__name__)

def check_threshold_level(img, threshold_level):
    threshold_level = threshold_level -1
    marker = False
    results = model(img)
    # модель возвращает нам все распознанные объекты, из которых нам нужны только люди, поэтому отфильтруем результат
    df = results.pandas().xyxy[0]
    persons = df[df['class'] == 0]
    # удалим лишние столбцы
    persons = persons.drop(['name', 'class', 'confidence'], axis=1)
    if len(persons) == 0:
        # на фото нет людей
        return img
    # переведем в int все значения координат
    persons = persons.astype('int32')
    # 0 не может быть координатой, заменим на 1
    for column in persons.columns.tolist():  
        persons.loc[persons[column] == 0, column] = 1
    i = 0
    # конвертируем данные в массив numpy для последующей построчной обработки
    points_array = persons.to_numpy()
    for row in points_array:
        # получаем координаты человека и выберем цвет: зеленый, если нет превышения, желтый, 
        # если на фото количество людей равно порогу и красный, если количество превышает пороговое значение
        xmin, ymin, xmax, ymax = row
        if i == threshold_level:
            color = (0, 223, 255)
            marker = True
        elif i > threshold_level:
            marker = True
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        # рисуем границы
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          color, thickness=2)
        # и подписываем                           
        cv2.putText(img, str(i+1),
                        (xmin+5, ymin - 5),
                        cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        thickness=2,
                        color=(80, 127, 255))
        i += 1
    if marker:
        return img

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture('demo.avi')
    while(cap.isOpened()):
        ret, frame = cap.read()  
        if not ret: # если файл закончился, снова запустим capture
            frame = cv2.VideoCapture("demo.avi")
            continue
        if ret:  
            results = model(frame)
            df = results.pandas().xyxy[0]
            persons = df[df['class'] == 0]
            persons = persons.drop(['name', 'class', 'confidence'], axis=1)
            if len(persons) != 0:
                persons = persons.astype('int32')
                for column in persons.columns.tolist():  
                    persons.loc[persons[column] == 0, column] = 1
                i = 0
                points_array = persons.to_numpy()
                for row in points_array:
                        xmin, ymin, xmax, ymax = row
                        if i == 3:
                            color = (0, 223, 255)
                        elif i > 3:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                        color, thickness=2)
                        cv2.putText(frame, str(i+1),
                                        (xmin+5, ymin - 5),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        fontScale=1,
                                        thickness=2,
                                        color=(80, 127, 255))
                        i += 1
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame') 
                    