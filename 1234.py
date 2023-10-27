import cv2
import numpy as np
from google.auth.transport import requests
import requests as requests_
import telegram
import datetime

# Загрузка предварительно обученной модели для обнаружения позы человека (YOLOv4)
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

# Загрузка видеофайла и инициализация видеопотока
video_path = '1231.mp4'
cap = cv2.VideoCapture(video_path)


# Перебор каждого кадра видео
def send_message_to_messenger(location, time, description):
    # Код отправки сообщения в Telegram
    bot_token = '6936415987:AAHHPXvjSY7pTyMxM9a9lY_GpDi0ZBJeing'
    chat_id = '6073245376'

    bot = telegram.Bot(token=bot_token)
    message = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nМесто: {location}\nВремя: {time}\nОписание: {description}"

    try:
        bot.send_message(chat_id=chat_id, text=message)
        print('Сообщение успешно отправлено в мессенджер')
    except telegram.TelegramError as e:
        print(f'Ошибка при отправке сообщения в мессенджер: {e}')

        send_message_to_messenger("Место происшествия", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  class_name)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Получение высоты и ширины кадра
    height, width, _ = frame.shape

    # Создание блоба из кадра для подачи на вход модели
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), (0, 0, 0), True, crop=False)

    # Установка блоба в модель и выполнение предсказания
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers_names)

    # Обработка предсказаний и отображение результатов на видео
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Определение класса объекта
            if class_id == 0:
                class_name = "Человек"
            elif class_id == 1:
                class_name = "Нож"
            elif class_id == 2:
                class_name = "Оружие"
            else:
                class_name = "Другой объект"

            # Отображение результатов предсказания на видео для человека
            if confidence > 0.5 and class_name.lower() == "человек":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Обводка прямоугольником
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Predicted class: {class_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Отображение результатов предсказания на видео для оружия
            if confidence > 0.5 and class_name.lower() in ["нож", "оружие"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Predicted class: {class_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Отправка сообщения с опасным действием в мессенджер
                send_message_to_messenger("Место происшествия", "Время происшествия", "Описание происшествия")

    # Отображение кадра с результатами на видео
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()