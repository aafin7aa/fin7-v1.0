import cv2
import requests
import numpy as np
import json
import dlib
import pygame
from pygame import mixer

# Функция для распознавания несущих опасность действий посетителей
def detect_threat(image):
    # Загружаем предварительно обученную модель распознавания лиц
    face_cascade = cv2.CascadeClassifier('glaz.xml')

    # Открытие видеопотока
    cap = cv2.VideoCapture(0)  # 0 означает использование встроенной камеры

    while True:
        # Чтение кадра из видеопотока
        ret, frame = cap.read()

        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Отрисовка прямоугольников вокруг обнаруженных лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Проверяем, является ли положение лица опасным
            if detect_faces(image):
                # Выполняем действия, соответствующие опасному положению лица
                perform_danger_action()

        # Отображение кадра
        cv2.imshow('Computer Vision', frame)

        # Прерывание цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы и закрываем окна
    cap.release()
    cv2.destroyAllWindows()

def detect_faces(image):
    # Создание объекта для детектирования лиц
    face_detector = dlib.get_frontal_face_detector()

    # Преобразование в оттенки серого для обработки
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детектирование лиц на изображении
    faces = face_detector(gray)

    return len(faces)

def detect_aggressive_behavior(image):
    # Создание объекта для детектирования лиц
    face_detector = dlib.get_frontal_face_detector()

    # Создание объекта для распознавания ключевых точек лица
    landmark_predictor = dlib.shape_predictor("object_lic_toch.dat")

    # Загрузка предварительно обученной модели для определения агрессивного поведения
    aggression_model = dlib.dlibsvm_c_trainer_radial_basis()
    aggression_model.load("arges_lic.md")

    # Преобразование в оттенки серого для обработки
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детектирование лиц на изображении
    faces = face_detector(gray)

    # Проверка агрессивного поведения каждого лица
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        aggression_features = landmark_predictor(gray, face)
        prediction = aggression_model.predict(aggression_features)

        if prediction == 1:
            return True

    return False

# Загрузка изображения для проверки
image = cv2.imread("image.jpg")

# Детектирование количества лиц на изображении
num_faces = detect_faces(image)
print("Количество лиц на изображении:", num_faces)

# Определение агрессивного поведения на изображении
aggressive_behavior = detect_aggressive_behavior(image)
if aggressive_behavior:
    print("Обнаружено агрессивное поведение на изображении")
else:
    print("Не обнаружено агрессивного поведения на изображении")

def perform_danger_action():
    
    pass

# Функция для отправки данных о происшествии в веб-приложение(Пульт охраны)
def send_data_to_web_app(data):
    url = 'АДРЕС_ВЕБ_ПРИЛОЖЕНИЯ'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print('Данные успешно отправлены в веб-приложение')
    else:
        print('Ошибка при отправке данных в веб-приложение')

# Функция для отправки сообщения в мессенджер
def send_message_to_messenger(message):
    bot_token = '6936415987:AAHHPXvjSY7pTyMxM9a9lY_GpDi0ZBJeing'
    chat_id = '6073245376'

    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    params = {'chat_id': chat_id, 'text': message}
    response = requests.post(url, params=params)
    if response.status_code == 200:
        print('Сообщение успешно отправлено в мессенджер')
    else:
        print('Ошибка при отправке сообщения в мессенджер')

# Пример использования
incident_data = {
    'type': 'Происшествие',
    'description': 'Описание происшествия',
    'location': 'Местоположение',
    'date': 'Дата и время'
}

send_data_to_web_app(incident_data)

message = f'Новое происшествие!\nТип: {incident_data["type"]}\nОписание: {incident_data["description"]}\nМесто: {incident_data["location"]}\nДата: {incident_data["date"]}'

send_message_to_messenger(message)

# Изображение с камеры
image = cv2.imread('path_to_image')

# Распознавание несущих опасность действий
is_threat = detect_threat(image)

# Проверка результата распознавания
if is_threat:
    # Отправка данных по происшествию веб-приложению на пульт охраны
    data = {
        'image': image,
        'location': 'ТЦ',
        'timestamp': '2023-10-25 00:00:00'
    }
    if send_data_to_web_app(data):
        print('Данные успешно отправлены на пульт охраны')
    else:
        print('Ошибка отправки данных на пульт охраны')
    
    # Отправка сообщения в мессенджеры группе реагирования
    message = 'Обнаружено опасное действие в ТЦ!'
    if send_message_to_messenger(message):
        print('Сообщение успешно отправлено в группу реагирования')
    else:
        print('Ошибка отправки сообщения в группу реагирования')
else:
    print('Нет обнаруженных опасных действий')
