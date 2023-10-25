import cv2
import requests
import numpy as np

# Функция для распознавания несущих опасность действий посетителей
def detect_threat(image):
    # Загружаем предварительно обученную модель распознавания лиц
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
            if is_face_position_dangerous(x, y, w, h):
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

def is_face_position_dangerous(x, y, w, h):
    # Реализуйте здесь логику определения, является ли положение лица опасным
    # Например, если лицо находится слишком близко к границе кадра - считается опасным

    # Возвращаем True, если положение лица опасно, иначе False
    return False

def perform_danger_action():
    # Реализуйте здесь выполнение действий, соответствующих опасному положению лица
    # Например, можно воспроизвести звуковой сигнал, отправить предупреждение и т.д.
  
    pass

# Функция для отправки данных по происшествию веб-приложению на пульт охраны
def send_data_to_security(data):
    # Отправка данных веб-приложению на пульт охраны
    response = requests.post('', data=data)
    
    # Проверка успешности отправки запроса
    if response.status_code == 200:
        return True
    else:
        return False

# Функция для отправки сообщения в мессенджеры группе реагирования
def send_message_to_group(message):

    # Отправка сообщения в группу тг
    response = requests.post('', data={
        'chat_id': 'group_chat_id',
        'text': message
    })
    
    if response.status_code == 200:
        return True
    else:
        return False

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
    if send_data_to_security(data):
        print('Данные успешно отправлены на пульт охраны')
    else:
        print('Ошибка отправки данных на пульт охраны')
    
    # Отправка сообщения в мессенджеры группе реагирования
    message = 'Обнаружено опасное действие в ТЦ Атриум!'
    if send_message_to_group(message):
        print('Сообщение успешно отправлено в группу реагирования')
    else:
        print('Ошибка отправки сообщения в группу реагирования')
else:
    print('Нет обнаруженных опасных действий')
