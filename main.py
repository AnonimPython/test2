import cv2

# Загрузка предварительно обученного классификатора для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка предварительно обученного классификатора для обнаружения глаз
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Загрузка изображения
image = cv2.imread('your_image.jpg')

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображении
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Для каждого обнаруженного лица
for (x, y, w, h) in faces:
    # Обнаружение глаз внутри лица
    eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])

    # Для каждого обнаруженного глаза
    for (ex, ey, ew, eh) in eyes:
        # Отображение квадрата вокруг левого глаза (можно изменить цвет и толщину линии)
        if ex < x + w/2:  # Проверка, что глаз находится слева от центра лица
            cv2.rectangle(image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Отображение изображения с квадратом вокруг левого глаза
cv2.imshow('Face Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()