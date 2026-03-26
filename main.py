import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time


# Константы
OCCUPIED_COLOR = (0, 0, 255)
FREE_COLOR = (0, 255, 0)
HUMAN_COLOR = (255, 0, 0)
RESOLUTION = (1280, 720) # разрешение окон
DEBOUNCE_TIME = 3.0  # дэбоунс состояния стола в секундах
GAP_TOLERANCE = 0.2  # сколько секунд можно отсутствовать без сброса таймера

# Внешние переменные
points = []


def is_window_closed(window_name):
    """Проверяет, закрыто ли окно"""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except:
        return True


def mouse_callback(event, x, y, flags, param):
    """Добавление точки полигона стола"""
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Добавлена точка: {(x, y)}")


def bbox_center(bbox):
    """Определение центра полигона человека"""
    x_1, y_1, x_2, y_2 = bbox
    return (x_1 + x_2) // 2, (y_1 + y_2) // 2


def point_in_polygon(point, polygon):
    """Есть ли пересечение точки в полигоне"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def point_in_circle_polygon(people_center, polygon, radius=30, steps=8):
    """
    Проверка: есть ли хотя бы одна точка из круга вокруг центра в полигоне
    people_center: (x, y)
    radius: радиус в пикселях
    polygon: np.array с координатами стола
    steps: сколько точек по окружности
    """
    cx, cy = people_center
    for angle in np.linspace(0, 2*np.pi, steps, endpoint=False):
        x = int(cx + radius * np.cos(angle))
        y = int(cy + radius * np.sin(angle))
        if point_in_polygon((x, y), polygon):
            return True
    return False



def main():
    # Создаём парсер аргументов
    parser = argparse.ArgumentParser(description="Программа для обработки видео")

    # Добавляем аргумент --video
    parser.add_argument(
        '--video',
        type=str,
        required=True,  # делаем аргумент обязательным
        help='Путь к видеофайлу'
    )

    # Разбираем аргументы
    args = parser.parse_args()
    # Определение модели детекцию движения
    model = YOLO("yolov8n.pt")

    # Выбор видео
    video = cv2.VideoCapture(f"videos/{args.video}")

    FPS = video.get(cv2.CAP_PROP_FPS)

    select_window_name = f"Select tabel {args.video}"
    video_name = str(args.video)

    # Открытие окна определение стола
    cv2.namedWindow(select_window_name)
    cv2.setMouseCallback(select_window_name, mouse_callback)

    # Первый кадр
    ret, frame = video.read()


    # =======================ОПРЕДЕЛЕНИЕ СТОЛА==================================

    while True:
        small_frame = cv2.resize(frame, RESOLUTION).copy()

        # Отрисовка выбранных точек
        for p in points:
            cv2.circle(small_frame, p, 5, FREE_COLOR, -1)

        # Соединение линий
        if len(points) > 1:
            cv2.polylines(small_frame, [np.array(points)], False, FREE_COLOR, 2)

        cv2.imshow(select_window_name, small_frame)


        # Определение нажатой клавиши
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            break
        elif key == 27 or is_window_closed(select_window_name):  # Esc или закрытие окна
            cv2.destroyAllWindows()
            video.release()
            exit()

    # Определение выбранных точек в roi массив
    roi_polygon = np.array(points)
    # Закрытие окна определения стола
    cv2.destroyWindow(select_window_name)

    # =======================ЗАПУСК ВИДЕО==================================

    prev_occupied = False
    candidate_start_time = None
    candidate_end_time = None
    last_seen_time = None  # последний момент, когда человек был в полигоне
    events = []

    # Создаём объект для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # кодек для .mp4
    output_video = cv2.VideoWriter(
        f"output/output_{args.video}",  # путь к файлу
        fourcc,
        FPS,
        RESOLUTION
    )

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Определение кадра
        frame_small = cv2.resize(frame, RESOLUTION)
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        results = model(frame_small)[0]

        # Определение и отрисовка людей в кадре
        people = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x_1, y_1, x_2, y_2 = map(int, box.xyxy[0])
                center = bbox_center((x_1, y_1, x_2, y_2))

                # динамический радиус (половина ширины)
                radius = (x_2 - x_1) // 2

                # сохраняем вместе с центром и радиусом
                people.append((center, radius))

                # рисуем квадрат человека
                cv2.rectangle(frame_small, (x_1, y_1), (x_2, y_2), HUMAN_COLOR, 2)
                # рисуем круг с радиусом
                cv2.circle(frame_small, center, radius, HUMAN_COLOR, 2)

        # Определение - есть ли человек за столом
        detected_now = False
        for center, radius in people:
            if point_in_circle_polygon(center, roi_polygon, radius):
                detected_now = True
                break

        now = time.time()

        if detected_now:
            last_seen_time = now
            candidate_end_time = None  # сбрасываем таймер выхода
            if candidate_start_time is None:
                candidate_start_time = now
        else:
            # Человека нет на кадре, но допускаем GAP_TOLERANCE
            if last_seen_time is not None and (now - last_seen_time) <= GAP_TOLERANCE:
                pass
            else:
                # Для входа
                candidate_start_time = None
                # Для выхода
                if prev_occupied and candidate_end_time is None:
                    candidate_end_time = now

        # Проверка, можно ли сменить состояние
        if not prev_occupied:
            if candidate_start_time is not None and (now - candidate_start_time) >= DEBOUNCE_TIME:
                prev_occupied = True
                candidate_start_time = None
                candidate_end_time = None
                events.append({"timestamp_sec": (frame_number / FPS), "event": "table_occupied"})

        else:
            if candidate_end_time is not None and (now - candidate_end_time) >= DEBOUNCE_TIME:
                prev_occupied = False
                candidate_start_time = None
                candidate_end_time = None
                events.append({"timestamp_sec": (frame_number / FPS), "event": "table_empty"})

        # Присваиваем цвет состояния столу
        state_color = OCCUPIED_COLOR if prev_occupied else FREE_COLOR
        cv2.polylines(frame_small, [roi_polygon], True, state_color, 2)


        # Отображение самого видео
        cv2.imshow(video_name, frame_small)

        # Запись в видео
        output_video.write(frame_small)

        # Определение нажатой кнопки
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or is_window_closed(video_name):  # Esc или закрытие окна
            break

    # Закрытие окна при завершении видео
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(events)
    csv_name = str(args.video).replace(".mp4", "")
    df.to_csv(csv_name, index=False)
    df.to_csv( f"output/events_{csv_name}.csv", index=False)

    empty_times = [e["timestamp_sec"] for e in events if e["event"] == "table_empty"]
    occupied_times = [e["timestamp_sec"] for e in events if e["event"] == "table_occupied"]

    # Считаем интервалы: от ухода до подхода следующего
    intervals = []
    j = 0
    for empty_time in empty_times:
        # ищем ближайшее occupied_time после empty_time
        while j < len(occupied_times) and occupied_times[j] <= empty_time:
            j += 1
        if j < len(occupied_times):
            intervals.append(occupied_times[j] - empty_time)

    if intervals:
        average_interval = np.mean(intervals)
        print(f"Среднее время между уходом гостя и подходом следующего человека: {average_interval:.2f} сек")
    else:
        print("Нет данных для расчета среднего времени между уходами и подходами")


if __name__ == "__main__":
    main()