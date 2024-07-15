import os
import dlib
import cv2
import numpy as np
import csv
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Путь к файлу модели
predictor_path = 'models/shape_predictor_68_face_landmarks.dat'

# Инициализация детектора лица и предсказателя ключевых точек
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


class EyeTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracking Application")

        # Инициализация виджетов интерфейса
        self.video_frame = tk.Label(root)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        self.info_text = tk.Text(root, height=10, width=50)
        self.info_text.grid(row=1, column=0, padx=10, pady=10)

        # Создание канвы для сетки
        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.grid(row=0, column=1, rowspan=2, padx=10, pady=10)
        self.draw_grid()

        # Инициализация кнопок управления
        self.start_live_button = tk.Button(root, text="Start Live", command=self.start_live_tracking)
        self.start_live_button.grid(row=2, column=0, sticky='ew', padx=5, pady=10)

        self.start_file_button = tk.Button(root, text="Start from File", command=self.start_file_tracking)
        self.start_file_button.grid(row=2, column=1, sticky='ew', padx=5, pady=10)

        self.start_folder_button = tk.Button(root, text="Start from Folder", command=self.start_folder_tracking)
        self.start_folder_button.grid(row=2, column=2, sticky='ew', padx=5, pady=10)

        self.save_button = tk.Button(root, text="Save", command=self.save_data)
        self.save_button.grid(row=2, column=3, sticky='ew', padx=5, pady=10)

        self.end_button = tk.Button(root, text="End", command=self.end_application)
        self.end_button.grid(row=2, column=4, sticky='ew', padx=5, pady=10)

        # Инициализация переменных состояния
        self.cap = None
        self.running = False
        self.timestamp = None
        self.output_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.count = 1
        self.sentiment = []
        self.timestamps = []
        self.horizontal_ratios = []
        self.vertical_ratios = []
        self.eye_positions = []
        self.video_files = []
        self.current_video_index = 0

        # Initialize self.out as None
        self.out = None

    def draw_grid(self):
        # Рисование сетки на канве
        self.canvas.create_rectangle(0, 0, 100, 100, outline='black')
        self.canvas.create_rectangle(100, 0, 200, 100, outline='black')
        self.canvas.create_rectangle(200, 0, 300, 100, outline='black')
        self.canvas.create_rectangle(0, 100, 100, 200, outline='black')
        self.canvas.create_rectangle(100, 100, 200, 200, outline='black')
        self.canvas.create_rectangle(200, 100, 300, 200, outline='black')
        self.canvas.create_rectangle(0, 200, 100, 300, outline='black')
        self.canvas.create_rectangle(100, 200, 200, 300, outline='black')
        self.canvas.create_rectangle(200, 200, 300, 300, outline='black')

    def highlight_section(self, section):
        # Подсветка выбранной области сетки
        self.canvas.delete("highlight")
        sections = {
            'Нижний Левый': (0, 200, 100, 300),
            'Нижний Центральный': (100, 200, 200, 300),
            'Нижний Правый': (200, 200, 300, 300),
            'Средний Левый': (0, 100, 100, 200),
            'Средний Центральный': (100, 100, 200, 200),
            'Средний Правый': (200, 100, 300, 200),
            'Верхний Левый': (0, 0, 100, 100),
            'Верхний Центральный': (100, 0, 200, 100),
            'Верхний Правый': (200, 0, 300, 100)
        }
        if section in sections:
            coords = sections[section]
            self.canvas.create_rectangle(coords, fill='yellow', outline='black', tags="highlight")

    def start_live_tracking(self):
        # Запуск захвата видео с камеры
        self.cap = cv2.VideoCapture(0)
        self.start_tracking()

    def start_file_tracking(self):
        # Запуск захвата видео из выбранного файла
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_files = [file_path]
            self.current_video_index = 0
            self.cap = cv2.VideoCapture(self.video_files[self.current_video_index])
            self.start_tracking()

    def start_folder_tracking(self):
        # Запуск захвата видео из всех файлов в выбранной папке
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                if f.endswith(('.mp4', '.avi', '.mov'))]
            if self.video_files:
                self.current_video_index = 0
                self.cap = cv2.VideoCapture(self.video_files[self.current_video_index])
                self.start_tracking()

    def start_tracking(self):
        # Инициализация переменных и начало обработки видео
        self.running = True
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join('outputs', self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        csv_filename = os.path.join(self.output_dir, 'output.csv')
        self.csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        fields = ['Время (сек)', 'Горизонтальное Соотношение', 'Вертикальное Соотношение', 'Область Взгляда']
        self.csv_writer.writerow(fields)
        self.process_video()

    def process_video(self):
        # Обработка видео покадрово
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.current_video_index += 1
            if self.current_video_index < len(self.video_files):
                self.cap = cv2.VideoCapture(self.video_files[self.current_video_index])
                self.process_video()
            else:
                self.end_tracking()
            return

        if self.count % 6 == 0:
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Преобразование времени в секунды
            processed_frame, gaze_info, section = self.process_frame(frame, timestamp)
            self.display_frame(processed_frame)
            self.update_info_text(gaze_info)
            self.highlight_section(section)

        self.count += 1
        self.root.after(1, self.process_video)

    def process_frame(self, frame, timestamp):
        # Обработка отдельного кадра для распознавания лица и определения взгляда
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        gaze_info = "Лицо не распознано"
        section = 'N/A'

        if len(faces) == 0:
            row = [round(timestamp, 2), 'N/A', 'N/A', 'Лицо не распознано']
            self.csv_writer.writerow(row)
            self.sentiment.append('Лицо не распознано')
            self.timestamps.append(timestamp)
            self.horizontal_ratios.append(None)
            self.vertical_ratios.append(None)
            self.eye_positions.append((None, None))
            return frame, gaze_info, section

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye_points = [landmarks.part(i) for i in range(36, 42)]
            right_eye_points = [landmarks.part(i) for i in range(42, 48)]

            left_eye_ratio = self.get_eye_aspect_ratio(left_eye_points)
            right_eye_ratio = self.get_eye_aspect_ratio(right_eye_points)
            eye_aspect_ratio = (left_eye_ratio + right_eye_ratio) / 2

            gaze_ratio_left_eye = self.get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame)
            gaze_ratio_right_eye = self.get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame)
            gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

            horizontal_ratio = gaze_ratio
            vertical_ratio = eye_aspect_ratio

            if vertical_ratio < 0.2:
                vertical_section = 'Нижний'
            elif vertical_ratio < 0.3:
                vertical_section = 'Средний'
            else:
                vertical_section = 'Верхний'

            if horizontal_ratio < 1.0:
                horizontal_section = 'Левый'
            elif horizontal_ratio == 1.0:
                horizontal_section = 'Центральный'
            else:
                horizontal_section = 'Правый'

            looking_at_section = f'{vertical_section} {horizontal_section}'

            row = [
                round(timestamp, 2),
                round(horizontal_ratio, 2),
                round(vertical_ratio, 2),
                looking_at_section
            ]

            self.csv_writer.writerow(row)
            self.sentiment.append(looking_at_section)
            self.timestamps.append(timestamp)
            self.horizontal_ratios.append(horizontal_ratio)
            self.vertical_ratios.append(vertical_ratio)

            left_eye_center = (int((left_eye_points[0].x + left_eye_points[3].x) / 2),
                               int((left_eye_points[0].y + left_eye_points[3].y) / 2))
            right_eye_center = (int((right_eye_points[0].x + right_eye_points[3].x) / 2),
                                int((right_eye_points[0].y + right_eye_points[3].y) / 2))

            self.eye_positions.append((left_eye_center, right_eye_center))

            # Отображение ключевых точек и информации о взгляде на кадре
            cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)

            cv2.putText(frame, f"Время: {round(timestamp, 2)} сек", (90, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (147, 58, 31), 1)
            cv2.putText(frame, f"Горизонтальное соотношение: {round(horizontal_ratio, 2)}", (90, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (147, 58, 31), 1)
            cv2.putText(frame, f"Вертикальное соотношение: {round(vertical_ratio, 2)}", (90, 90),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (147, 58, 31), 1)
            cv2.putText(frame, f"Смотрит в {looking_at_section}", (90, 120),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (147, 58, 31), 1)

            gaze_info = f"Время: {round(timestamp, 2)} сек\nГоризонтальное соотношение: {round(horizontal_ratio, 2)}\nВертикальное соотношение: {round(vertical_ratio, 2)}\nСмотрит в {looking_at_section}"
            section = looking_at_section

        return frame, gaze_info, section

    def display_frame(self, frame):
        # Отображение обработанного кадра в интерфейсе
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        img_resized = img.resize((400, 300))  # Изменение размера кадра на 400x300
        imgtk = ImageTk.PhotoImage(image=img_resized)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

    def update_info_text(self, gaze_info):
        # Обновление текстовой информации о взгляде
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, gaze_info)

    def get_eye_aspect_ratio(self, eye):
        # Вычисление соотношения сторон глаза (EAR)
        A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        return (A + B) / (2.0 * C)

    def get_gaze_ratio(self, eye_points, facial_landmarks, gray, frame):
        # Вычисление отношения взгляда
        eye_region = np.array(
            [(facial_landmarks.part(eye_points[i]).x, facial_landmarks.part(eye_points[i]).y) for i in range(6)],
            np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        if threshold_eye is None or threshold_eye.size == 0:
            return 1.0

        height, width = threshold_eye.shape

        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white

        return gaze_ratio

    def save_data(self):
        # Сохранение данных в CSV файл
        if self.csv_file:
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
            messagebox.showinfo("Success", "Data saved successfully.")

    def end_tracking(self):
        # Завершение обработки видео и сохранение результатов
        self.running = False

        # Визуализация и сохранение результатов
        counts = Counter(self.sentiment)
        total_sum = sum(counts.values())
        normalized_values = [value / total_sum for value in counts.values()]

        # Столбчатая диаграмма
        plt.figure(figsize=(10, 6))
        plt.bar(counts.keys(), counts.values(), color='skyblue')
        plt.xlabel('Область Взгляда')
        plt.ylabel('Количество')
        plt.title('Распределение направлений взгляда')
        bar_chart_path = os.path.join(self.output_dir, 'output_bar_chart.png')
        plt.savefig(bar_chart_path)

        # Линейный график
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.horizontal_ratios, label='Горизонтальное соотношение', color='b')
        plt.plot(self.timestamps, self.vertical_ratios, label='Вертикальное соотношение', color='r')
        plt.xlabel('Время (сек)')
        plt.ylabel('Соотношение')
        plt.title('Изменение соотношений во времени')
        plt.legend()
        line_chart_path = os.path.join(self.output_dir, 'output_line_chart.png')
        plt.savefig(line_chart_path)

        # Диаграмма рассеяния
        plt.figure(figsize=(10, 10))
        plt.scatter(self.horizontal_ratios, self.vertical_ratios, color='purple', alpha=0.5)
        plt.xlabel('Горизонтальное соотношение')
        plt.ylabel('Вертикальное соотношение')
        plt.title('Диаграмма рассеяния горизонтального и вертикального соотношений')
        scatter_plot_path = os.path.join(self.output_dir, 'output_scatter_plot.png')
        plt.savefig(scatter_plot_path)

        # Тепловая карта
        heatmap, xedges, yedges = np.histogram2d(
            [pos[0][0] for pos in self.eye_positions if pos[0] is not None],
            [pos[0][1] for pos in self.eye_positions if pos[0] is not None],
            bins=(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 20, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 20)
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Позиция по X')
        plt.ylabel('Позиция по Y')
        plt.title('Тепловая карта позиций глаз')
        heatmap_path = os.path.join(self.output_dir, 'output_heatmap.png')
        plt.savefig(heatmap_path)

        plt.close('all')

        # Освобождение ресурсов
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        if self.csv_file:
            self.csv_file.close()
        messagebox.showinfo("Info", "Video has ended. Returning to initial screen.")
        self.reset_application()

    def end_application(self):
        # Завершение работы приложения
        self.running = False
        if self.cap:
            self.cap.release()
        if self.csv_file:
            self.csv_file.close()
        self.root.destroy()

    def reset_application(self):
        # Сброс состояния приложения
        self.video_frame.config(image='')
        self.info_text.delete(1.0, tk.END)
        self.canvas.delete("all")
        self.draw_grid()
        self.count = 1
        self.sentiment = []
        self.timestamps = []
        self.horizontal_ratios = []
        self.vertical_ratios = []
        self.eye_positions = []


if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackingApp(root)
    root.mainloop()
