from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


class FactoryApp:

  def __init__(self, root):
    self.root = root
    self.root.title("Информационная система контроля спецодежды")
    self.root.geometry("520x350")
    self.root.resizable(False, False)

    self.video_path = tk.StringVar()
    self.excel_name = tk.StringVar(value="factory_report.xlsx")

    self.build_ui()

  def build_ui(self):
    tk.Label(
        self.root,
        text="Мониторинг соблюдения техники безопасности",
        font=("Arial", 11, "bold"),
    ).pack(pady=15)

    f_vid = tk.Frame(self.root)
    f_vid.pack(fill="x", padx=20, pady=10)
    tk.Label(f_vid, text="Видеофайл:", width=14, anchor="w").pack(side="left")
    tk.Entry(f_vid, textvariable=self.video_path, width=30).pack(
        side="left", padx=5
    )
    tk.Button(f_vid, text="Обзор...", command=self.browse_video).pack(
        side="left"
    )

    f_exc = tk.Frame(self.root)
    f_exc.pack(fill="x", padx=20, pady=10)
    tk.Label(f_exc, text="Имя отчета:", width=14, anchor="w").pack(side="left")
    tk.Entry(f_exc, textvariable=self.excel_name, width=30).pack(
        side="left", padx=5
    )
    tk.Label(f_exc, text=".xlsx", font=("Arial", 9)).pack(side="left")

    self.lbl_status = tk.Label(
        self.root,
        text="Статус: Готово к запуску",
        font=("Arial", 9),
        fg="gray",
    )
    self.lbl_status.pack(pady=10)

    tk.Button(
        self.root,
        text="Запустить обработку и анализ",
        command=self.start_processing,
        font=("Arial", 10, "bold"),
        bg="#2196F3",
        fg="white",
        padx=10,
        pady=5,
    ).pack(pady=5)

  def browse_video(self):
    path = filedialog.askopenfilename(
        title="Выберите видео",
        filetypes=[
            ("Видео файлы", "*.mp4 *.avi *.mov *.mkv"),
            ("Все файлы", "*.*"),
        ],
    )
    if path:
      self.video_path.set(path)

  def start_processing(self):
    v_path = self.video_path.get()
    exc_name = self.excel_name.get().strip()

    if not v_path or not os.path.exists(v_path):
      messagebox.showerror("Ошибка", "Выберите существующий видеофайл!")
      return

    if not exc_name:
      exc_name = "factory_report.xlsx"
    elif not exc_name.endswith(".xlsx"):
      exc_name += ".xlsx"

    self.lbl_status.config(
        text="Статус: Обработка видео и генерация графиков...", fg="blue"
    )
    self.root.update()

    try:
      total_records, chart1, chart2 = self.run_yolo_process(v_path, exc_name)
      self.lbl_status.config(text="Статус: Успешно завершено", fg="green")
      messagebox.showinfo(
          "Готово",
          f"Обработка завершена!\nЗаписано работников: {total_records}\nОтчет: {exc_name}\nСозданы графики: {chart1}, {chart2}",
      )
    except Exception as e:
      self.lbl_status.config(text="Статус: Ошибка", fg="red")
      messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

  def run_yolo_process(self, video_path, excel_filename):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    log_data = []
    frame_stats = []

    frame_idx = 0
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      frame_idx += 1
      results = model.track(
          frame, classes=[0], conf=0.4, persist=True, verbose=False
      )

      current_frame_detections = 0
      if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        current_frame_detections = len(track_ids)

        for box, track_id in zip(boxes, track_ids):
          x1, y1, x2, y2 = map(int, box)
          person_crop = frame[y1:y2, x1:x2]

          if person_crop.size == 0:
            continue

          hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
          mask = cv2.inRange(
              hsv, np.array([90, 50, 50]), np.array([130, 255, 255])
          )
          blue_pixels = cv2.countNonZero(mask)
          total_pixels = person_crop.shape[0] * person_crop.shape[1]

          has_blue = (
              (blue_pixels / total_pixels) > 0.15
              if total_pixels > 0
              else False
          )
          status = "Спецодежда найдена" if has_blue else "Нарушение"
          color = (0, 255, 0) if has_blue else (0, 0, 255)

          cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
          cv2.putText(
              frame,
              f"ID {int(track_id)}: {status}",
              (x1, y1 - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              color,
              2,
          )

          log_data.append({
              "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "Worker ID": int(track_id),
              "Status": status,
          })

      frame_stats.append({"Frame": frame_idx, "Detections": current_frame_detections})

      cv2.imshow("Video Processing - Press 'q' to stop", frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cap.release()
    cv2.destroyAllWindows()

    chart1_path = "compliance_pie_chart.png"
    chart2_path = "activity_line_chart.png"

    if log_data:
      df = pd.DataFrame(log_data)
      
      status_counts = df["Status"].value_counts()
      plt.figure(figsize=(6, 6))
      plt.pie(
          status_counts,
          labels=status_counts.index,
          autopct="%1.1f%%",
          colors=["#4CAF50", "#F44336"],
          startangle=140,
      )
      plt.title("Соотношение соблюдения норм спецодежды")
      plt.tight_layout()
      plt.savefig(chart1_path, dpi=300)
      plt.close()

      if frame_stats:
        df_frames = pd.DataFrame(frame_stats)
        plt.figure(figsize=(8, 4))
        plt.plot(
            df_frames["Frame"],
            df_frames["Detections"],
            color="#2196F3",
            linewidth=2,
        )
        plt.title("Динамика количества обнаруженных работников по кадрам")
        plt.xlabel("Номер кадра")
        plt.ylabel("Количество людей в кадре")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart2_path, dpi=300)
        plt.close()

      df.drop_duplicates(subset=["Worker ID"], keep="last", inplace=True)
      df.to_excel(excel_filename, index=False)
      return len(df), chart1_path, chart2_path

    return 0, "", ""


if __name__ == "__main__":
  root = tk.Tk()
  app = FactoryApp(root)
  root.mainloop()