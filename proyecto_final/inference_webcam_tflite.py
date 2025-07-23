import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
import time

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # --- Carga del Modelo TFLite ---
        model_path = "best_saved_model/best_int8.tflite"
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # --- Inicializar la Webcam ---
        self.cap = cv2.VideoCapture(0) # 0 es usualmente la webcam por defecto
        self.current_frame = None

        # --- Configuración de la Interfaz Gráfica ---
        
        # Botón para capturar la imagen
        self.btn_capture = tk.Button(window, text="Capturar e Inferir", command=self.capture_and_infer)
        self.btn_capture.pack(pady=10)

        # Frame principal para las imágenes y texto
        main_frame = tk.Frame(window)
        main_frame.pack()

        # Etiqueta para mostrar la webcam en vivo
        self.webcam_label = tk.Label(main_frame)
        self.webcam_label.grid(row=0, column=0, padx=10)
        
        # Etiqueta para mostrar la imagen procesada
        self.processed_label = tk.Label(main_frame)
        self.processed_label.grid(row=0, column=1, padx=10)
        
        # Etiqueta para mostrar el conteo de clases y tiempo
        self.class_text = tk.StringVar()
        self.class_label = tk.Label(main_frame, textvariable=self.class_text, font=("Arial", 12), justify="left", anchor="nw")
        self.class_label.grid(row=0, column=2, padx=10, sticky="n")

        # Iniciar el bucle para actualizar la webcam
        self.update_webcam()
        
        # Asegurarse de que la cámara se libere al cerrar la ventana
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_webcam(self):
        """Lee un frame de la webcam y lo muestra en la interfaz."""
        ret, frame = self.cap.read()
        if ret:
            # Guardar el frame actual (en formato BGR)
            self.current_frame = frame
            
            # Convertir de BGR a RGB para mostrar en Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para la vista previa
            img = Image.fromarray(rgb_frame).resize((400, 400))
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
            
        # Llamar a esta función de nuevo después de 10ms
        self.window.after(10, self.update_webcam)

    def capture_and_infer(self):
        """Captura el frame actual de la webcam y ejecuta la inferencia."""
        if self.current_frame is not None:
            # La inferencia se hace sobre el frame BGR capturado
            proc, counts, inf_time = self.run_inference(self.current_frame)
            
            # Mostrar la imagen procesada
            proc_img = Image.fromarray(proc).resize((400, 400))
            tk_proc = ImageTk.PhotoImage(proc_img)
            self.processed_label.configure(image=tk_proc)
            self.processed_label.image = tk_proc
            
            # Actualizar el texto de conteo
            self.update_class_count_text(counts, inf_time)

    def run_inference(self, image_bgr, conf_threshold=0.2, nms_threshold=0.4):
        """Ejecuta inferencia en un array de imagen (frame de la cámara)."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img_rgb.shape

        # PREPROCESAR LA IMAGEN
        input_size = (640, 640)
        resized = cv2.resize(img_rgb, input_size)
        input_tensor = (np.expand_dims(resized, axis=0) / 255.0).astype(np.float32)

        # INFERENCIA
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time

        # PROCESAR SALIDA
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data = np.squeeze(output_data).T

        boxes, scores, class_ids = [], [], []
        for row in output_data:
            confidence = row[4]
            if confidence > conf_threshold:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = int((cx - w / 2) * w_img)
                y1 = int((cy - h / 2) * h_img)
                boxes.append([x1, y1, int(w * w_img), int(h * h_img)])
                scores.append(float(confidence))
                class_ids.append(0)

        final_indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

        img_copy = img_rgb.copy()
        class_counts = {}

        if len(final_indices) > 0:
            for i in final_indices.flatten():
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cls_id, conf = class_ids[i], scores[i]
                label = f"Clase {cls_id}: {conf:.2f}"
                label_y = y + 15 if y + 15 < h_img else y - 10
                cv2.putText(img_copy, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        
        return img_copy, class_counts, inference_time

    def update_class_count_text(self, class_counts, inf_time):
        """Actualiza el texto con el conteo."""
        if not class_counts:
            text = "No se detectaron objetos\n"
        else:
            lines = [f"Clase {cls_id}: {count}" for cls_id, count in sorted(class_counts.items())]
            text = "\n".join(lines) + "\n"

        text += f"\nTiempo inferencia: {inf_time*1000:.1f} ms"
        self.class_text.set(text)
        
    def on_closing(self):
        """Se ejecuta al cerrar la ventana."""
        self.cap.release() # Libera la cámara
        self.window.destroy() # Cierra la ventana

# --- Iniciar la aplicación ---
if __name__ == "__main__":
    app = WebcamApp(tk.Tk(), "Detección desde Webcam con TFLite")
    app.window.mainloop()