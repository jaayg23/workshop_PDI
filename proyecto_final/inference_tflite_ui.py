import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
import time

# --- CARGA DEL MODELO TFLITE ---

# Ruta al archivo del modelo TensorFlow Lite (.tflite)
model_path = "best_saved_model/best_int8.tflite"

# Interpreter es el objeto que maneja el modelo TFLite
# Se encarga de cargar el modelo y administrar los tensores para entrada y salida
interpreter = Interpreter(model_path=model_path)

# allocate_tensors() reserva memoria para los tensores internos del modelo
# Antes de llamar a invoke(), esta función debe ser llamada una vez para preparar el intérprete
interpreter.allocate_tensors()

# input_details y output_details contienen información sobre los tensores de entrada y salida
# Ej: nombre, forma, tipo de dato, índice para acceder a ellos
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def run_inference(image_path, conf_threshold=0.2, nms_threshold=0.4):
    """
    Ejecuta inferencia con un modelo TFLite de YOLOv8.
    """
    # Leer la imagen y obtener sus dimensiones
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    # 1. PREPROCESAR LA IMAGEN DE ENTRADA
    input_size = (640, 640) # Tamaño de entrada esperado por YOLOv8
    resized = cv2.resize(img_rgb, input_size)
    input_tensor = (np.expand_dims(resized, axis=0) / 255.0).astype(np.float32)

    # Cargar imagen en el tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Medir tiempo de inferencia
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # 2. OBTENER LA SALIDA ÚNICA DEL MODELO
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # La forma es (1, 5, 8400), la transponemos a (1, 8400, 5) para que sea más fácil de manejar
    output_data = np.squeeze(output_data).T  # Ahora la forma es (8400, 5)

    # 3. PROCESAR LA SALIDA DE YOLO
    boxes, scores, class_ids = [], [], []

    for row in output_data:
        # La confianza de la clase es el quinto elemento en el caso de un modelo con una sola clase
        confidence = row[4]
        if confidence > conf_threshold:
            # Coordenadas del centro, ancho y alto
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            # Convertir a coordenadas de esquina (x_min, y_min)
            x1 = int((cx - w / 2) * w_img)
            y1 = int((cy - h / 2) * h_img)
            
            # Guardar la caja, la confianza y el ID de la clase (en este caso siempre es 0)
            boxes.append([x1, y1, int(w * w_img), int(h * h_img)])
            scores.append(float(confidence))
            class_ids.append(0) # Asumimos que la clase es 0 si solo hay una

    # 4. APLICAR SUPRESIÓN DE NO MÁXIMOS (NMS)
    # NMS elimina las cajas redundantes y superpuestas para la misma detección
    final_indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

    img_copy = img_rgb.copy()
    class_counts = {}

    if len(final_indices) > 0:
        for i in final_indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # Dibujar la caja final
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Preparar etiqueta
            cls_id = class_ids[i]
            conf = scores[i]
            label = f"Clase {cls_id}: {conf:.2f}"
            
            # Mostrar etiqueta
            label_y = y + 15 if y + 15 < h_img else y - 10
            cv2.putText(img_copy, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Incrementar contador de detecciones
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    # (El código para guardar en .txt y el resto de la función puede permanecer igual o adaptarse si es necesario)
    
    return img_rgb, img_copy, class_counts, inference_time


def display_images(orig, proc):
    """
    Muestra en la interfaz las imágenes original y con detecciones.

    Parámetros:
        orig (np.array): Imagen original RGB.
        proc (np.array): Imagen con detecciones dibujadas.
    """
    orig_img = Image.fromarray(orig).resize((300, 300))
    proc_img = Image.fromarray(proc).resize((300, 300))

    tk_orig = ImageTk.PhotoImage(orig_img)
    tk_proc = ImageTk.PhotoImage(proc_img)

    original_label.configure(image=tk_orig)
    original_label.image = tk_orig
    processed_label.configure(image=tk_proc)
    processed_label.image = tk_proc


def update_class_count_text(class_counts, inf_time):
    """
    Actualiza el texto con el conteo de detecciones y el tiempo de inferencia.

    Parámetros:
        class_counts (dict): Conteo de detecciones por clase.
        inf_time (float): Tiempo que tardó la inferencia (segundos).
    """
    if not class_counts:
        text = "No se detectaron objetos\n"
    else:
        lines = [f"Clase {cls_id}: {count}" for cls_id, count in sorted(class_counts.items())]
        text = "\n".join(lines) + "\n"

    # Agregar tiempo de inferencia al texto
    text += f"\nTiempo inferencia: {inf_time*1000:.1f} ms"

    class_text.set(text)


def select_image():
    """
    Evento para seleccionar una imagen desde el disco.
    Corre la inferencia y actualiza la interfaz.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        orig, proc, counts, inf_time = run_inference(file_path)
        display_images(orig, proc)
        update_class_count_text(counts, inf_time)


# --- INTERFAZ GRÁFICA ---

window = tk.Tk()
window.title("Detección con TFLite y Conteo por Clase")

btn = tk.Button(window, text="Seleccionar imagen", command=select_image)
btn.pack(pady=5)

main_frame = tk.Frame(window)
main_frame.pack()

original_label = tk.Label(main_frame)
original_label.grid(row=0, column=0, padx=10)

processed_label = tk.Label(main_frame)
processed_label.grid(row=0, column=1, padx=10)

class_text = tk.StringVar()
class_label = tk.Label(main_frame, textvariable=class_text, font=("Arial", 12), justify="left", anchor="nw")
class_label.grid(row=0, column=2, padx=10, sticky="n")

window.mainloop()

