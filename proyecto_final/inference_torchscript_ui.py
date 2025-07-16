import torch
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

# --- CARGA DEL MODELO TORCHSCRIPT ---

# Ruta al archivo TorchScript (.torchscript) exportado desde PyTorch
model_path = "best.torchscript"

# Cargar el modelo TorchScript
# TorchScript es una forma serializada y optimizada para ejecutar modelos PyTorch sin necesidad
# del entorno completo de Python o entrenamiento.
model = torch.jit.load(model_path)
model.eval()  # Poner modelo en modo evaluación (sin gradientes)

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Redimensiona y rellena la imagen para mantener la relación de aspecto
    sin deformar, adaptándola al tamaño esperado por el modelo.

    Parámetros:
        image (np.array): Imagen original en RGB.
        new_shape (tuple): Tamaño deseado (height, width).
        color (tuple): Color de relleno para los bordes.

    Retorna:
        padded (np.array): Imagen redimensionada y con relleno.
        ratio (float): Escala usada para redimensionar.
        dw (float): Padding horizontal aplicado (izquierda y derecha).
        dh (float): Padding vertical aplicado (arriba y abajo).
    """
    shape = image.shape[:2]  # height, width
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Escala para redimensionar
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))  # Nuevas dimensiones sin padding
    dw = new_shape[1] - new_unpad[0]  # Diferencia anchura para padding
    dh = new_shape[0] - new_unpad[1]  # Diferencia altura para padding
    dw /= 2  # Padding por cada lado horizontal
    dh /= 2  # Padding por cada lado vertical

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    # Añadir borde con color uniforme para completar el tamaño final (new_shape)
    padded = cv2.copyMakeBorder(resized, int(dh), int(dh), int(dw), int(dw),
                               cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, dw, dh

def run_inference(image_path, conf_threshold=0.25):
    """
    Ejecuta inferencia usando el modelo TorchScript para una imagen dada.

    Pasos conceptuales:
    1. Leer y convertir la imagen a RGB.
    2. Redimensionar y rellenar (letterbox) para mantener relación de aspecto.
    3. Normalizar y transformar a tensor (CHW).
    4. Ejecutar el modelo para obtener predicciones.
    5. Filtrar detecciones por confianza.
    6. Ajustar coordenadas de las cajas detectadas al tamaño original.
    7. Dibujar cajas y etiquetas en la imagen.
    8. Guardar resultados en archivo txt.

    Parámetros:
        image_path (str): Ruta a la imagen.
        conf_threshold (float): Umbral mínimo de confianza para filtrar detecciones.

    Retorna:
        rgb (np.array): Imagen original en RGB.
        img_out (np.array): Imagen con las detecciones dibujadas.
        class_counts (dict): Conteo de objetos detectados por clase.
        inference_time (float): Tiempo de inferencia en segundos.
    """
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocesamiento con letterbox
    img, ratio, dw, dh = letterbox(rgb, (640, 640))

    # Normalizar y transformar a tensor CHW (Channels x Height x Width)
    tensor = img.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.from_numpy(tensor).unsqueeze(0)  # Añadir dimensión batch

    # Ejecutar inferencia y medir tiempo
    start_time = time.time()
    with torch.no_grad():  # No calcular gradientes en inferencia (más eficiente)
        preds = model(tensor)[0].cpu().numpy()  # Salidas en numpy
    inference_time = time.time() - start_time

    detections = []
    h_img, w_img = rgb.shape[:2]
    class_counts = {}

    # Procesar las predicciones
    for det in preds:
        obj_conf = det[4]  # Confianza de objeto (objeto vs fondo)
        if obj_conf < conf_threshold:
            continue

        # Clases predichas con probabilidades, encontrar clase más probable
        class_id = np.argmax(det[5:])
        class_conf = det[5 + class_id]

        # Confianza final: objeto * confianza clase
        conf = obj_conf * class_conf
        if conf < conf_threshold:
            continue

        # Coordenadas de caja (cx, cy, w, h) en imagen procesada (640x640)
        cx, cy, w, h = det[0], det[1], det[2], det[3]

        # Ajustar cajas para imagen original con padding y escala inversa
        x1 = (cx - w / 2 - dw) / ratio
        y1 = (cy - h / 2 - dh) / ratio
        w_box = w / ratio
        h_box = h / ratio

        # Limitar dentro de la imagen original
        x1 = max(0, min(x1, w_img))
        y1 = max(0, min(y1, h_img))
        w_box = max(0, min(w_box, w_img - x1))
        h_box = max(0, min(h_box, h_img - y1))

        x2 = x1 + w_box
        y2 = y1 + h_box

        # Normalizar coordenadas para guardarlas en archivo txt
        x1n = x1 / w_img
        y1n = y1 / h_img
        x2n = x2 / w_img
        y2n = y2 / h_img

        detections.append([x1n, y1n, x2n, y2n, conf, class_id])

        # Contar detecciones por clase
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    # Guardar detecciones en archivo txt
    txt_path = image_path.rsplit(".", 1)[0] + "_detections.txt"
    with open(txt_path, "w") as f:
        for det in detections:
            f.write(f"{det[0]:.6f} {det[1]:.6f} {det[2]:.6f} {det[3]:.6f} {det[4]:.6f} {int(det[5])}\n")
    print(f"Detections saved to: {txt_path}")

    # Dibujar cajas en la imagen original para mostrar
    img_out = rgb.copy()
    for det in detections:
        xmin = int(det[0] * w_img)
        ymin = int(det[1] * h_img)
        xmax = int(det[2] * w_img)
        ymax = int(det[3] * h_img)
        score = det[4]
        cls_id = int(det[5])

        # Rectángulo verde
        cv2.rectangle(img_out, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Etiqueta con clase y confianza
        label = f"Clase {cls_id}: {score:.2f}"
        cv2.putText(img_out, label, (xmin, ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return rgb, img_out, class_counts, inference_time


def display_images(orig, proc):
    """
    Muestra en la UI la imagen original y la imagen con detecciones.

    Parámetros:
        orig (np.array): Imagen original en RGB.
        proc (np.array): Imagen con cajas dibujadas.
    """
    orig_img = Image.fromarray(orig).resize((320, 320))
    proc_img = Image.fromarray(proc).resize((320, 320))
    tk_orig = ImageTk.PhotoImage(orig_img)
    tk_proc = ImageTk.PhotoImage(proc_img)
    original_label.configure(image=tk_orig)
    original_label.image = tk_orig
    processed_label.configure(image=tk_proc)
    processed_label.image = tk_proc


def update_class_count_text(class_counts, inf_time):
    """
    Actualiza el texto en la UI con el conteo de objetos por clase
    y el tiempo que tardó la inferencia.

    Parámetros:
        class_counts (dict): Conteo de detecciones por clase.
        inf_time (float): Tiempo de inferencia en segundos.
    """
    if not class_counts:
        text = "No se detectaron objetos\n"
    else:
        lines = [f"Clase {cls_id}: {count}" for cls_id, count in sorted(class_counts.items())]
        text = "\n".join(lines) + "\n"

    text += f"\nTiempo inferencia: {inf_time*1000:.1f} ms"
    class_text.set(text)


def select_image():
    """
    Abre un diálogo para seleccionar imagen, corre inferencia y actualiza UI.
    """
    fp = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.jpeg *.png *.bmp")])
    if fp:
        orig, proc, counts, inf_time = run_inference(fp)
        display_images(orig, proc)
        update_class_count_text(counts, inf_time)


# --- INTERFAZ GRÁFICA ---

window = tk.Tk()
window.title("Detección YOLOv10 TorchScript con Conteo por Clase")

btn = tk.Button(window, text="Seleccionar imagen", command=select_image)
btn.pack(pady=5)

main_frame = tk.Frame(window)
main_frame.pack()

# Columna 1: Imagen original
original_label = tk.Label(main_frame)
original_label.grid(row=0, column=0, padx=10)

# Columna 2: Imagen con inferencia
processed_label = tk.Label(main_frame)
processed_label.grid(row=0, column=1, padx=10)

# Columna 3: Conteo por clase
class_text = tk.StringVar()
class_label = tk.Label(main_frame, textvariable=class_text, font=("Arial", 12), justify="left", anchor="nw")
class_label.grid(row=0, column=2, padx=10, sticky="n")

window.mainloop()

