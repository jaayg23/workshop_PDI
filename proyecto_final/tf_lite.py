from ultralytics import YOLO

# 1. Carga tu modelo YOLO entrenado con Roboflow
# La librería lee el .pt y entiende la arquitectura automáticamente
model = YOLO('best.pt')  # <--- Usa la ruta a tu archivo .pt

# 2. Exporta el modelo directamente a TFLite INT8
# La magia ocurre aquí
model.export(
    format='tflite',    # El formato de salida
    int8=True,          # ¡Activa la cuantización INT8!
    data='proyecto_PDI.v8i.yolov8/data.yaml',   # La ruta a tu archivo .yaml para la calibración
    imgsz=640,          # El tamaño de imagen con el que entrenaste (ej: 640)
    simplify=True       # Opcional: Simplifica el grafo ONNX intermedio para mayor compatibilidad
)

print("✅ ¡Conversión a TFLite INT8 completada!")
print("Busca el archivo 'tu_modelo_int8.tflite' en la misma carpeta.")