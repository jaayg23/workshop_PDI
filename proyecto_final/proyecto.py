import torch
import cv2
import numpy as np

modelo = torch.load("best.pt")

modelo.eval()



# 1. Cargar la imagen con OpenCV
img = cv2.imread("9999937_00000_d_0000187_jpg.rf.a9fe6e5b70b7c22b6331dc4816efc3bc.jpg.jpg")  # Devuelve imagen en BGR
img = cv2.resize(img, (640, 640))  # Redimensionar si lo requiere tu modelo

# 2. Convertir de BGR a RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3. Normalizar a [0,1] si el modelo espera eso
img = img.astype(np.float32) / 255.0

# 4. Cambiar de HWC (alto, ancho, canales) a CHW (canales, alto, ancho)
img = np.transpose(img, (2, 0, 1))

# 5. Convertir a tensor
tensor = torch.from_numpy(img)

# 6. Añadir dimensión batch (opcional, pero necesario para inferencia)
tensor = tensor.unsqueeze(0)  # Ahora shape: (1, 3, 640, 640)

# 7. (Opcional) Mover a GPU si estás usando CUDA
# tensor = tensor.to("cuda")

print(tensor.shape)  # Debe ser (1, 3, 640, 640)


dummy_input = torch.randn(1, 3, 640, 640)  # (batch, channels, height, width)

# Exporta a ONNX
torch.onnx.export(
    modelo,                      # modelo pytorch
    dummy_input,                # entrada de prueba
    "model.onnx",               # nombre del archivo de salida
    input_names=['input'],     # nombre del input
    output_names=['output'],   # nombre del output
    dynamic_axes={              # permite batch size variable
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11            # versión del estándar ONNX (11 es compatible)
)