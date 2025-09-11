import cv2
import numpy as np
import matplotlib.pyplot as plt

# Функция для создания синтетического изображения
def create_synthetic_image(h=300, w=400):
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Заполняем градиентом
    for i in range(h):
        color = int(255 * (i / h))
        img[i, :, :] = (color, 255 - color, color // 2)

    # Добавляем фигуру
    cv2.circle(img, (w // 2, h // 2), 70, (0, 0, 255), -1)
    cv2.putText(img, "Test", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return img


# Создаём изображение и сохраняем в разных форматах
synthetic_img = create_synthetic_image()

cv2.imwrite("image.jpg", synthetic_img)
cv2.imwrite("image.png", synthetic_img)
cv2.imwrite("image.bmp", synthetic_img)
cv2.imwrite("image.tiff", synthetic_img)

# Сравниваем размеры файлов
formats = ["jpg", "png", "bmp", "tiff"]
file_sizes = {}

for fmt in formats:
    success, buffer = cv2.imencode(f".{fmt}", synthetic_img)
    if success:
        file_sizes[fmt.upper()] = len(buffer)

# Отображение всех изображений
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, fmt in zip(axes.ravel(), formats):
    img = cv2.imread(f"image.{fmt}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(f"{fmt.upper()} ({file_sizes[fmt.upper()]} bytes)")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Выводим в консоль информацию о форматах
print("Информация о сохраненных изображениях:")
for fmt, size in file_sizes.items():
    print(f"{fmt}: размер {size} байт")
