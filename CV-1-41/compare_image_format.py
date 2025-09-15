import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_synthetic_image(height=300, width=400):
    """Создать синтетическое изображение"""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        color = int(255 * (i / height))
        img[i, :, :] = (color, 255 - color, color // 2)

    cv2.circle(img, (width // 2, height // 2), 70, (0, 0, 255), -1)
    cv2.putText(img, "Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 3)
    return img


def save_images(img, base_filename="synthetic_image", formats=None):
    """Сохранить изображение в указанных форматах"""
    if formats is None:
        formats = ["jpg", "png", "bmp", "tiff"]

    saved_files = []

    for fmt in formats:
        filename = f"{base_filename}.{fmt}"
        success = cv2.imwrite(filename, img)
        if success:
            saved_files.append(filename)
        else:
            print(f"[Ошибка] Не удалось сохранить файл: {filename}")

    return saved_files


def get_file_sizes(file_list):
    """Вернуть размеры файлов в байтах"""
    sizes = {}
    for file in file_list:
        try:
            size = os.path.getsize(file)
            ext = os.path.splitext(file)[1][1:].upper()
            sizes[ext] = size
        except Exception as e:
            print(f"[Ошибка] Не удалось получить размер файла {file}: {e}")
    return sizes


def display_images(file_list, file_sizes):
    """Отобразить изображения с подписями о формате и размере"""
    num_images = len(file_list)
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))

    for ax, file in zip(axes.ravel(), file_list):
        try:
            img = cv2.imread(file)
            if img is None:
                raise ValueError("Файл не прочитан")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = os.path.splitext(file)[1][1:].upper()
            ax.imshow(img_rgb)
            ax.set_title(f"{fmt} ({file_sizes.get(fmt, 0)} bytes)")
            ax.axis("off")
        except Exception as e:
            ax.set_title(f"[Ошибка] {file}")
            ax.axis("off")
            print(f"[Ошибка] Не удалось отобразить {file}: {e}")

    plt.tight_layout()
    plt.show()


def main():
    synthetic_img = create_synthetic_image()

    formats = ["jpg", "png", "bmp", "tiff"]
    saved_files = save_images(synthetic_img, formats=formats)
    file_sizes = get_file_sizes(saved_files)

    print("\nИнформация о сохранённых форматах:")
    for fmt, size in file_sizes.items():
        print(f"{fmt}: размер {size} байт")

    display_images(saved_files, file_sizes)


if __name__ == "__main__":
    main()
