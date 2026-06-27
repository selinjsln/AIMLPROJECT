import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. LOAD DATASET
# ============================================================
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True,  # <-- ini biar kita bisa lihat info dataset
)

# ============================================================
# 2. INFO DASAR DATASET
# ============================================================
print("=== INFO DATASET ===")
print(ds_info)

print("\n=== UKURAN DATA ===")
print(f"Jumlah data train : {ds_info.splits['train'].num_examples:,}")
print(f"Jumlah data test  : {ds_info.splits['test'].num_examples:,}")
print(f"Jumlah kelas      : {ds_info.features['label'].num_classes}")

# Label mapping: angka → karakter
label_names = ds_info.features["label"].names
print(f"\nContoh label mapping:")
for i in range(10):
    print(f"  index {i} → '{label_names[i]}'")

# ============================================================
# 3. LIHAT BENTUK 1 SAMPEL
# ============================================================
for image, label in ds_train.take(1):
    print(f"\n=== BENTUK SATU SAMPEL ===")
    print(f"Shape image : {image.shape}")  # harusnya (28, 28, 1)
    print(f"Dtype image : {image.dtype}")  # uint8
    print(f"Label index : {label.numpy()}")
    print(f"Label char  : '{label_names[label.numpy()]}'")
    print(f"Pixel min   : {image.numpy().min()}")
    print(f"Pixel max   : {image.numpy().max()}")

# ============================================================
# 4. VISUALISASI 25 SAMPEL PERTAMA
# ============================================================
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("25 Sampel Pertama EMNIST Balanced", fontsize=14)

for i, (image, label) in enumerate(ds_train.take(25)):
    row, col = divmod(i, 5)

    # EMNIST perlu di-rotate & flip karena orientasinya terbalik
    img = image.numpy().squeeze()  # hapus channel dim → (28, 28)
    img = np.rot90(img, k=3)  # rotate 270 derajat
    img = np.fliplr(img)  # flip horizontal

    axes[row][col].imshow(img, cmap="gray")
    axes[row][col].set_title(f"'{label_names[label.numpy()]}'", fontsize=10)
    axes[row][col].axis("off")

plt.tight_layout()
plt.savefig("sample_preview.png")
plt.show()
print("\nGambar tersimpan: sample_preview.png")

# ============================================================
# 5. DISTRIBUSI KELAS (apakah data seimbang?)
# ============================================================
print("\n=== DISTRIBUSI KELAS ===")
label_counts = np.zeros(47, dtype=int)

for _, label in ds_train:
    label_counts[label.numpy()] += 1

plt.figure(figsize=(14, 4))
plt.bar(label_names, label_counts)
plt.title("Distribusi Jumlah Sampel per Kelas (Train Set)")
plt.xlabel("Karakter")
plt.ylabel("Jumlah Sampel")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("distribusi_kelas.png")
plt.show()
print("Grafik tersimpan: distribusi_kelas.png")