import glob
import os
import tensorflow as tf
import pathlib

img_paths = glob.glob(os.path.join("/home/kali/.keras/datasets/BDD_FingerVeins/",'001*/*.bmp'))
bad_paths = []
print(img_paths)

badFiles=0
good=0

for path in img_paths:
    try:
      img_bytes = tf.io.read_file(path)
      decoded_img = tf.io.decode_image(img_bytes)
    except tf.errors.InvalidArgumentError as e:
      print(f"Found bad path {path}...{e}")
      badFiles=badFiles+1
      bad_paths.append(path)


print("BAD PATHS:")
for bad_path in bad_paths:
    print(f"{bad_path}")

print(good)
print(badFiles)
print(len(bad_paths))