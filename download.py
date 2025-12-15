import tensorflow as tf
import pathlib
import shutil
import os

def download_flowers():
    print("Downloading flower dataset...")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    
    # Destination path: ../data/train
    dest_dir = pathlib.Path("../data/train")
    
    if not dest_dir.exists():
        os.makedirs(dest_dir)

    print("Moving images to project folder...")
    # Move the flower subfolders (daisy, rose, etc.) to our data/train folder
    for item in data_dir.glob('*'):
        if item.is_dir():
            target = dest_dir / item.name
            if target.exists():
                shutil.rmtree(target) # Clean up if it already exists
            shutil.move(str(item), target)
            
    print(f"Success! Images are now in {dest_dir.resolve()}")

if __name__ == "__main__":
    download_flowers()