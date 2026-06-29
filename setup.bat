@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing packages...
pip install numpy==1.26.4
pip install tensorflow==2.21.0 tensorflow-datasets scikit-learn matplotlib seaborn Pillow opencv-python

echo Checking TensorFlow version...
python -c "import tensorflow as tf; print(tf.__version__)"

echo Exploring dataset...
python explore_dataset.py

echo Training model...
python train.py

echo Setup complete!
pause