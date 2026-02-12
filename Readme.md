1. SB Versiya v6
py -3.11 -m venv venv
venv\Scripts\activate

#Install
python -m pip install --upgrade pip
pip install torch opencv-python sounddevice librosa numpy

###Audio uchun FM radio####
pip install requests pydub ffmpeg-python

# Qo'shimcha modul
pip install requests



#Run Ishga tushirish
python main.py