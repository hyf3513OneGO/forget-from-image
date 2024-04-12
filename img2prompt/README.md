# Img2Prompt

Converts input image into prompt.

### Setup

#### Linux

1. **Install Python packages:**

```
pip install -r requirements.txt
```

2. **Run the setup script:**

Make the setup script executable and run it:

```
chmod +x setup.sh
./setup.sh
```

#### Windows

1. **Install Python packages:**

```
pip install -r requirements.txt
```

2. **Run the setup script:**

Open Git Bash in the directory and run:

```
./setup.sh
```

### Running the Project
#### Have a quick look at the result
Input the path to the image file, and the mode to use: ('best', 'classic', 'fast')

```
python img2prompt.py path_to_img.jpg --mode best
```
#### Use it as a lib
```
from img2prompt.imgConvertor import ImgConvertor
imgConvertor = ImgConvertor(device = device1)
raw_prompt = imgConvertor.inference(imgPath,"fast")
```