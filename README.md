# Zaamin---An-AI-Power-Video-Analytics-System

## Dataset Preparation

### Collection

For collection of images for annotation an open source tool [Google-Image Scraper](https://github.com/ohyicong/Google-Image-Scraper.git) was used

The main.py we used can be found in [webscraping](https://github.com/Neha-Jafry/Zaamin---An-AI-Power-Video-Analytics-System/blob/main/web-scaping.py).

Steps:

1. clone the repository

2. Install dependencies

    > pip install selenium, requests, pillow

3. replace main.py with webscraping.py

4. python webscraping.py

### Annotation

For annotation, [labelImg](https://sourceforge.net/projects/labelimg.mirror/) an open source image annotation tool was used. LabelImg allows annotation in both PascalVOC and YOLO formats; Yolo format was used for this project.

The class ids were fixed for our particular use ([fix_annot.ipynb](https://github.com/Neha-Jafry/Zaamin---An-AI-Power-Video-Analytics-System/blob/main/fix_annot.ipynb)).

### Augmentation

For augmentation, [Imgaug](https://github.com/aleju/imgaug.git) was used. Details of augmentation can be found in [aug.ipynb](https://github.com/Neha-Jafry/Zaamin---An-AI-Power-Video-Analytics-System/blob/main/aug.ipynb)