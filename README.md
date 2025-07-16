Here's a well-structured **`README.md`** file tailored for your YOLOv8 custom training project using **Label Studio** and **Ultralytics**, aimed at helping students learn the complete pipeline:

---

````markdown
# 🚦 Vehicle Detection Using YOLOv8 & Custom Dataset

Welcome to this beginner-friendly YOLOv8 training project! This repository walks you through the complete pipeline of training a YOLOv8 object detection model on a custom dataset annotated using **Label Studio**.

> 🎯 Goal: Detect 7 vehicle classes from traffic footage using YOLOv8

---

## 📌 Classes

- AutoRickshaw
- Bike
- Bus
- Car
- Cycle
- HeavyVehicle
- Truck

---

## 🧠 What You'll Learn

This repo is perfect if you're a beginner trying to understand how to:

- Annotate custom images using **Label Studio**
- Convert annotations to YOLO format
- Set up `data.yaml` for training
- Train a YOLOv8 model using the Ultralytics CLI
- Evaluate model performance with metrics and confusion matrix

---

## 🛠 Requirements

- Python 3.8+
- GPU (optional but recommended)
- pip

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/yolov8-custom-training.git
cd yolov8-custom-training
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Note: This project uses the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework.

```bash
pip install ultralytics
```

---

## 🏷️ Annotations with Label Studio

We use **Label Studio** to annotate images and export them in YOLO format.

### 🔗 Installation

Please refer to [Label Studio GitHub Repo](https://github.com/heartexlabs/label-studio) for setup instructions.

### ✅ What to Do

1. Launch Label Studio
2. Upload your dataset
3. Create annotation labels (see screenshots)
4. Annotate images
5. Export annotations in **YOLO format**

> 📁 Ensure the dataset directory structure looks like this:

```
datasets/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
```

---

## 📄 data.yaml

Your `data.yaml` should look like this:

```yaml
path: datasets
train: images/train
val: images/val

nc: 7
names: ["bike", "bus", "truck", "car", "autorickshaw", "cycle", "heavy_vehicle"]
```

---

## 🚀 Training the Model

Use the following command to start training:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=4 workers=0 mosaic=0
```

---

## 📊 Model Evaluation

After training:

* Results are saved in `runs/detect/train`
* Metrics include:

  * mAP\@0.5
  * mAP\@0.5:0.95
  * Precision and Recall per class
* Confusion matrix and prediction visuals are automatically generated

---

## 📷 Screenshots

Here are some sample screenshots from the process:

### 🔖 Annotation in Label Studio

![Annotation Sample](screenshots/label-studio-annotation.png)

### 📉 Confusion Matrix

![Confusion Matrix](screenshots/confusion-matrix.png)

### 📈 Training Graphs

![Training Metrics](screenshots/training-results.png)

---

## 📚 Credits & Resources

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Label Studio](https://github.com/heartexlabs/label-studio)
* Dataset collected from custom traffic footage

---

## 🤝 Contributing

Pull requests are welcome if you'd like to expand this tutorial or improve anything!

---

## 📩 License

MIT License — free to use and modify

---

## ❤️ Support

Star ⭐ this repo if you found it helpful!

Happy training! 🔥

```

---

Let me know:
- If you want me to include the actual output of your `results.png`
- If you’d like to host a Google Colab notebook version of this
- If you'd like me to create a `requirements.txt` for this repo as well

Let’s make this super helpful for other students!
```
