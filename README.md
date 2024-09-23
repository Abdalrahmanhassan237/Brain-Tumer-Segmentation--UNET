# Brain Tumor Segmentation using U-Net
![scans](https://github.com/user-attachments/assets/660893b6-8bc6-42b1-bc48-3caecaf0a953)

## Project Overview
This project focuses on brain tumor segmentation using MRI images and a deep learning approach. The U-Net architecture, a popular convolutional neural network for biomedical image segmentation, is used to distinguish tumor regions from non-tumor areas in brain MRI scans. The dataset used for this task is the **LGG MRI Segmentation Dataset**, which contains paired MRI images and corresponding tumor masks.

## Problem Statement
Brain tumors, particularly low-grade gliomas (LGG), are life-threatening and need timely detection. Accurate segmentation of the tumor from MRI images is critical for planning treatment. Manual segmentation is time-consuming and subject to human error. This project aims to automate the segmentation process using U-Net to enhance precision and reduce the time required for diagnosis.
![Screenshot_23-9-2024_153845_bfdogplmndidlpjfhoijckpakkdjkkil](https://github.com/user-attachments/assets/fd07eacd-9056-420e-a459-17bca867759c)

## Objective
- Develop a U-Net model to perform pixel-wise segmentation of brain tumors in MRI images.
- Enhance the model using image preprocessing and data augmentation techniques.
- Achieve high segmentation performance, focusing on metrics like Dice Coefficient and Intersection over Union (IoU).

## Project Workflow

### 1. Data Access
- The dataset is sourced from Kaggle: **LGG MRI Segmentation Dataset**.
- Images are preprocessed to match the input size for the model.

## Dataset
The dataset used for this project is the **LGG MRI Segmentation Dataset**, which is available on Kaggle. You can access the dataset via the following link:

[Kaggle: LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

### 2. Data Preprocessing
- Resize images and masks to a consistent shape.
- Convert images to grayscale when necessary.
- Perform data augmentation (e.g., random rotations, flips) to increase model robustness.

### 3. U-Net Architecture
The U-Net model is designed for image segmentation tasks, featuring:
- **Encoder (Contracting Path)**: Consists of repeated convolutional layers followed by max-pooling to downsample the feature maps.
- **Bottleneck**: Central part that captures abstracted features of the image.
- **Decoder (Expanding Path)**: Involves upsampling layers and concatenation with corresponding encoder layers to restore spatial resolution.
- **Dropout** and **Batch Normalization** are used to improve generalization.

### 4. Training Setup
- The model is trained using the **Adam** optimizer.
- **EarlyStopping** is employed to prevent overfitting by stopping training when the validation loss plateaus.
- **ModelCheckpoint** ensures that the best-performing model (based on validation performance) is saved.

### 5. Performance Evaluation
- The primary metric used for evaluating segmentation performance is the **Dice Coefficient**, which measures the overlap between predicted segmentation and ground truth masks.
- Additional metrics include **IoU** and pixel-wise accuracy.

![Screenshot_23-9-2024_153954_bfdogplmndidlpjfhoijckpakkdjkkil (1)](https://github.com/user-attachments/assets/f8891cb7-d956-4e83-912d-3e3b183969d7)
![Screenshot_23-9-2024_15407_bfdogplmndidlpjfhoijckpakkdjkkil](https://github.com/user-attachments/assets/0f56e272-d642-4f7e-9541-a0d99bb9d8d2)

### 6. Results & Findings
- The U-Net model successfully segments tumor regions from the MRI images.
- Data augmentation improves model generalization on unseen data.
- Metrics indicate a high degree of overlap between predicted tumor regions and actual masks, with a Dice Coefficient close to the state-of-the-art benchmarks.
![Screenshot_23-9-2024_15411_bfdogplmndidlpjfhoijckpakkdjkkil](https://github.com/user-attachments/assets/2cf64cbd-b9fd-4086-af6d-157a331564a5)
![Screenshot_23-9-2024_15412_bfdogplmndidlpjfhoijckpakkdjkkil](https://github.com/user-attachments/assets/d7f72285-ff87-445d-9ca3-c997458d4fe9)
![Screenshot_23-9-2024_154043_bfdogplmndidlpjfhoijckpakkdjkkil](https://github.com/user-attachments/assets/d8ac0ecd-a667-43ce-bd74-f7e75dcef33d)
![Screenshot_23-9-2024_154032_bfdogplmndidlpjfhoijckpakkdjkkil](https://github.com/user-attachments/assets/cd382963-8d77-4b76-a7c8-fdb98d4a10ed)

## Tools and Libraries Used
- **TensorFlow**: Model building and training.
- **OpenCV**: Image processing and visualization.
- **Scikit-image**: Image transformation utilities.
- **NumPy** and **Pandas**: Data handling and matrix operations.
- **Matplotlib**: Visualizing training progress and results.

## Insights & Learnings
- **U-Net's encoder-decoder structure**: It is highly effective for pixel-wise image segmentation, allowing the network to recover spatial information and refine segmentation boundaries.
- **Data Augmentation**: Critical for preventing overfitting, especially with limited medical imaging data.
- **Preprocessing**: Proper resizing, normalization, and augmentation play an essential role in enhancing model performance.

## Future Work
- **Hyperparameter Tuning**: Explore different optimizer configurations and learning rates to further boost performance.
- **3D Segmentation**: Extend the model to handle 3D MRI data for volumetric tumor segmentation.
- **Real-time Segmentation**: Improve the model's inference speed for deployment in real-time diagnostic systems.

## Conclusion
This project showcases the ability of deep learning models, specifically U-Net, to accurately segment brain tumors from MRI scans. By automating this critical step in the diagnosis process, the model has the potential to greatly assist medical professionals, improving diagnosis speed and accuracy.

## How to Run the Project
1. Clone the repository.
2. Install required libraries using `pip install -r requirements.txt`.
3. Ensure the dataset is downloaded from Kaggle.
4. Run the Jupyter notebook or Python script to train and evaluate the model.
