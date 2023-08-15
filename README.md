# Image Action Recognition with Stanford40 Dataset and GUI


This project is an Image Action Recognition system that uses the Stanford40 action dataset. It allows users to interact with the system through a Graphical User Interface (GUI) to recognize actions performed in images and we use VIT model as the base model. The Stanford40 dataset contains images of 40 different human action classes, such as "applauding", "fishing", "holding an umbrella" etc.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Demo Images](#demo-images)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Accuracy](#accuracy)
- [Model](#model)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [Contact](#contact)

## Features

- Recognize actions in images using pre-trained models.
- Interact with the system through a user-friendly GUI.
- Display the recognized action class along with the input image.

## Requirements

- Python 3.x
- Tkinter library for GUI (usually comes pre-installed with Python)

## Installation

1. Clone the repository to your local machine:
```bash
$ git clone https://github.com/ARHPA/Image-Action-Recognition.git
```
2. Navigate to the project directory:
```bash    
$ cd Image-Action-Recognition
```
3. Install the required dependencies:
```bash   
$ pip install -r requirements.txt
```

## Demo Images

![sample Image](demo_images/Screenshot%20from%202023-07-22%2021-36-04.png)

![sample Image](demo_images/Screenshot%20from%202023-07-22%2021-36-27.png)

![sample Image](demo_images/Screenshot%20from%202023-07-22%2021-36-43.png)

![sample Image](demo_images/Screenshot%20from%202023-07-22%2021-36-59.png)

![sample Image](demo_images/Screenshot%20from%202023-07-22%2021-38-17.png)

## Usage

1. Run the GUI application:
    python GUI.py


2. The GUI will open, allowing you to interact with the system.

3. To analyze an image, click the "Select Image" button and choose an image from your local system.

4. Click the "Analyze image" button to initiate the recognition process.

5. The system will display the recognized action class below the image.

6. Repeat steps 3 to 5 for analyzing more images.

## Training the Model
If you want to train this model on your own, follow these instructions (after Installation):

1. Download model weights:
```bash
$ wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16.npz
```
2. Download and unzip dataset
```bash    
$ wget http://vision.stanford.edu/Datasets/Stanford40.zip
$ unzip Stanford40.zip
```
3. Now you can change config.json and then run train.py file:
```bash   
$ python train.py
```
Alternatively, you can simply run the train.ipynb notebook

## Accuracy

During the training of the image action recognition model, the model achieved the following accuracy on the test set:

- Test Accuracy: 82% 
- Top 2 Accuracy: 90%

These accuracy metrics demonstrate the effectiveness of the trained model in recognizing human action classes in images from the Stanford40 dataset.

## Model

The image action recognition model used in this project is based on the Vision Transformer [(VIT)](https://arxiv.org/pdf/2010.11929.pdf) architecture. The VIT model is a state-of-the-art deep learning model for image classification tasks.

The VIT model implementation used in this project is based on the repository [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) by jeonsworld. The pre-trained VIT model provided in this repository is fine-tuned on the Stanford40 dataset for action recognition.

For more details on the architecture and implementation of the VIT model, please refer to [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) repository.

## Project Structure

The project follows the structure template provided by [pytorch-template](https://github.com/victoresque/pytorch-template), which provides a well-organized and scalable project structure for deep learning projects. Below is a brief overview of the project structure:

### Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── GUI.py - Graphical User Interface of project 
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── train.ipynb - train notebook
  ├── model_best.pth - best model state dicts
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - create dataset and data loader
  │   └── data_loaders.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```
## Dataset

The Stanford40 dataset used in this project contains images of human action classes and their corresponding annotations. You can find more information and access the dataset from the official Stanford40 website: [Stanford40 Dataset](http://vision.stanford.edu/Datasets/40actions.html).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Stanford40 dataset was created and made publicly available by the Stanford Vision Lab.
- The VIT model used in this project were developed by [jeonsworld](https://github.com/jeonsworld/ViT-pytorch).

## Contributing

Contributions are welcome! If you find any issues or want to enhance the project, feel free to submit a pull request.

## Contact

For any inquiries or feedback, please contact [ARHPA00@gmail.com](mailto:ARHPA00@gmail.com).



