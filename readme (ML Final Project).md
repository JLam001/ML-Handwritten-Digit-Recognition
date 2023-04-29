<a name="br1"></a>James Lam

Panther ID: 6194394

Handwritten Digit Recognition

This program uses a convolutional neural network to recognize handwritten digits. The programrequires the following libraries: Numpy, TensorFlow, Tkinter, PIL, cv2, and os. Below are theinstructions to install the necessary components and run the program.

Installation Guide

Miniconda3

1\. Install Miniconda3 by following the instructions [here](https://docs.conda.io/en/latest/miniconda.html).

2\. Run the installer and follow the prompts to complete the installation.

3\. Open a new terminal window to ensure the changes take effect.

4\. Open the Anaconda Prompt and create a new environment:

(yourenvname is a placeholder name, you can name it whatever you like)

conda create --name yourenvname python=3.10

5\. Activate the new environment:

conda activate yourenvname

6\. Before proceeding, ensure that pip has been updated using the command:

pip install --upgrade pip

7\. For **Windows users**, to enable GPU support, make sure you have your graphics driver

**(NVIDIA GPU)** installed, and install the following:

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

8\. Install the required libraries:

conda install tensorflowconda install numpyconda install pillowconda install opencv




<a name="br2"></a>Visual Studio Code (VSC)

1\. Download the installer for your operating system from[ ](https://code.visualstudio.com/download)<https://code.visualstudio.com/download>.

2\. Run the installer and follow the prompts to complete the installation.

Python Extension for Visual Studio Code

1\. Open Visual Studio Code.

2\. Click on the Extensions icon on the left-hand side of the window.

3\. In the search bar, type "Python" and press Enter.

4\. Click on the "Python" extension, and then click on the "Install" button.

Select Interpreter

1\. Open VS code and enter ctrl + shift + p (for Windows) or command + p (macOS)

2\. Type Select interpreter and click on your newly created environment Python 3.10.9

(yourenvname:conda)

Usage

The program first checks if a saved model file "Handwritten\_digit\_mnist.h5" exists. If the fileexists, the program loads the saved model. If the file does not exist, the program loads theMNIST dataset, preprocesses the data, defines a convolutional neural network model, compilesthe model, trains the model, saves the trained model as "Handwritten\_digit\_mnist.h5", andevaluates the model.

Important note for macOS users:

If you downloaded miniconda3 with Python 3.10, Tkinter Canvas won’t draw properly. Whendrawing on the canvas the black strokes aren’t visible but it is being drawn and the system willregister it.
