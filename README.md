# Überblick: Code zur Veröffentlichung "Sprachkommandoerkennung auf einem FPGA"

## Zusammenfassung

Wir untersuchen am Beispiel einer Sprachkommandoerkennung, wie man ein neuronales Netz in einer stark leistungsbeschränkten Umgebung am besten nutzen kann, indem man ein feldprogrammierbares Gate-Array (FPGA) einsetzt. Das im vorliegenden Bericht beschriebene Projekt baut auf Projekten in früheren Semestern auf, und es wird fortgesetzt werden. Im aktuellen Semester wurden die nötigen Verarbeitungsschritte an neue Audio-Hardware angepasst sowie detaillierter identifiziert und festgelegt. Aus einem aufgetretenen Schnittstellenproblem heraus ergab sich eine neue Einsicht zu einer Systemarchitektur für eine effiziente Verarbeitung auf einem FPGA. Bei den einzelnen Teilaufgaben wurden ansonsten bedauerlicherweise keine wesentlichen Fortschritte gegenüber dem Vorgängerprojekt erzielt. Im Ausblick am Ende fassen wir den aktuellen Stand der Ideen zusammen, wie die laufende Serie von Mini-Projekten fortgesetzt werden kann.

## Links auf die Veröffentlichung zum Code

* https://homepages.on.hs-bremen.de/~jbredereke/de/forschung/veroeffentlichungen/sprachkommandoerkennung-auf-einem-fpga-wp_embeds-2025.html
* https://doi.org/10.26092/elib/3777

## Liste aller zugehörigen Repositories auf GitHub

* https://github.com/JanBredereke/wp_embeds2425_ki
* https://github.com/JanBredereke/wp_embeds2425_library
* https://github.com/JanBredereke/wp_embeds2425_pl-definition

Für eine Beschreibung siehe Anhang A auf Seite 87 der obigen Veröffentlichung.

# Anleitung für dieses Code-Repository

## Installation for Windows

First, install Python 3.10.11 from [here](https://www.python.org/downloads/release/python-31011/).

There is one external dependency you need to install before running this project:

- [ffmpeg](https://ffmpeg.org/) or [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/) for loading MP3s using PyDub.
  The `\bin` folder needs to be added to the PATH.
  Otherwise, the dependency can be installed like this:

```sh
pip install -r requirements.txt
```

```sh
ffdl install --add-path
```

The `--add-path` option adds the installed FFmpeg folder to the user's system path.
Re-open the Python window and both `ffmpeg` and `ffprobe` will be available to your program.

You might get the error message **ModuleNotFoundError: No module named 'pyaudioop'**.
The reason for this is that the `audioop` library was removed in Python 3.13, which caused the `pyaudioop` fallback to
be called, but this also gives an error.
Try to run the following command in the project directory:

```sh
pip install audioop-lts
```

If that is not working, change the import to the following one:

```python
import pydub.pyaudioop as audioop
```

## Installation for Linux

First, you will need to add the deadsnakes PPA to your system's sources list. This will allow you to install Python 3.10
```sh
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

Then, tou can install Python 3.10 and the necessary packages with the following commands:
```sh
sudo apt install python3.10 python3.10-venv python3.10-dev
python3 --version
```

You will also need to install the following packages:
```sh
sudo apt-get install ffmpeg 
```

## Run the Project

To run the project, execute the `Neural Network Model.ipynb` file in Jupyter Notebook.

## Run with an AMD GPU

If you have an AMD GPU you can use ROCm to run the project. You can install ROCm by following the instructions on the [ROCm website](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

After installing ROCm, you will will need to uninstall the tensorflow package (if it is installed):

```sh
pip uninstall tensorflow torchvision torchaudio
```

Then, you will need to follow the instructions on [PyTorch website](https://pytorch.org/) and install the Stable build for ROCm.


## Test Data

Test Data may be changed, but it needs to have the following folder structure:

```
data
  └── background
  └── commands
        └── command1
        └── command2
        └── ...
  └── negatives
```

The `data` folder needs to be located directly under `neural_network_model`.

The designation of the folder `data` may be changed. Please make sure to rename the dedicated variable in `main.py` as
well.

The folders **background** and **negatives** are necessary. It is expected that they have MP3 files. Other files will be
ignored. They may be empty.

The folder **commands** is expected to have at least one or multiple subfolders. The subfolders are expected to have MP3
files. You may name the subfolders to your liking.

## Preprocessing Data

To preprocess the data, use the `preprocess_data` function from the `preprocess_data/preprocess_data.py` file. This
function splits the samples into training and testing data and returns the number of categories and the list of
categories.

```python
from preprocess_data.preprocess_data import preprocess_data
from enums.data_classes import AudioData

samples_dict = {
    "category1": [AudioData(...), ...],
    "category2": [AudioData(...), ...],
    ...
}

train_data, test_data, number_of_categories, categories = preprocess_data(samples_dict)
```

## Models

The project includes several neural network models defined in the `models/create_and_get_models.py` file. Here are the
main classes:

- `QuantNet`
- `SequenceQuantNet`
- `PartiallyQuantizedNet`
- `AudioNet`

Each class defines a different architecture for processing audio data. You can instantiate and use these models as
follows:

```python
from models.create_and_get_models import QuantNet, SequenceQuantNet, PartiallyQuantizedNet, AudioNet

# Example instantiation
model = QuantNet(in_channels=1, num_classes=10)
```
