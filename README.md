# Neurale Netwerk voor Cijfer- en Letterherkenning

## Overzicht van het project

Dit project betreft een neurale netwerk implementatie voor zowel cijfer- als letterherkenning. Het oorspronkelijke model is getraind op de MNIST dataset voor cijferherkenning en vervolgens uitgebreid om de EMNIST-letters dataset te gebruiken voor letterherkenning. Met behulp van dezelfde GUI kan de gebruiker handgeschreven cijfers of letters tekenen, waarna het model de getekende invoer voorspelt.

## Bestanden en mappenstructuur

-   **data (.gitignore)**
    -   **digits**
        -   `mnist-original.mat`
    -   **letters**
        -   `emnist-letters-test.csv`
        -   `emnist-letters-train.csv`
-   **digitRecognition**
    -   `GUI.py`
    -   `main.py`
-   **letterRecognition**
    -   `GUI.py`
    -   `main.py`
-   **model**
    -   **digits**
        -   `Theta1.txt`
        -   `Theta2.txt`
        -   `Theta3.txt`
    -   **letters**
        -   `Theta1.txt`
        -   `Theta2.txt`
        -   `Theta3.txt`
-   `Model.py`
-   `Prediction.py`
-   `RandInitialize.py`

## Instructies

1. **Project Setup**

    - Maak een folder genaamd `data` in de hoofdmap van het project.
    - Maak in de `data` folder twee subfolders aan genaamd `digits` en `letters`.

2. **Data Download**
    - Download `mnist-original.mat` van [Kaggle MNIST Original Dataset](https://www.kaggle.com/datasets/avnishnish/mnist-original) en plaats dit bestand in de `digits` subfolder.
    - Download `emnist-letters-test.csv` en `emnist-letters-train.csv` van [Kaggle EMNIST Dataset](https://www.kaggle.com/datasets/crawford/emnist) en plaats deze bestanden in de `letters` subfolder.

## Model Training

De Theta-bestanden zijn al opgenomen in het project, dus het is niet noodzakelijk om het model opnieuw te trainen. Als je het model echter opnieuw wilt trainen, volg dan de bovenstaande instructies voor het downloaden van de datasets en voer de `main.py` bestanden uit in de respectieve mappen (`digitRecognition` en `letterRecognition`).

## Project Uitvoeren

Om de GUI te starten voor cijferherkenning of letterherkenning, navigeer naar de respectieve map (`digitRecognition` of `letterRecognition`) en voer `GUI.py` uit. Dit opent een venster waar je cijfers of letters kunt tekenen en het model zal een voorspelling doen op basis van je invoer.

### Voorbeeld Commando's

```bash
# Voor cijferherkenning
cd digitRecognition
python GUI.py

# Voor letterherkenning
cd letterRecognition
python GUI.py
```
