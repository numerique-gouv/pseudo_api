
# Pseudo API
This API is part of the document's pseudonymization effort lead at [Etalab's](https://www.etalab.gouv.fr/) [Lab IA](https://github.com/etalab-ia/).  Other Lab IA projects can be found at the [Lab IA](https://github.com/etalab-ia).

#### Project Status: [Active]

## Intro/Objectives

The purpose of this repo is to provide an API endpoint to the pseudonymize documents. The API should make it easy to developers to automoate document pseudonnymization with their own models.

The larger goal of the pseudonymization project is to help the French Counsil of State open their court decisions to the general public, as required by law. More info about pseudonymization and this project can be found in our French pseudonymization guide [here](https://guides.etalab.gouv.fr/pseudonymisation/). Our API uses a Named Entity Recognition model to find and replace **first names, last names, and addresses** in court decisions (specifically those of the Counsil of State). 

You need to train a NER model with the [Flair library](https://github.com/flairNLP/flair). Unfortunately, currently we cannot share our model nor the data it was trained on as it contains non-public information.

### Methods Used
* Natural Language Processing: Information Extraction : Named Entity Recognition
* Natural Language Processing: Language Modelling / Feature Learning: Word embeddings
* Machine Learning: Deep Learning: Recurrent Networks: BiLSTM+CRF

### Technologies
* Python
* Flair
* Flask, gunicorn, nginx
* SQLite
* Pandas
* Docker

## API Description

The API has two endpoints:

### 1. Pseudonymization 

Analyzes a given string. The output consists of the pseudonymized text, and (optionaly) an enriched version of the original text where extracted entities have ben wrapped in tags <PER> (for "person"), <LOC> (for places), <ORG> (for organizations)  

**URL** : `/`

**Method** : `POST`

**Data example** All fields must be sent.

```json
{
    "text": "M. Pierre Sailly demeurant au 14 rue de la Felicité, 75007 Vienne.",
}
```

#### Success Response

**Condition** : If everything is OK and the model inference was performed correctly

**Code** : `200 OK`

**Content example**

```json
{
    "success": true,
    "pseudo": "M. A. demeurant au 14 rue B. "
}
```

**URL** : `/tags/`

**Method** : `POST`

**Data example** All fields must be sent.

```json
{
    "text": "M. Pierre Sailly demeurant au 14 rue de la Felicité, 75007 Vienne.",
}
```

#### Success Response

**Condition** : If everything is OK and the model inference was performed correctly

**Code** : `200 OK`

**Content example**

```json
{
    "success": true,
    "pseudo": "M. A. demeurant au 14 rue B. ",
    "tag": "<text><sentence><PER>M. Pierre Sailly</PER> demeurant au <LOC>14 rue de la Felicité, 75007 Vienne</LOC></sentence></text>.",

}
```


## Getting Started
The easiest way to test this application is by using Docker and Docker Compose.

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Set the environment variable `PSEUDO_MODEL_PATH` in the `.env` file.
3. Launch the wrapper bash file `run_docker.sh`. This file will clean and rebuild the required Docker containers by calling `docker-compose.yml`.
4. Access the API at `localhost/` and `localhost/api_stats`.

## Project Deliverables
* [Pseudonymization Demo](https://github.com/etalab-ia/pseudo_app)
* This API
* [Pseudonymization Guide](https://guides.etalab.gouv.fr/pseudonymisation/)


## Contact
* Feel free to contact [@psorianom](https://github.com/psorianom/) or other [Lab IA](https://github.com/etalab-ia/) team members with any questions or if you are interested in contributing!
