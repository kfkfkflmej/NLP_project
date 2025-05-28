# NLP_project

This is an NLP project that investigates the significance of linguistic relations, transliteration, and lemmatization in the context of the NER tagging task. We use 4 Slavic languages for our testing: Bulgarian, Slovene, Ukrainian, and Russian. Our data is gathered from the Slav-NER: the 3rd Multilingual Named Entity Challenge in Slavic languages.

## Installation

We have provided a requirements.txt that can help with the installation of the libraries necessary for running our project. 

## Project Structure

Inside the script folder, you can find all of the code created for the project. In there, we store the data for running the whole project. It is worth noting that, because of storage efficiency, we decided to lemmatize and transkiterate the data and not to include these preprocessing steps inside the code of our training pipeline. The Model_training.ipynb is the file that generated all the models used in the project. All results on our test set can be found in the span_f1_notebook.ipynb. Additionally, we have left some of the scripts we used to develop our tools, which are implemented inside the utilities_2.py.

