# UPC-NLP-Neural-NERC

The goal of this project is to build and analyze a language classifying pipeline
in order to gain insight and familiarity to typical Natural Language Processing (NLP) tools and strategies.


## HOW TO RUN

1) In CMD move to *this_dir*/source
2) For the best performing model run:
	python langdetect.py -i "..\data\dataset.csv" -v 1000 -a "word"
-------------------------------------------------------------------------------------------------------
Note:
- The -v (vocabulary size) is a modifiable parameter and -a can be set to 'word' or 'char' granularity
- You need Python3 with a selection of packages like ntlk, Sklearn and pandas. Look at the error code given
by the cmd to know what to install using pip. 
