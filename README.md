
# Comment Toxicity Detector

### Description

This comment toxicity detector can detect if a comment pasted into its input is toxic or not. It also gives a confidence level on a scale of 0 to 1 where  0 means that the AI is confident that the comment is toxic and 1 means that the comment is confident that  
the comment is not toxic. A toxic comment in this case is defined as being hateful, violent, using exessive swearing, or other comments that may be against the terms of service for major social media sites.

### Requirements

- pip: package installer for python
- pipenv: virtualenv tool for installing dpendencies, you can download it with 
```bash
pip install pipenv
```
- python >3.10

### Other Frameworks Used

All the other frameworks used in this program can be downloaded using the pipfile. The frameworks used in this program include Pytorch, Pandas, Flask, and the 🤗 Transformers package.


![Python logo](https://www.python.org/static/img/python-logo.png)
![Pytorch logo](https://imgs.search.brave.com/pL5Cz0-sW-GQG4PMOE8b76GUmcAcpTIdxn8-tkcHPSc/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9weXRv/cmNoLm9yZy9hc3Nl/dHMvaW1hZ2VzL2xv/Z28td2hpdGUuc3Zn)
![Pandas logo](https://pandas.pydata.org/static/img/pandas_white.svg)
![Flask logo](https://flask.palletsprojects.com/en/3.0.x/_images/flask-horizontal.png)
![Huggingface logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)

### How to run this program

1. Install the code as a zip file and unzip the file  
2. run  
```bash
pipenv install  
```
to install all the dependencies  
3. In the python console run  
```python  
python toxicity_detector.py  
```  
4. Follow the link given in the console or go to localhost://7000
5. Copy a comment from any website, input it into the textbox, and press submit