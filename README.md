# balance_metabolic_syndrome
Проект, прогнозирующий наличие метаболического синдрома на основе информации о пациенте
This project demonstrates how to present machine learning solution as a web application using Streamlit framework. The data used in this repo is the Titanic dataset from Kaggle.

Try app here!

Files
app.py: streamlit app file
model.py: script for generating the Random Forest classifier model
titanic.csv and model_weights.mw: data file and pre-trained model
requirements.txt: package requirements files
Dockerfile for docker deployment
Run Demo Locally
Shell
For directly run streamlit locally in the repo root folder as follows:

$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py
Open http://localhost:8501 to view the app.

Docker
For build and run the docker image named st-demo:

$ docker build -t st-demo .
$ docker run -it --rm -p '8501:8501' st-demo
-it keeps the terminal interactive

--rm removes the image once the command is stopped (e.g. using control + c)

Open http://localhost:8501/ to view the app.

Streamlit Cloud Deployment
Put your app on GitHub (like this repo) Make sure it's in a public folder and that you have a requirements.txt file.

Sign into Streamlit Cloud Sign into share.streamlit.io with your GitHub email address, you need to have access to Streamlit Cloud service.

Deploy and share!
Click "New app", then fill in your repo, branch, and file path, choose a Python version (3.9 for this demo) and click "Deploy", then you should be able to see your app.

https://metabolicsyndrome.streamlit.app/
