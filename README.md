# Build and launching the API 

To run the api, you have 2 options : 

1. Python  

2. Docker 

## Python   

In the terminal, use ` pip install numpy flask flask-restful ` or `pip install -r requirements.txt` to install the dependecies. This is only needed once. 

Then, issue the following commands in this order: 
- To move to the project folder : `cd path/to/project_folder`
- To launch the API : `python main.py`

## Docker 
In the terminal, issue the following commands in this order: 

- To build the API (only needed once ):  `docker build -t spaas .` 
- To launch the API : `docker run -p 8888:8888 spaas`



