<span style="width:100%;display:flex;justify-content:center;">![title](static/public/images/company_logo.jpg)</span>
## Compatibility: Python 3.10 and above is a currently supported version of Python
#### Setup Steps :

- ##### <span style="color:red">Step 1</span> : 
    Clone Repository using below command
    Using Https
    ```GIT
    git clone https://github.com/rahuldhamecha-infobeans/spam-ham-detection.git
    ```
    Using SSH
    ```GIT
    git clone git@github.com:rahuldhamecha-infobeans/spam-ham-detection.git
    ```
- ##### Step 2 : 
    Now execute  command one by one to install models and remaining dependencies
    ```shell
    pip install git+https://github.com/openai/whisper.git
    ```
    ```shell
    sudo apt update && sudo apt install ffmpeg
    ```
    ```shell
    conda install mysqlclient
    ```
    ```shell
    pip install mysql-connector-python
    ```
    ```shell
    pip install kaleido
    ```
- ##### Step 3 : 
    Now install all required dependencies using below command
    ```shell
    pip install -r requirement.txt
    ```
  
- ##### Step 4 : 
    Now Change the Config in app_config folder 
    For Sign in with Google
    ```python
    def google_config():
        return {
            'GOOGLE_CLIENT_ID': 'xxxxxxxxxxxxxxxxxxxxxxx',
            'GOOGLE_CLIENT_SECRET': 'xxxxxxxxxxxxxxxxxxxx',
            'GOOGLE_ENV_LOCAL': True,
            'GOOGLE_SCOPE': ['profile', 'email'],
        }
    ```
    For Email Configuration
    ```python
    def mail_config():
        return {
            'MAIL_SERVER': 'smtp.gmail.com',
            'MAIL_PORT': 587,
            'MAIL_USERNAME': 'xxxxxxxxxx',
            'MAIL_PASSWORD': 'xxxxxxxxxx',
            'MAIL_USE_TLS': True,
            'MAIL_USE_SSL': False,
        }
    ```
    For MySql Database configuration
    ```python
    def database_config():
        return {
            'USERNAME': 'xxxx',
            'PASSWORD': 'xxxxxx',
            'HOST': 'localhost',
            'DATABASE': 'xxxxxx'
        }
    ```

- ##### Step 5 : 
    Now set the flask app file to FLASK_APP path
    ```shell
    export FLASK_APP=app.py
    ```
    run to enable debug mode
    ```shell
    export FLASK_DEBUG=1
    ```
- #### Step 6:
  Run the Database migration command
  ```shell
  flask db init
  ```  
  ```shell
  flask db migrate
  ```  
  ```shell
  flask db upgrade
  ```    
  ```shell
  flask migrate-data
  ```  

- ##### Step 6 : 
    Now run the flask application using below command
    ```shell
    flask run
    ```

