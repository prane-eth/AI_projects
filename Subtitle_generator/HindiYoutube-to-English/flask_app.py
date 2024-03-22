' Web app which detects Brute force attacks '

import os, threading, time, sys
from flask import *
# from my_main import transcript, download_video

app = Flask(__name__)
app.secret_key = 'my_secret_key_123'
sys.stdout = open('output.txt', 'a')


class var:
    link = ''
    html_code = '''
        <title> Youtube to English </title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <div align="center" class="border"> 
            <div class="header">
                <h1 class="word"> Youtube to English </h1>
                <br>
            </div>
            <h2 class="word"> 
                <form action="/" method="post"> 
                <input id="link" name="link" type="text" placeholder="Enter YouTube link" class="textbox"> </br> </br> 
                <input type="submit" class="btn btn-primary" value="Translate">
                </form> 
                <div class="msg"> {{ msg }} </div> 
            </h2> 
        </div>
        Last translated: 
        <a href='./last_translated/'> Click here </a> <br>
        <a href='./output/'> Code output </a> <br>
        <a href='https://translate.google.com/?sl=hi&tl=en'> Google Translator </a> <br>
        <a href='http://koobo.ml/reader/'> Reader </a> <br>
    '''


# https://stackoverflow.com/questions/59850517/how-to-run-background-tasks-in-python
class BackgroundTask(threading.Thread):
    def run(self, *args, **kwargs):
        os.system('python3 my_main.py ' + var.link)


@app.route('/', methods = ['POST', 'GET'])
def home():
    msg = ''
    txt = '-'
    if request.method == 'POST':
        link = request.form['link']
        msg = 'Link added to queue'
        var.link = link
        t = BackgroundTask()
        t.start()
    #
    return render_template_string(var.html_code, msg=msg)


@app.route('/last_translated/')
def last_translated():
    filename = 'last_translated.txt'
    output1 = ''
    with open(filename) as file:
        output1 = file.read()
    output1 = output1.replace('\n', '<br>')
    return render_template_string(output1)


@app.route('/output/')
def output():
    filename = 'output.txt'
    output1 = ''
    with open(filename) as file:
        output1 = file.read()
    output1 = output1.replace('\n', '<br>')
    return render_template_string(output1)


if __name__ == "__main__":
    app.run(port=8080)

