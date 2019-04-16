#FPO.py

## Prerequisites
Python 3.6.x

### Install pip3 and major libraries
https://linoxide.com/linux-how-to/install-flask-python-ubuntu/

```
sudo apt-get update
sudo apt-get install python3-pip python3-dev
sudo apt-get install python3.6-venv
```
### Setup

Create a virtual environment to store this project requirements
```
python3 -m venv env
```

This will install a local copy of Python and pip into a directory called flasksupportserverenv within your project directory.
Before we install applications within the virtual environment, we need to activate it. You can do so by typing:

```
source env/bin/activate
```

###Install dependencies
```
pip3 install -r requirements.txt
```

### Running the virtual environment
```
source env/bin/activate
```

### Doc
https://pdoc3.github.io/pdoc/doc/pdoc/

Update doc
```
pdoc --html src/fpo.py --overwrite
```
### Running the test


## Deployment



## References
https://github.com/mitmproxy/pdoc
https://docs.python.org/3.7/library/index.html
https://pypi.org/
http://flask.pocoo.org/docs/1.0/tutorial/layout/
https://docs.python.org/3.6/howto/functional.html
https://www.oreilly.com/ideas/2-great-benefits-of-python-generators-and-how-they-changed-me-forever
https://stackoverflow.com/questions/102535/what-can-you-use-python-generator-functions-for
https://data-flair.training/blogs/advantages-and-disadvantages-of-python/
https://github.com/getify/FPO/blob/master/docs/core-API.md#fpocomplement
