# Installation

Set the visualization directory as the current working directory and run the following commands:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You may exit the virtual environment by running the command `deactivate`.

# Usage

Make sure to have the virtual environment activated, then run the script by one of the following commands:
```
python3 main.py INSTANCE_BEST <path-to-instance>
python3 main.py SINGLE_SOLUTION <path-to-instance> <path-to-solution>
python3 main.py MULTI_SOLUTION <path-to-instance> <path-to-solution>
```

The first command plots the best-known solution of the given instance.

The second command shows only the solution of the given instance that you provided.

The third command compares the solution you provided with the best known on the given instance.

