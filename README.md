# Inversion Course Practicals

This is a part of the inversion course. You can either run the notebooks from [Colab](https://colab.research.google.com/) or run them locally in your own machine.

## Table of contents

- [Run the notebooks from Colab](README.md#run-the-notebooks-from-colab)
- [Run the notebooks locally](README.md#run-the-notebooks-locally)
    - [Get the dependencies](README.md#1-get-the-dependencies)
    - [Get the notebooks](README.md#2-get-the-notebooks)
    - [Run the notebooks via JupyterLab](README.md#3-run-the-notebooks-via-jupyterlab)

## Run the notebooks from Colab

1. From GitHub, navigate to the notebook you'd like to run
2. Click on the "Open on Colab" badge at the top of the notebook
3. Within Colab, uncomment the code block with  `# !git clone https://github.com/msambridge/InversionPracticals` and run this block
4. Run the rest of the notebook as normal
5. There are several options if you'd like to save your changes. To do this, click "File" (on top left of Colab page) -> "Save a copy...", or "Download".

## Run the notebooks locally

### 1. Get the dependencies

Here are detailed
instructions on how to get all the dependencies.

#### 1.1. Set up a python virtual environment [optional]

It's recommended to use a python virtual environment (e.g. [`venv`](https://docs.python.org/3/library/venv.html), [`virtualenv`](https://virtualenv.pypa.io/en/latest/), [`mamba`](https://mamba.readthedocs.io/en/latest/) or [`conda`](https://docs.conda.io/en/latest/)) so that installation of dependendcies doesn't conflict with your other Python projects. 

Open a terminal (or a Cygwin shell for Windows users) and refer to the cheat sheet below for how to create, activate, exit and remove a virtual environment. `$ ` is the system prompt.

<details>
  <summary>venv</summary>

  Ensure you have and are using *python >= 3.6*. It may not be called `python` but something like `python3`, `python3.10` etc.

  Use the first two lines below to create and activate the new virtual environment. The other lines are for your
  future reference.

  ```console
  $ python -m venv <path-to-new-env>/inversion_course           # to create
  $ source <path-to-new-env>/inversion_course/bin/activate      # to activate
  $ deactivate                                                  # to exit
  $ rm -rf <path-to-new-env>/inversion_course                   # to remove
  ```
  
</details>

<details>
  <summary>virtualenv</summary>

  Use the first two lines below to create and activate the new virtual environment. The other lines are for your
  future reference.

  ```console
  $ virtualenv <path-to-new-env>/inversion_course -p=3.10       # to create
  $ source <path-to-new-env>/inversion_course/bin/activate      # to activate
  $ deactivate                                                  # to exit
  $ rm -rf <path-to-new-env>/inversion_course                   # to remove
  ```

</details>

<details>
  <summary>mamba</summary>

  Use the first two lines below to create and activate the new virtual environment. The other lines are for your
  future reference.

  ```console
  $ mamba create -n inversion_course python=3.10                # to create
  $ mamba activate inversion_course                             # to activate
  $ mamba deactivate                                            # to exit
  $ mamba env remove -n inversion_course                        # to remove
  ```

</details>

<details>
  <summary>conda</summary>

  Use the first two lines below to create and activate the new virtual environment. The other lines are for your
  future reference.

  ```console
  $ conda create -n inversion_course python=3.10                # to create
  $ conda activate inversion_course                             # to activate
  $ conda deactivate                                            # to exit
  $ conda env remove -n inversion_course                        # to remove
  ```

</details>


#### 1.3. Installation

Type the following in your terminal (or Cygwin shell for Windows users):

```console
$ pip install jupyterlab matplotlib numpy scipy pyrf96 seaborn tqdm
```

</details>

<details>
  <summary>Troubleshooting installation</summary>

  If you run into an error while running the above command, try the following:

  ```console
  $ pip install jupyterlab matplotlib seaborn tqdm
  $ pip install pyrf96
  ```

  If you see any error while running the command, try to search for the error you see
  on Google/DuckDuckGo/ChatGPT.

  If you see an error while running the second command, (ctrl/cmd + f) search `error` from
  your terminal history to locate which error is causing the installation failure.

</details>

### 2. Get the notebooks

Type the following in your terminal (or Cygwin shell for Windows users):

```bash
$ cd <path-where-you-want-it-to-be-downloaded>
$ git clone https://github.com/msambridge/InversionPracticals.git
```

### 3. Run the notebooks via JupyterLab

`cd` (change directory) into the path where this repository was downloaded and run `jupyter-lab`:

```bash
$ cd <path-to-practicals>/JupyterPracticals
$ jupyter-lab
```

Wait for a while and your browser will be opened up automatically with a web-based IDE.
