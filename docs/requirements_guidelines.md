# #Issue001 : Dependency conflit while install `Appworld` and `LangChain`/`LangGraph`

## Issue Content
dependency confilt occured on `pydantic` pakage while install `Appworld` benchmark pakage and `LangChain`/`LangGraph` in same virtual environment.
- `LangChain`, `LangGraph` : `pydantic` require `2.7 <=`
- `Appworld` : `pydantic` 1.x dependency installed (when use `pip install appworld`)

## How to Solve

### Main Task
1. install `LangChain`, `LangGraph` first (install `pydantic` >= 2.12)
2. git clone `Appworld` source from github, and install appworld from local files

### Process

```
# `pydantic` >= 2.12 is installed
pip install lanchain langgraph ...

# install `git-lfs` to download large file from appworld repository
brew install git-lfs
# for `conda install -c conda-forge git-lfs`

# download source from git hub
git clone https://github.com/StonyBrookNLP/appworld; cd appworld

# install appworld pakage from local file
pip install -e .
appworld install --repo

# download data
appworld donwload data
```

