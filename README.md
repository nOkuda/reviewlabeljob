# reviewlabeljob

## Introduction

This code was used to conduct a user study and to analyze the collected data.

## Running the User Study Server

Note that this code runs with Python 3.

Before running the user study server, make sure that prerequisite libraries are
installed.  These are listed in `requirements.txt` and should be installable via
`pip3`.  Assuming that I had already set up and activated my virtual
environment, installation should simply be a matter of running
```
pip3 install -r requirements.txt
```

Once prerequisite libraries are installed, the server can be run with
```
python3 server_toptopic.py
```

Participant results are written to the `userData` directory.

State is saved in `last_state.pickle`.  Thus, deleting `last_state.pickle`
resets the server.  Server state includes which participants are still labeling
documents, which documents they have already labeled, and measures of
participants' labeling accuracy and error.

The default port on which the server runs is 3000.  This can be changed by
editing the last line of `server_toptopic.py`, changing the number next to
`port=` to whatever port number is desired, and then re-starting the server.

## Client-side Information

Participants connect to the server using a web browser.  Assuming that the
server is hosted and configured properly, connecting to the server should be
nothing more than directing the web browser to the appropriate URL.

The server identifies participants with a UUID, which is assigned after the
participant pushes the big green button on the initial page.  That UUID is
stored in the participant's browser as a cookie.  If for any reason, the
participant's browser's state needs to be reset, navigate the browser to the
appropriate URL, appended with `/end.html`.

## Deeper Details

### Preparing the Corpus

All of the files necessary to run the server should be included in this
repository.  If you want to generate the files yourself, you will need to
install additional libraries.  These additional libraries are listed in
`analyze/requirements.txt`, so you can install them with `pip3`.  If pip3 is not
installing the libraries correctly, either make sure that you have the latest
version of pip3 or install the libraries listed in `requirements.txt` one by
one.

In order to generate `filedict.pickle`, you will need to run
`filedict_builder.py`.  Change `FILENAME`, `LABELS`, and `PROFANITY` to the path
where your corpus, document labels, and profanity files are.  The corpus should
file should contain one document per line, where each line contains a document
identifier, a tab, and the text of the document.  The document labels file
should contain one label per line, where each line contains a document
identifier, a tab, and the document's label.  The profanity file contains any
words you would like censored from the participants' view; the file contains one
word (or phrase) per line.

In order to generate `toptopic.pickle`, you will need to run multiple scripts.
First, build a corpus pickle with `pickle_data.py`, available in the `activetm`
library.  If you do now know where your Python libraries are installed, refer to
appropriate documentation.  There are many details to the settings file required
to run this script, and they are explained in the `activetm` README, but an
example settings file can be found in `analyze/amazon.settings`.  You will have
to change the `corpus`, `labels`, and `stopwords` lines to point to the correct
locations.  Running
```
python3 pickle_data.py -h
```
should give you enough information to generate the pickled corpus.

Once the pickled corpus is generated, you will need to run
`toptopic_builder.py`, with the pickled corpus as an argument.  So if the
pickled corpus is in the same directory as `toptopic_builder.py` and is named
`label_cost_corpus.pickle`, running
```
python3 toptopic_builder.py label_cost_corpus.pickle
```
should generate `toptopic.pickle`.

### Analyzing the Results

There are more files that need to be generated in order to analyze the results.
Some of the generated files are already present in this repository, but others
are available in other repositories.

If you really want to generate the files yourself, you will need to run
`analyze/get_titles.py` to get `titles.pickle`.  You will also need to run
`simil_builder.py` to get `simil.npy`.

Then run `analyze/analyze.py` to generate various plots.  Comment and uncomment
as you see fit to get the plots you want.
