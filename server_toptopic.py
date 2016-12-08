"""Runs a user interface for the interactive anchor words algorithm"""

import os
import uuid
import threading
import pickle
import random

import flask


APP = flask.Flask(__name__, static_url_path='')
# users must label REQUIRED_DOCS documents
REQUIRED_DOCS = 80
SWITCH_COUNT = 4
FILEDICT_PICKLE = 'filedict.pickle'
TOPTOPIC_PICKLE = 'toptopic.pickle'


def get_user_dict_on_start():
    """Loads user data"""
    # This maintains state if the server crashes
    try:
        last_state = open('last_state.pickle', 'rb')
    except IOError:
        print('No last_state.pickle file, assuming no previous state')
    else:
        state = pickle.load(last_state)
        print("Last state: " + str(state))
        last_state.close()
        return state['USER_DICT']
    # but if the server is starting fresh, so does the user data
    return {}


def grab_pickle(filename):
    """Load pickle"""
    with open(filename, 'rb') as ifh:
        return pickle.load(ifh)


################################################################################
# Everything in this block needs to be run at server startup
# USER_DICT holds information on users
USER_DICT = get_user_dict_on_start()
LOCK = threading.Lock()
RNG = random.Random()
# FILEDICT is a docnumber to document dictionary
FILEDICT = grab_pickle(FILEDICT_PICKLE)
# TOPTOPIC is a list of lists of (topic prevalence, docnumber); the outer list
# is by topic number; the inner list is sorted by prevalence of topic, high to
# low; note that docnumber is a string
TOPTOPIC = grab_pickle(TOPTOPIC_PICKLE)
NUM_TOPICS = len(TOPTOPIC)
################################################################################


def save_state():
    """Saves the state of the server to a pickle file"""
    last_state = {}
    last_state['USER_DICT'] = USER_DICT
    print(USER_DICT)
    pickle.dump(last_state, open('last_state.pickle', 'wb'))


def update_topic(user_id):
    """Force context switch if appropriate"""
    completed = USER_DICT[user_id]['completed']
    cumul = USER_DICT[user_id]['cumul']
    pos = USER_DICT[user_id]['pos']
    if completed < REQUIRED_DOCS and completed == cumul[pos]:
        # move to new topic (context switch)
        pos += 1
        topic = RNG.randrange(NUM_TOPICS)
        different = USER_DICT[user_id]['different']
        already = USER_DICT[user_id]['already']
        while len(TOPTOPIC[topic]) < different[pos] or topic in already:
            topic = RNG.randrange(NUM_TOPICS)
        with LOCK:
            # update USER_DICT
            cur = RNG.randrange(len(TOPTOPIC[topic]))
            USER_DICT[user_id]['pos'] = pos
            USER_DICT[user_id]['cur'] = cur
            USER_DICT[user_id]['topic'] = topic
            USER_DICT[user_id]['already'][topic] = {cur: True}


def get_doc_number(user_id):
    """Get document number for user

    Note that the returned value is actually a string identifier for the
    document; the naming was chosen since the original corpus we used named its
    documents by numbers
    """
    return TOPTOPIC[USER_DICT[user_id]['topic']][USER_DICT[user_id]['cur']][1]


def get_doc_info(user_id):
    """Grab document info based on USER_DICT"""
    completed = USER_DICT[user_id]['completed']
    correct = USER_DICT[user_id]['correct']
    if completed >= REQUIRED_DOCS:
        return 0, '', completed, correct
    update_topic(user_id)
    doc_number = get_doc_number(user_id)
    document = FILEDICT[doc_number]['text']
    completed = USER_DICT[user_id]['completed']
    correct = USER_DICT[user_id]['correct']
    return doc_number, document, completed, correct


@APP.route('/get_doc')
def get_doc():
    """Gets the current document for whoever is asking"""
    user_id = flask.request.headers.get('uuid')
    if user_id in USER_DICT:
        doc_number, document, completed, correct = get_doc_info(user_id)
        cma = USER_DICT[user_id]['cma']
    print(doc_number, len(document), completed, correct)
    print(document)
    print(cma)
    # Return the document
    return flask.jsonify(
        document=document,
        doc_number=doc_number,
        completed=completed,
        correct=correct,
        cma=cma)


@APP.route('/')
def serve_landing_page():
    """Serves the landing page for the Active Topic Modeling UI"""
    return flask.send_from_directory('static', 'index.html')


@APP.route('/docs.html')
def serve_ui():
    """Serves the Active Topic Modeling UI"""
    return flask.send_from_directory('static', 'docs.html')


@APP.route('/scripts/script.js')
def serve_ui_js():
    """Serves the Javascript for the Active TM UI"""
    return flask.send_from_directory('static/scripts', 'script.js')


@APP.route('/end.html')
def serve_end_page():
    """Serves the end page for the Active TM UI"""
    return flask.send_from_directory('static', 'end.html')


@APP.route('/finalize')
def finalize():
    """Serves final statistics for the given user
    and erases the user from the database
    """
    user_id = flask.request.headers.get('uuid')
    cma = float('inf')
    correct = 0
    completed = -1
    with LOCK:
        if user_id in USER_DICT:
            cma = USER_DICT[user_id]['cma']
            completed = USER_DICT[user_id]['completed']
            correct = USER_DICT[user_id]['correct']
            del USER_DICT[user_id]
            save_state()
    return flask.jsonify(cma=cma, completed=completed, correct=correct)


@APP.route('/scripts/end.js')
def serve_end_js():
    """Serves the Javascript for the end page for the Active TM UI"""
    return flask.send_from_directory('static/scripts', 'end.js')


@APP.route('/scripts/js.cookie.js')
def serve_cookie_script():
    """Serves the Javascript cookie script"""
    return flask.send_from_directory('static/scripts', 'js.cookie.js')


@APP.route('/stylesheets/style.css')
def serve_ui_css():
    """Serves the CSS file for the Active TM UI"""
    return flask.send_from_directory('static/stylesheets', 'style.css')


def cumsum(numbers):
    """Return cumulative sum of numbers per list position"""
    result = []
    total = 0
    for num in numbers:
        total += num
        result.append(total)
    return result


@APP.route('/uuid')
def get_uid():
    """Sends a UUID to the client"""
    uid = uuid.uuid4()
    data = {'id': uid}
    # there are SWITCH_COUNT+1 partitions for SWITCH_COUNT context changes
    different = [round(REQUIRED_DOCS / (SWITCH_COUNT + 1))] * (SWITCH_COUNT + 1)
    cumul = cumsum(different)
    topic = RNG.randrange(NUM_TOPICS)
    pos = 0
    while len(TOPTOPIC[topic]) < different[pos]:
        topic = RNG.randrange(NUM_TOPICS)
    cur = RNG.randrange(len(TOPTOPIC[topic]))
    already = {topic: {cur: True}}
    with LOCK:
        USER_DICT[str(uid)] = {
            'completed': 0,
            'correct': 0,
            'cma': 0.0,
            'cur': cur,
            'different': different,
            'cumul': cumul,
            'pos': pos,
            'topic': topic,
            'already': already}
    user_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/userData"
    file_to_open = user_data_dir+"/"+str(uid)+".data"
    with open(file_to_open, 'a') as user_file:
        user_file.write("#TopTopics\n")
    save_state()
    return flask.jsonify(data)


@APP.route('/rated', methods=['POST'])
def get_rating():
    """Receives and saves a user rating to a specific user's file"""
    flask.request.get_data()
    input_json = flask.request.get_json(force=True)
    user_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/userData"
    user_id = input_json['uid']
    topic = USER_DICT[user_id]['topic']
    doc_number = input_json['doc_number']
    guess = input_json['rating']
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)
    file_to_open = user_data_dir+"/"+user_id+".data"
    with open(file_to_open, 'a') as user_file:
        user_file.write(
            str(input_json['start_time']) + '\t' + str(input_json['end_time']) +
            '\t' + str(topic) + '\t' + str(doc_number) +
            '\t' + str(guess) + '\n')
    prevlabel = FILEDICT[doc_number]['label']
    completed = 0
    with LOCK:
        if user_id in USER_DICT:
            # update user progress
            USER_DICT[user_id]['completed'] += 1
            completed = USER_DICT[user_id]['completed']
            cur = RNG.randrange(len(TOPTOPIC[topic]))
            already = USER_DICT[user_id]['already']
            while cur in already[topic]:
                cur = RNG.randrange(len(TOPTOPIC[topic]))
            USER_DICT[user_id]['already'][topic][cur] = True
            USER_DICT[user_id]['cur'] = cur
            # cumulative moving average
            diff = abs(guess - prevlabel)
            cma = USER_DICT[user_id]['cma']
            USER_DICT[user_id]['cma'] += (diff - cma) / completed
            cma = USER_DICT[user_id]['cma']
            if diff == 0:
                USER_DICT[user_id]['correct'] += 1
    # Save state (in case the server crashes)
    correct = USER_DICT[user_id]['correct']
    save_state()
    return flask.jsonify(
        label=prevlabel,
        completed=completed,
        correct=correct,
        cma=cma)


if __name__ == '__main__':
    APP.run(debug=True,
            host='0.0.0.0',
            port=3000)
