"""For user study:  random vs. topic ordering"""

import os
import uuid
import threading
import pickle
import random

import flask


APP = flask.Flask(__name__, static_url_path='')
# users must label REQUIRED_DOCS documents
REQUIRED_DOCS = 90
DOCS_PER_TREATMENTS = 40
# have user label some number of documents to get used to task but don't count
# them when running statistical tests
LEARN_IGNORE = REQUIRED_DOCS - (2 * DOCS_PER_TREATMENTS)
HALF_LEARN_IGNORE = int(LEARN_IGNORE / 2)
FILEDICT_PICKLE = 'filedict.pickle'
TOPTOPIC_PICKLE = 'toptopic.pickle'
TOPICDISTANCES_PICKLE = 'topicdistances.pickle'
LAST_STATE_PICKLE = 'last_state_compare.pickle'
RESULTS_DIR = 'compareData'
SORTED_TEXT = 'sorted_order.txt'


def reload_state():
    """Loads user data"""
    # This maintains state if the server crashes
    try:
        last_state = open(LAST_STATE_PICKLE, 'rb')
    except IOError:
        print('No ' + LAST_STATE_PICKLE + ' file, assuming no previous state')
    else:
        state = pickle.load(last_state)
        print("Last state: " + str(state))
        last_state.close()
        return state['USER_DICT'], state['SERVED']
    # the server is starting fresh
    return {}, 0


def grab_pickle(filename):
    """Loads pickle"""
    with open(filename, 'rb') as ifh:
        return pickle.load(ifh)


def get_doc2topic(toptopic):
    """Associates docnum with its top topic"""
    result = {}
    for i, docs in enumerate(toptopic):
        for (prevalence, docnumber) in docs:
            result[docnumber] = (i, prevalence)
    return result


def get_topic(docnum):
    """Gets (top topic, topic prevalence) for docnum"""
    return DOC2TOPIC[docnum]


def _get_topic_order(topic_starts):
    """Generates topic order with random starting point and greedy minimum
    divergence according to what topics are available"""
    result = []
    remaining = {}
    for topic in topic_starts:
        remaining[topic] = True
    start = DOCRNG.choice(sorted([topic for topic in topic_starts]))
    result.append(start)
    print(result)
    del remaining[start]
    while remaining:
        sorted_indices = [
            b[0]
            for b in
            sorted(
                enumerate(TOPICDISTANCES[result[-1]]),
                key=lambda i:i[1])]
        for index in sorted_indices:
            if index in remaining:
                result.append(index)
                del remaining[index]
                break
    return result


def _get_topic_starts(docnums):
    """Gets indices for where a topic cluster starts in a list of top topic
    sorted document numbers
    """
    result = {}
    for i, docnum in enumerate(docnums):
        topic = get_topic(docnum)[0]
        if topic not in result:
            result[topic] = i
    return result


def order_docs(docs, group_size):
    """For each group_size chunk of documents, sort by top topic; then sort
    topic clusters by topic similarity
    """
    result = []
    done = 0
    while done + group_size < len(docs):
        tmp = sorted(
            docs[done:done+group_size],
            key=get_topic,
            reverse=True)
        topic_counts = {}
        for docnum in tmp:
            topic = get_topic(docnum)[0]
            if topic not in topic_counts:
                topic_counts[topic] = 1
            else:
                topic_counts[topic] += 1
        topic_starts = _get_topic_starts(tmp)
        topic_order = _get_topic_order(topic_starts)
        for topic in topic_order:
            start = topic_starts[topic]
            result.extend(tmp[start:start+topic_counts[topic]])
        done += group_size
    if done < len(docs):
        # sort remaining documents
        result.extend(sorted(
            docs[done:],
            key=get_topic,
            reverse=True))
    with open(SORTED_TEXT, 'w') as ofh:
        for docnum in result:
            ofh.write(str(get_topic(docnum)) + '\t' + docnum + '\n')
    return result


###############################################################################
# Everything in this block needs to be run at server startup
# USER_DICT holds information on users; SERVED is the number of users served
# thus far
USER_DICT, SERVED = reload_state()
LOCK = threading.Lock()
RNG = random.Random()
# FILEDICT is a docnumber to document dictionary
FILEDICT = grab_pickle(FILEDICT_PICKLE)
# TOPTOPIC is a list of lists of (topic prevalence, docnumber); the outer list
# is by topic number; the inner list is sorted by prevalence of topic, high to
# low; note that docnumber is a string
TOPTOPIC = grab_pickle(TOPTOPIC_PICKLE)
TOPICDISTANCES = grab_pickle(TOPICDISTANCES_PICKLE)
# DOC2TOPIC is a dictionary of {docnumber: (top topic, topic prevalence)};
# again, docnumber is a string
DOC2TOPIC = get_doc2topic(TOPTOPIC)
# Make sure that if the server goes down, we can recover which documents have
# already been assigned for labeling
DOCRNG = random.Random(531)
DOC_NUMS = sorted([d for d in DOC2TOPIC])
DOCRNG.shuffle(DOC_NUMS)
HALF_DOCS_COUNT = int(len(DOC_NUMS)/2)
# GROUP_SIZE must be cleverly chosen such that
# GROUP_SIZE % DOCS_PER_TREATMENTS == 0
GROUP_SIZE = 1000
USERS_PER_GROUP = int(GROUP_SIZE / DOCS_PER_TREATMENTS)
ORDEREDS = order_docs(DOC_NUMS[:HALF_DOCS_COUNT], GROUP_SIZE)
RANDOMS = DOC_NUMS[HALF_DOCS_COUNT:]
NUM_TOPICS = len(TOPTOPIC)
###############################################################################


def save_state():
    """Saves the state of the server to a pickle file"""
    last_state = {}
    last_state['USER_DICT'] = USER_DICT
    last_state['SERVED'] = SERVED
    print(USER_DICT)
    with open(LAST_STATE_PICKLE, 'wb') as ofh:
        pickle.dump(last_state, ofh)


def get_doc_info(user_id):
    """Grab document info based on USER_DICT"""
    completed = USER_DICT[user_id]['completed']
    correct = USER_DICT[user_id]['correct']
    if completed >= REQUIRED_DOCS:
        return 0, '', completed, correct
    if completed >= DOCS_PER_TREATMENTS:
        doc_number = \
            USER_DICT[user_id]['second'][
                completed-(DOCS_PER_TREATMENTS+HALF_LEARN_IGNORE)]
    else:
        doc_number = USER_DICT[user_id]['first'][completed]
    document = FILEDICT[doc_number]['text']
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
    global SERVED
    global ORDEREDS
    global RANDOMS
    uid = uuid.uuid4()
    data = {'id': uid}
    with LOCK:
        mynum = SERVED
        random_first = mynum % 2 == 0
        mystart = mynum*DOCS_PER_TREATMENTS
        randoms = list(RANDOMS[-HALF_LEARN_IGNORE:])
        randoms.extend(RANDOMS[mystart:mystart+DOCS_PER_TREATMENTS])
        ordereds = list(ORDEREDS[-HALF_LEARN_IGNORE:])
        ordereds.extend(ORDEREDS[mystart:mystart+DOCS_PER_TREATMENTS])
        USER_DICT[str(uid)] = {
            'completed': 0,
            'correct': 0,
            'cma': 0.0,
            'first': randoms if random_first else ordereds,
            'second': ordereds if random_first else randoms}
        SERVED += 1
    user_data_dir = \
        os.path.join(os.path.dirname(os.path.realpath(__file__)), RESULTS_DIR)
    file_to_open = user_data_dir+"/"+str(uid)+".data"
    with open(file_to_open, 'a') as user_file:
        if random_first:
            user_file.write(
                '# ' +
                str(int(SERVED / USERS_PER_GROUP)) +
                ', Random First\n')
        else:
            user_file.write(
                '# ' +
                str(int(SERVED / USERS_PER_GROUP)) +
                ', Ordered First\n')
    save_state()
    return flask.jsonify(data)


@APP.route('/rated', methods=['POST'])
def get_rating():
    """Receives and saves a user rating to a specific user's file"""
    flask.request.get_data()
    input_json = flask.request.get_json(force=True)
    user_data_dir = \
        os.path.join(os.path.dirname(os.path.realpath(__file__)), RESULTS_DIR)
    user_id = input_json['uid']
    doc_number = input_json['doc_number']
    topic = DOC2TOPIC[doc_number][0]
    guess = input_json['rating']
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)
    file_to_open = user_data_dir+"/"+user_id+".data"
    with open(file_to_open, 'a') as user_file:
        user_file.write(
            str(input_json['start_time']) + '\t' +
            str(input_json['end_time']) +
            '\t' + str(topic) + '\t' + str(doc_number) +
            '\t' + str(guess) + '\n')
    prevlabel = FILEDICT[doc_number]['label']
    completed = 0
    with LOCK:
        if user_id in USER_DICT:
            # update user progress
            USER_DICT[user_id]['completed'] += 1
            completed = USER_DICT[user_id]['completed']
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
            port=3000,
            threaded=True)
