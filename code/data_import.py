class example(object):
    def __init__(self, sentences, question, answer, hints):
        '''
            Object which contains relevant information for inputting into the
            model.
        '''
        self.sentences = sentences
        self.question = question
        self.answer = answer
        self.hints = hints

    def __repr__(self):
        return ("Training example: \n\t Info: %s \n\t Question: %s \n\t Answer: %s \n\t Hint: %s \n" \
                % (self.sentences, self.question, self.answer, self.hints))

# Some of the answers aren't words eg: (n,s):
# This replaces it with "north south"
def fix_directions(examples):
    directions = {'n':'north','e':'east','s':'south','w':'west'}
    for example in examples:
        dirs = example.answer.split(',')
        newdirs = [directions[d] for d in dirs]
        example.answer = " ".join(newdirs)

# Each Set consists of several lines (eg:)
# 1 Mary is in the park.
# 2 Julie travelled to the office.
# 3 Is Julie in the kitchen? 	no	2
# 4 Julie went back to the school.
# 5 Mary went to the office.
# 6 Is Mary in the office? 	yes	5
# 7 Fred is in the cinema.
# 8 Julie is either in the kitchen or the bedroom.
# 9 Is Julie in the bedroom? 	maybe	8
def file_to_examples(file):
    f = open(file, "r")
    lines = f.readlines()
    information = []
    questans = []

    # Want tuples (information, information ..., information, answer)
    for line in lines:
        split = line.strip().split('\t')
        linesplit = split[0].split(' ')
        linenum = int(linesplit[0])
        sentence = " ".join(linesplit[1:]).strip()

        # Signals start of new set
        if linenum == 1:
            information = []

        # For each question, add as the information all of the previous
        # sentences that could have been relevent.
        if sentence[-1] == "?":
            question = sentence
            answer = split[1]
            # hint = int(split[2])
            questans.append(example(sentences=list(information), \
                                    answer=answer, \
                                    question=question, \
                                    hints=""))
        else:
            information.append(sentence)

    return questans

# Returns (train_examples, test_examples)
def get_data(datadir, tasknum):
    if tasknum == 1:
        train_examples = file_to_examples(datadir+"qa1_single-supporting-fact_train.txt")
        test_examples = file_to_examples(datadir+"qa1_single-supporting-fact_test.txt")
    elif tasknum == 5:
        train_examples = file_to_examples(datadir+"qa5_three-arg-relations_train.txt")
        test_examples = file_to_examples(datadir+"qa5_three-arg-relations_test.txt")
    elif tasknum == 7:
        train_examples = file_to_examples(datadir+"qa7_counting_train.txt")
        test_examples = file_to_examples(datadir+"qa7_counting_test.txt")
    elif tasknum == 17:
        train_examples = file_to_examples(datadir+"qa17_positional-reasoning_train.txt")
        test_examples = file_to_examples(datadir+"qa17_positional-reasoning_test.txt")
    elif tasknum == 19:
        train_examples = file_to_examples(datadir+"qa19_path-finding_train.txt")
        test_examples = file_to_examples(datadir+"qa19_path-finding_test.txt")
        # hack to replace directions with their actual words
        fix_directions(train_examples)
        fix_directions(test_examples)
    else:
        raise NotImplementedError("Task %d has not been implemented yet" % tasknum)
    return (train_examples, test_examples)


