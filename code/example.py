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
