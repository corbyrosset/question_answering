# Questions
# - at what point are we going to ask it the questions

# possible format
# statements
# story_id line_number statement

# questions
# story_id query answer support_line

import pandas as pd
import example
import glob


# Reads in a file expecting a sequence of numbered sentences
# Assumes stories begin with a line numbered '1 '
def find_stories(filename):
    stories = []
    with open(filename, 'r') as f:
        story = [f.readline().rstrip()]      # start with the first sentence
        for line in f.readlines():
            if line[:2] == '1 ':
                stories.append(story)
                story = []
            story.append(line.rstrip())
    return stories


# Return an array of all the statements and all the questions corresponding to a
# given story. The questions use TSV formatting to denote separate parts.
def process_story(story):
    statements = []
    queries = []
    for line in story:
        offset = line.index(' ')
        line_id = line[:offset]
        line_content = line[offset+1:]
        if '?' in line:
            queries.append([line_id] + line_content.split('\t'))
        else:
            statements.append([line_id, line_content])
    return statements, queries


# Given a list of stories, stories_to_pandas aggregates all of the story
# informations into a one large dataframe for the statements and the questions
def stories_to_pandas(stories, process=process_story,
                      scol_names=['line_id', 'statement'],
                      qcol_names=['line_id', 'query', 'answer', 'support_line']):
    statements = []
    queries = []
    for i, story in enumerate(stories):
        s, q = process(story)
        s = pd.DataFrame(s, columns=scol_names)
        q = pd.DataFrame(q, columns=qcol_names)
        s['story_id'] = i
        q['story_id'] = i
        statements.append(s)
        queries.append(q)
    Statements = pd.concat(statements).reset_index(drop=True)
    Queries = pd.concat(queries).reset_index(drop=True)
    return Statements, Queries


def generate_examples(Statements, Queries):
    examples = []
    for story_id in xrange(max(Statements.story_id)):
        ss = Statements[Statements.story_id == story_id]
        qs = Queries[Queries.story_id == story_id]
        for _, q in qs.iterrows():
            statements = ss[ss.line_id < q.line_id].statement.values
            question = q.query
            answer = q.answer
            hints = map(int, q.support_line.split())
            examples.append(example.example(statements, question, answer, hints))
    return examples


# Example usage
#
# statements and queries will be pandas dataframes with columns that correspond
# to the relevant information
# Hardcoded to load the first dataset
def get_task_examples():
    task_examples = []
    for i in xrange(1, 2): # will change the upper one to 21 when the other datafiles are loaded
        filestump = 'qa%d_*train.txt' % i
        stories = find_stories(glob.glob(filestump)[0])
        statements, queries = stories_to_pandas(stories)
        examples = generate_examples(statements, queries)
        task_examples.append(examples)
    return task_examples

