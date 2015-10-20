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

def getdata(file):

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

        if linenum == 1:
            information = []

        # Question
        if sentence[-1] == "?":
            question = sentence
            answer = split[1]
            # hint = int(split[2])
            questans.append({"q": question, "a": answer, "s": list(information)})
        else:
            information.append(sentence)
    return questans

