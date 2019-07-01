import random

def build_dataset(sent_nums):
    text = open("data/data.txt", "r").read()
    sent = text.split('\n')
    random.seed(0)
    random.shuffle(sent)
    sent = sent[0:sent_nums]
    
    t = int(sent_nums*0.8)
    train = '\n'.join(sent[0:t])
    test = '\n'.join(sent[t:])
    
    with open("data/train.txt", "w") as file:
        file.write("%s" % train)
    with open("data/test.txt", "w") as file:
        file.write("%s" % test)
        
build_dataset(50000)