import random
def strgenerator(guess,answer):
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    res = ""
    for c in range(len(guess)):
        if guess[c] != answer[c]:
            res = res + alphabet[random.randrange(27)]
        else:
            res = res + guess[c]
    return res

def scoring(str1,answer):
    correct = 0
    for c, letter in enumerate(answer):
        if letter == str1[c]:
            correct+=1
    return (correct/len(answer)*100)

def runner():
    answer = "methinks it is like a weasel"
    guess=""
    for i in range(28):
        guess = guess + "abcdefghijklmnopqrstuvwxyz "[random.randrange(27)]
    score = scoring(guess, answer)
    best = (guess, score)
    counter = 1
    while score < 100:
        counter+=1
        guess = strgenerator(guess,answer)
        score = scoring(guess, answer)
        if score > best[1]:
            best = (guess, score)
        if counter%100 == 0:
            print(best,counter)
runner()