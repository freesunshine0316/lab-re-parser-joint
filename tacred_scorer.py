from collections import Counter
import sys
NO_RELATION = 0

def score(key, prediction, handler=sys.stdout, verbose=True):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(str(relation)), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            handler.write(("{:<" + str(longest_relation) + "}").format(relation))
            handler.write("  P: ")
            if prec < 0.1: handler.write(' ')
            if prec < 1.0: handler.write(' ')
            handler.write("{:.2%}".format(prec))
            handler.write("  R: ")
            if recall < 0.1: handler.write(' ')
            if recall < 1.0: handler.write(' ')
            handler.write("{:.2%}".format(recall))
            handler.write("  F1: ")
            if f1 < 0.1: handler.write(' ')
            if f1 < 1.0: handler.write(' ')
            handler.write("{:.2%}".format(f1))
            handler.write("  #: %d" % gold)
            handler.write("\n")
        print("", file=handler, flush=True)

    # Print the aggregate score
    if verbose:
        print("Final Score:", file=handler, flush=True)
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro), file=handler, flush=True)
    print("   Recall (micro): {:.3%}".format(recall_micro), file=handler, flush=True)
    print("       F1 (micro): {:.3%}".format(f1_micro), file=handler, flush=True)
    return prec_micro, recall_micro, f1_micro