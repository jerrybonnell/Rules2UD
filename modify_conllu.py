from typing import OrderedDict
import bs4 as bs
import os
import sys
import glob
import json
import re
import numpy as np
import esupar
from datetime import datetime
from rich.progress import track
from rich.console import Console
from rich.pretty import pprint as rich_pretty
import collections
import itertools
import subprocess
import io


#sys.path.append('../')
from evaluation_script import conll18_ud_eval

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

console = Console()
verbose = False

def pprint(s):
    rich_pretty(s)

def vprint(s):
    if verbose:
        console.print(s)

# for determining a bad or good reading from ginza output
ok_cats = ['KATAKANA', 'SYMBOL', 'SPACE',
           'NUMERIC', 'ALPHA', 'GREEK',
           'CYRILLIC']
bad_cats = ['KANJINUMERIC', 'KANJI']

cat_info_list = []
cat_info_map = {}

rule_sets = []
for rule_json in glob.glob(f"{FILE_DIR}/rules/*.json"):
    with open(rule_json, "r", encoding='utf-8') as f:
        rules = json.load(f)
    rule_sets.append(rules)

all_keys = []
for rule_set in rule_sets:
    all_keys.append(list(rule_set['rules'].keys()))
all_keys = [j for i in all_keys for j in i]
# there is overlap in these three rules (as expected) so remove them from
# check in following line
all_keys = [j for j in all_keys if j not in ['つた', 'つて', 'つと']]
# there should be no duplicates in any of the keys
assert len(all_keys) == len(set(all_keys))

# sort the rules of each dictionary from longer to shorter substitutions
def order_dict(d, by_index=1):
    return collections.OrderedDict(
        sorted(d.items(), key=lambda t: len(t[by_index]), reverse=True))

# subsumption code: delete any rules that have same application as other
# smaller rules that can do more and so are more general/useful
def eliminate_redundancy(rule_group):
    to_delete = []
    for before, after in rule_group.items():
        for i in range(min(len(before), len(after))):
            if before[i] == after[i]:
                before_new_right = before[i+1:]
                after_new_right = after[i+1:]
                before_new_left = before[:i]
                after_new_left = after[:i]
                if before_new_left in rule_group and \
                        rule_group[before_new_left] == after_new_left and \
                        before[i:] == after[i:]:
                    to_delete.append(before)
                elif before_new_right in rule_group and \
                        rule_group[before_new_right] == after_new_right and \
                        before[:i+1] == after[:i+1]:
                    # last condition checks what comes before
                    to_delete.append(before)

    # delete the redundant rules from the full rule set
    if len(to_delete) > 0:
        console.print(f"delete redundant rules : {to_delete}")
    [all_rules.pop(e) for e in to_delete if e in all_rules]

all_rules = {}
for i in range(len(rule_sets)):
    all_rules.update(rule_sets[i]['rules'])
    eliminate_redundancy(rule_sets[i]['rules'])


def load_dictionary():
    with open(f'{FILE_DIR}/char.def', 'r') as f:
        cat_info_list = f.read().split('\n')
    cat_info_list = [l.split(' ') for l in cat_info_list][:-1]
    # convert the unicode representation to just an integer
    for line in cat_info_list:
        line[0] = int(line[0], 16)
        if len(line) > 2:
            line[1] = int(line[1], 16)
        else:
            # things strictly one value are added to the map
            cat_info_map[line[0]] = line[1]
    # delete all sublists that are strictly one value
    cat_info_list = [l for l in cat_info_list if len(l) == 3]
    # sort the list
    cat_info_list.sort(key=lambda x: x[0])
    return cat_info_list, cat_info_map


def get_word_cat(char):
    assert len(char) == 1
    char_value = ord(char)
    # vprint(f"char: {char} value: {char_value}")
    # now we do search in the char.def file to determine which
    # category of unknown this char is from (which corresponds
    # to the cat_info_list and cat_info_map ds's)
    if char_value in cat_info_map:
        return cat_info_map[char_value]
    for entry in cat_info_list:
        if char_value >= entry[0] and char_value <= entry[1]:
            return entry[2]
    return 'DEFAULT'  # default is a mandatory category


cat_info_list, cat_info_map = load_dictionary()
#vprint(get_word_cat("^"))

def from_rule_set(rules):
    for rule_set in rule_sets:
        if rule_set['rules'] == rules:
            return rule_set['name']
    raise ValueError("unknown rule set")


global_array = []
"""
a b c d e f g h i j   original sentence
1 2 3 4 5 6 7 8 9 10
  ***           ****

two rules match on orig sentence: [(2,3), (9,10)]
 > assume each rule makes sequence 1 longer
scenario 1: change (9, 10) first

global array [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

- - - - - - - - + +  +
1 2 3 4 5 6 7 8 9 10 11
  ***           ****

step I.
change the start at index 9
(note: it shouldn't matter if we set arr[9] (start) to 1 or arr[10] (end) to 1)
global array [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

step II.
array sum to start=2 (not inclusive): 0
 > therefore, do not add any diff to start or end

- @ @ @ - - - - -  + +  +
1 2 3 4 5 6 7 8 9 10 11 12
  ***           ****

global array [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]


scenario 2: change (2, 3) first

step I.
global array [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

- + + + - - - - - -  -
1 2 3 4 5 6 7 8 9 10 11
  ***           ****

global array [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

step II.

- + + + - - - - - @  @  @
1 2 3 4 5 6 7 8 9 10 11 12
  ***           ****

array sum to start=9 (not inclusive): 1
  > therefore, diff = 1 which gets added to start and end
  > therefore, new start = 10 and new end = 11

global array [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]

scenario 3: 3 hard replace rules

a b c d e f g h i j   original sentence
1 2 3 4 5 6 7 8 9 10
  ***   *****   ****

three rules match on orig sentence: [(2,3), (5,7), (9,10)]

### process (2,3)

global array [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
> note that the example is one-based; also we add diffs to the start, not end

a + + + d e f g h i  j
1 2 3 4 5 6 7 8 9 10 11
  ***   *****   ****

global array [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

### process (9,10)

a + + + d e f g h @  @  @
1 2 3 4 5 6 7 8 9 10 11 12
  ***   *****   ****

global array [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
array sum to start=9: 1

### process (5, 7)

a + + + % % % % g h  @  @  @
1 2 3 4 5 6 7 8 9 10 11 12 13
  ***   *****   ****

global array [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
array sum to start=5: 1
"""

def handle_replace(sent, new_sent, rules, log, func, model):
    global global_array

    for orig_word in rules:
        # skip all rules that we detected as being redundant
        if orig_word not in all_rules:
            continue
        """
        three notes about w_indices4sent:
        (1) returns a list of indices where the orig_word pattern occurs
        (2) the new sentence is continuously evolving when this method gets called
            by the different rule sets (easy, hard, before_after, etc); so we need
            to keep track of the indices in the current version of the sentence
            because we can't rely on the indices of the original sentence anymore
            because the lengths are changing
        (3) for some sentences, we do not realize all rules that need
            to be applied until we have first applied some basic rules;
            it is only then where we see that there are some hard
            rules that become applicable, so we need to keep track of
            indices that correspond to the new sentence, not original; put
            another way, if we were to try a lookup into the original
            sentence, we would not find it (NOTE doesn't apply right now)
        """
        # we want to keep the indexes corresponding to the original sentence
        # when doing the substitution so we need to keep track of the indices
        # in the sentence before any modification
        w_indices4sent = [(m.start(), m.start() + len(orig_word) - 1)
                         for m in re.finditer(orig_word, sent)]
        w_orig_indices = []  # list of indices that pass the rule
        # this is the difference in the length between the before and after
        # characters in a rule
        diff = len(rules[orig_word]) - len(orig_word)
        # if there are multiple occurrences where a hard rule needs to be
        # applied, then we use counter to keep track of the current number
        # of times we've done a hard rule replacement; recall that each
        # time we do one, it changes the length of the string, hence
        # we multiply counter * diff
        # ex: {'如斯き': [(0, 2), (4, 6)]} note two occurrences of the pattern
        # counter = 0

        # we do replacements with respect to the new sentence
        for w_tuple in w_indices4sent:
            start, end = w_tuple
            orig_new_sent = (new_sent + '.')[:-1]  # deep copy

            if func(sent, start, end):
                # do the sentence replacement
                vprint(f"!!! in the if here with orig_word {orig_word}")
                # handle the log with respect to the original sentence
                vprint(f"w_indices4org_sent: found {w_tuple}")
                w_orig_indices.append((start, end))
                # update start and end if there are multiple occurrences
                # for this one search word

                # handle squiggly logic
                if start - 2 >= 0 and orig_word == "〜〜":
                    # two characters before squiggly
                    replaced_with = sent[start - 2:end - 1]
                elif orig_word == "〜〜":
                    # only one character before the squiggly
                    # repeat that one character twice
                    # NOTE the two squigglies are place with the single
                    # character that comes before it so actually this
                    # reduces the length of the sentence by 1; so we need
                    # to put a -1 to show the length has decreased
                    replaced_with = sent[0]
                    global_array[end] = -1
                else:
                    replaced_with = rules[orig_word]

                # for global array add up all values from 0 to start
                array_sum = 0
                for i in range(start):
                    array_sum += global_array[i]

                # NOTE this was changed from global_array[end]
                global_array[start] = diff
                start = start + array_sum
                end = end + array_sum

                new_sent = new_sent[:start] + \
                    replaced_with + \
                    new_sent[end + 1:]
                vprint(f"new sent is now : {new_sent}")
                # for handling the rule "ふ" -> "う" which the application
                # of this rule may not always be a good idea
                # this is special logic for checking to see if the rule
                # has a good effect on the ginza output. more specifically,
                # we are checking to see if match_word is one of the morphemes
                # returned in the ginza output; if not, we have regret and undo

                # the reason for checking "in" a list is to generalize to other
                # rules that may need this logic
                if orig_word in ['ふ']:
                    # vprint(all_rules[orig_word])
                    if start > 0:
                        match_word = new_sent[start - 1] + all_rules[orig_word]
                        vprint(f"match word : {match_word}")
                    else:
                        raise ValueError("start is at 0")
                    ginza_out = call_model([new_sent], model)[0][1:]
                    # vprint(ginza_out)
                    morphemes = [m[1] for m in ginza_out if type(m) is list]
                    vprint(f"morphemes {morphemes}")
                    if match_word not in morphemes:
                        # undo the work
                        vprint(f"*** undoing {w_tuple}!")
                        w_orig_indices.pop()
                        new_sent = orig_new_sent



        # if w_orig_indices has elements then we had a match so we add
        # information to the log about the match
        if len(w_orig_indices) > 0:
            vprint(f"orig word {orig_word}")
            vprint(f"found in {from_rule_set(rules)} : {orig_word} -> {rules[orig_word]}")
            # if rules[orig_word] in inferred_rules.values():
            #     vprint("*** this rule was inferred!")
            # add (4,6) to the list with (0,2) {'如斯き': [(0, 2)]}
            if orig_word in log:
                log[orig_word].extend(w_orig_indices)
            # elif orig_word in sent or orig_word in inferred_rules:
            elif orig_word in sent:
                vprint("added to log")
                log[orig_word] = w_orig_indices
            vprint("+++")


    return new_sent


def detect_overlap(pos_tuples, pos_counter, old_sent):
    vprint("--- checking overlap ---")
    # look at the entire collection of tuples. if the index of the
    # replaced word are the same, we won't count that as an overlapped
    # case
    # go through character by character
    for i, count in pos_counter.items():
        if count > 1:
            overlap_parts = set()
            for tup in pos_tuples:
                if i >= tup[0] and i <= tup[1]:
                    # NOTE even though the indices may overlap, as given by
                    # the counter, that is not enough to say for sure that
                    # it is indeed an overlap/beach case. to do that, we
                    # have to check for disagreements in the
                    # after string/replacement among the rules; if multiple
                    # rules are trying to replace to the same thing, then
                    # there really is no overlap
                    after_str = all_rules[old_sent[tup[0]:tup[1] + 1]]
                    # diff needed for hard rules; NOTE it can be -1 if the
                    # rule goes shorter
                    diff = len(after_str) - len(old_sent[tup[0]:tup[1] + 1])
                    vprint(f"{i}{tup} after_str={after_str} diff={diff}")
                    # special condition for case when the rule becomes shorter
                    # i - tup[0] check is needed to prevent out of bounds when
                    # we try after_str[after_str_overlap_pos] after
                    if diff < 0 and i - tup[0] < len(after_str):
                        after_str_overlap_pos = i - tup[0]
                    else:
                        after_str_overlap_pos = i - tup[0] + diff
                    # if after_str_overlap_pos < 0:
                    # after_str_overlap_pos = 0
                    # get only the single character that overlaps in after_str
                    overlap_part = after_str[after_str_overlap_pos]
                    vprint(f"{i}{tup} overlap part : {overlap_part}")
                    # the overlap parts from the iterations in this nested for
                    # loop are going to different characters so this is a
                    # true overlap case
                    if overlap_part not in overlap_parts \
                        and len(overlap_parts) > 0:
                        # parts are different, so return true
                        vprint(f"{i}{tup} overlap parts : {overlap_parts}")
                        return True
                    overlap_parts.add(overlap_part)
                    vprint(f"{i}{tup} else overlap parts : {overlap_parts}")

    return False

def handle_elimination(sent, tuples_for_comb, pos_counter):
    ## two tier / elimination logic
    vprint(f"tuples_for_comb {tuples_for_comb}")
    vprint(f"pos counter {pos_counter}")
    # this gives the tuples in tuples_for_comb together with the characters
    # that appear at those indices
    before_parts = [(sent[t[0]:t[1]+1], (t[0], t[1])) for t in tuples_for_comb]
    # create a dictionary that goes from the before part to a list of tuples
    # where it occurs
    for_comb_log = collections.defaultdict(list)
    for before in before_parts:
        for_comb_log[before[0]].extend((before[1],))
    for_comb_log = dict(for_comb_log)

    vprint(f"before_parts {before_parts}")
    vprint(f" for comb log {for_comb_log}")

    before_pairs = list(itertools.combinations(before_parts, 2))
    # filter out identical pairs
    before_pairs = [b for b in before_pairs if b[0][0] != b[1][0]]
    # sort it so the smallest rule (specifically the characters in before part)
    # always comes first
    before_pairs = [
        tuple(sorted(t, key=lambda x: len(x[0]))) for t in before_pairs]
    vprint(before_pairs)

    for pair in before_pairs:
        # if the pair is not overlapping, forget it
        # get the indices that are influenced by each rule in the pair as
        # a list of lists
        occurred_indices = [list(range(t[1][0], t[1][1]+1)) for t in pair]
        # flatten that list and do a counter
        occurred_indices = [v for subl in occurred_indices for v in subl]
        pair_cntr = list(collections.Counter(occurred_indices).values())
        # if the counter has any values that are more than 1 then there
        # is overlap that needs to be handled; otherwise continue
        pair_cntr = [c for c in pair_cntr if c != 1]
        if len(pair_cntr) == 0:
            continue

        if pair[0][0] in pair[1][0]:
            # the smaller rule is fully contained by the larger
            # one; get rid of the smaller rule
            vprint(f"TWO TIER : {pair[0]} IN {pair[1]}")
            tuples_for_comb.discard(pair[0][1])
            vprint(f"ELIMINATE {pair[0]}")


def handle_overlap(sent, pos_tuples, pos_counter, model):
    tuples_for_comb = set() # involved in overlap
    tuples_not_for_comb = set() # not involved in overlap

    # make a data structure tuples_for_comb that gives the tuples that are
    # involved in an overlap
    for index, count in pos_counter.items():
        if count > 1:
            # index has overlap
            tuples_for_comb.update(
                [tup for tup in pos_tuples if index >= tup[0] and index <= tup[1]])

    # vprint(f"tup for comb {tuples_for_comb}")
    tuples_not_for_comb = set(pos_tuples) - tuples_for_comb
    # vprint(f"tup not for comb {tuples_not_for_comb}")
    assert tuples_not_for_comb.union(tuples_for_comb) == set(pos_tuples)

    # do the two-tier / elimination step
    vprint(f"tuples_for_comb before elimation: {tuples_for_comb}")
    handle_elimination(sent, tuples_for_comb, pos_counter)
    vprint(f"tuples_for_comb after elimation: {tuples_for_comb}")

    combs_to_try = []
    # run the detection code again to see what the tuple combination
    # should look like
    detect_overlap_flag = detect_overlap(tuples_for_comb, pos_counter, sent)

    for i in range(2, len(tuples_for_comb)+1):
        possible_combs = list(itertools.combinations(tuples_for_comb, i))
        for tuples in possible_combs:
            occurred_indices = []
            # check if the combination is a valid one and insert to
            # combs_to_try if yes
            for tup in tuples:
                occurred_indices += list(range(tup[0], tup[1] + 1))
            c = list(collections.Counter(occurred_indices).values())
            c = [l for l in c if l != 1]
            # if no conflicts add it in
            if len(c) == 0:
                combs_to_try.append(tuples)

    # if there aren't any possible combinations then the best we can do
    # now is just try each tuple one at a time
    if len(combs_to_try) == 0:
        combs_to_try.extend(tuples_for_comb)
    else:
        # if a single-tuple (one rule) combination is not part of a two
        # tuple (two rule) or more combination, then we should still
        # keep that one-tuple combination
        combs_temp = {ele for tup in combs_to_try for ele in tup}
        vprint(f"combs_temp {combs_temp}")
        for item in tuples_for_comb:
            vprint(item)
            vprint(combs_temp)
            if item not in combs_temp:
                combs_to_try.extend((item,))

    if tuple(tuples_for_comb) in combs_to_try or not detect_overlap_flag:
        # if it is possible to use all rules at once then there is no
        # reason to look at all possible combinations
        vprint("overlap: it is possible to use everything")
        combs_to_try = [tuple(tuples_for_comb)]

    new_sent = None
    lowest_score = np.inf
    overlap_flag = False
    final_replace_log = {}

    vprint(f"combs to try: {combs_to_try}")
    for comb in combs_to_try:
        vprint(f"attempt: {comb}")
        # bring in the tuples not involved in the combination
        if type(comb[0]) is tuple:
            tuples_using = tuples_not_for_comb.union(comb)
        else:
            tuples_using = tuples_not_for_comb.union((comb, ))
        tuples_using = list(tuples_using)
        # sort the list if we need to handle hard rules
        tuples_using.sort(key=lambda t: t[1], reverse=True)

        sent_applied = (sent + '.')[:-1]  # deep copy
        replace_log = collections.defaultdict(list)
        for tup in tuples_using:
            sent_slice = sent[tup[0]:tup[1]+1]

            # look up each tuple in all_rules
            # do the replacement so that we have the new sentence
            # with no conflicts
            vprint(f"slice {sent_slice}")
            # if "〜" in sent_slice or "ゝ" in sent_slice:
            if "〜" not in sent_slice:
                replaced_word = all_rules[sent_slice]
            else:
                if tup[0] - 2 >= 0:
                    # if there is more than one character before the squiggly
                    replaced_word = sent[tup[0] - 2:tup[1]-1]
                else:
                    # there is only one character that appears before squiggly
                    replaced_word = sent[tup[0] - 1]

            replace_log[sent_slice].extend((tup, ))

            sent_applied = sent_applied[0:tup[0]] + \
                replaced_word + sent_applied[tup[1]+1:]

        # use these two things to run ginza
        sent_applied_out = call_model([sent_applied], model)[0]
        # calculate score
        num_rows = len(sent_applied_out[1:-2])
        num_bad_readings = 0
        for row in sent_applied_out[1:-2]:
            reading = row[-1].split("=")[-1]
            # go through every individual character in the reading
            # as get_word_cat() only accepts a character as input
            for char in reading:
                if get_word_cat(char) not in ok_cats:
                    # must be one of kanji or kanjinumeric
                    assert get_word_cat(char) in bad_cats
                    num_bad_readings += 1
                    break

        reverse_subst_step(sent_applied_out, sent_applied,
                        sent, replace_log, model)

        for i in range(len(sent_applied_out)):
            if type(sent_applied_out[i]) == list:
                sent_applied_out[i] = "\t".join(sent_applied_out[i])
        sent_applied_out = "\n".join(sent_applied_out)


        out_orig = call_model([sent], model)[0]
        for i in range(len(out_orig)):
            if type(out_orig[i]) == list:
                out_orig[i] = "\t".join(out_orig[i])
        out_orig = "\n".join(out_orig)
        sent_applied_out_f = io.StringIO(sent_applied_out)
        out_orig_f = io.StringIO(out_orig)

        # compare the parsing from new output with parsing from original output
        # if there is no difference between the parsings score will be closer
        # to 100% and, otherwise, closer to 0%
        eval_out = conll18_ud_eval.my_evaluate_wrapper(sent_applied_out_f, out_orig_f)
        eval_score = sum(
            [float(l.split("|")[-1].strip()) for l in eval_out[-3:]])

        score = num_rows + num_bad_readings + eval_score/300
        vprint(f"    score {score}")
        vprint(f"    bad read {num_bad_readings} + rows {num_rows}")
        vprint(f"    eval score {eval_score/300}")
        vprint(f"    replace log {dict(replace_log)}")
        if score < lowest_score:
            vprint(f"    best so far")
            lowest_score = score
            new_sent = sent_applied
            final_replace_log = replace_log
            overlap_flag = False
        elif score == lowest_score:
            vprint(f"    ties with best")
            log_len = len(replace_log)
            final_log_len = len(final_replace_log)
            if log_len < final_log_len:
                new_sent = sent_applied
                final_replace_log = replace_log
                overlap_flag = False
                vprint(f"    log is smaller, so is now the new best")
            elif log_len == final_log_len:
                vprint(f"    log length is tied")
                overlap_flag = True


    # we only do substitution step for the winner
    final_replace_log = dict(final_replace_log)

    vprint(f"log {final_replace_log}")
    vprint(f"overlap flag {overlap_flag}")

    return new_sent, overlap_flag, final_replace_log


def ginza_replace(sent, model):
    """
    returns
        (1) the new sentence
        (2) original sentence
        (3) dict that goes from orig_word
            to tuple indices of start, end where it occurs
    """
    replace_log = {}
    new_sent = f"{sent}" # the sentence that will be returned after replacement

    for rule_set in rule_sets:
        new_sent = handle_replace(sent, new_sent, rule_set['rules'],
                                  replace_log,
                                  eval(f"lambda x, y, z: {rule_set['condition']}"),
                                  model)

    # we need to sort the rules by its length b/c we need to have
    # the longest rule checked first in the overlap detection code
    # following; hence, we need an ordered dict
    ordered_replace_log = order_dict(replace_log, by_index=0)
    # code to detect when a "blocking" situation occurs, so it
    # doesn't get flagged as a case where overlapping rules occur
    # the goal here is to eliminate the blocked rule from the log
    vprint(f"replace log ordered : {ordered_replace_log}")

    check_overlap_log = {**ordered_replace_log}

    pos_tuples = [p for _, pos in check_overlap_log.items() for p in pos]
    vprint(pos_tuples)
    occurred_indices = []
    for tup in pos_tuples:
        occurred_indices += list(range(tup[0], tup[1] + 1))
    pos_counter = collections.Counter(occurred_indices)
    vprint(pos_counter)

    # this function is a gatekeeper to prevent all examples which
    # don't have overlap from going into the handler code which
    # can be very slow
    has_overlap = detect_overlap(pos_tuples, pos_counter, sent)

    if has_overlap:
        res = handle_overlap(sent, pos_tuples, pos_counter, model)
        new_sent, overlap_flag, \
            replace_log = res
        if overlap_flag:
            console.print("has overlap 🏖 ", style="bold red")
            console.print(f"sent is {sent}", style="bold red")
        else:
            console.print("overlap resolved ⛱", style="bold green")
    else:
        vprint("no overlap detected 🏝")


    return new_sent, sent, replace_log


def replace_text_details(out, new_sent, old_sent):
    out[0] = out[0].replace(new_sent, old_sent)


def sub_ginza_out(out, old_s, log, model):
    # get an array of tuples that have form (word, (start, end))
    w_pos_tuples = [(orig_word, p) for orig_word, pos in log.items() for p in pos]
    # sort by the end position of the character because of the
    # hard rules that have changed the string length by this point;
    # by applying in sorted order the indices will be corrected as
    # we go along
    w_pos_tuples = sorted(w_pos_tuples, key = lambda x: x[1][1])
    vprint(f"w_pos_tuples {w_pos_tuples}")
    # two rules need to do processing over the same block so we keep a record
    for orig_word, pos in w_pos_tuples:
        start, end = pos
        vprint("***********")
        vprint(f"orig word {orig_word}")
        vprint(f"start {start} end {end}")
        # length difference of the hard replace rule and the word that is
        # being replaced after we replace the squiggly, it could change
        # to a hard replace rule; hence we need the diff
        if orig_word == "〜〜":
            vprint(f"squiggly start {start} end {end}")
            # we need to replace the squiggly with two characters before it
            replaced_with = old_s[start - 2:end - 1]
            vprint(f"replaced with {replaced_with}")
            if len(replaced_with) == 0:
                # means there is only one character before it so the diff
                # will be shorter; example: へ〜〜る
                diff = -1
            else:
                diff = 0
        else:
            # general situation
            diff = len(all_rules[orig_word]) - len(orig_word)
        vprint(f"diff {diff}")
        counter = 0 # says where we are in the string
        # go through the ginza output list (-1 b/c the last 1 are empty list)
        for i in range(1, len(out) - 1):
            counter += len(out[i][1]) # the counter steps by blocks
            vprint(f"counter {counter}")
            # this condition tells us if there is a substitution that needs to be
            # made; as soon as counter > start, we need to do a substitution;
            # otherwise no substitution needed
            if counter > start:
                # ex: 抑, に, 於い, て (from the new sentence)
                # counter: 1, 2, 4, 5
                # when we reach the 3rd unit, we have:
                #   bui = 2 and eui = 3
                # rule: 於て > 於いて (rule applies to two different blocks)
                end_unit_index = counter - 1
                begin_unit_index = counter - len(out[i][1])
                vprint(f"bui {begin_unit_index} eui {end_unit_index}")
                # recreate the piece with the old character(s) attached
                # the format:
                # get the left part, replace with part, the remaining part
                # this can be used to replace one character, two chars,
                # or more than 2 chars

                # all the substitutions happen together in one block
                # the diff is needed for the hard replace rules
                if end_unit_index >= (end + diff):
                    vprint("in the if statement")
                    vprint(f"start : {start} begin_unit_index : {begin_unit_index}")
                    vprint(f"* {out[i][1][0:start - begin_unit_index]}")

                    # if this is true, it means we came out of the larger
                    # else statement from the previous iteration; this means
                    # the bui is "ahead" of start, so we've progressed onto
                    # a new block
                    if start - begin_unit_index < 0:
                        vprint("in if")
                        # add remaining of orig word because it is a
                        # continuation from the last block; compare with the
                        # following else where
                        # we need to add out before orig_word (out + orig_word ...);
                        # then we just add on whatever comes after it from out
                        # for more notes, see the following else
                        new_str = orig_word + \
                            out[i][1][diff + start -
                                    begin_unit_index + len(orig_word):]
                    else:
                        vprint(f"diff is : {diff}")
                        vprint(
                            f"1st part : {out[i][1][0:start - begin_unit_index]}")
                        vprint(f"2nd : {orig_word}")
                        # need to account for the diff again because the length may
                        # have changed during the replacement
                        vprint(f"3rd : {out[i][1][diff + start - begin_unit_index + len(orig_word):]}")

                        new_str = out[i][1][0:start - begin_unit_index] + \
                            orig_word + \
                            out[i][1][diff + start
                                - begin_unit_index + len(orig_word):]
                    vprint(f"new_str {new_str}")
                    out[i][1] = new_str # do the substitution
                    # for esupar, this would never trigger anyway
                    if out[i][2] == "表現":
                        out[i][2] = "表顕"
                    if out[i][1] == "如斯き" and out[i][2] == "こんな":
                        out[i][2] = "かくごとき"
                        old = out[i][-1].split("=")[-1]
                        out[i][-1] = out[i][-1].replace(old, "カクゴトキ")
                    # we've completed the substitution so go to next
                    # item in the w_pos_tuples
                    break
                # the substitution is split across multiple blocks
                else:
                    vprint("sub is split across multiple blocks")

                    vprint(f"start - bui : {start - begin_unit_index}")
                    # A: the piece returned by ginza is kept if it is not affected
                    # by a rule ("abc" part)
                    piece_a = out[i][1][0:start - begin_unit_index]
                    vprint(f"out[i][1] : {out[i][1]}")
                    vprint(f"piece_a : {piece_a}")
                    # B: determine how many characters from original word contribute
                    # to the current block; note how this also has start - bui
                    # like in piece_a
                    #
                    # out[i][1]    [ piece_a  | piece_b  ]
                    #                           ^^^^^^^^
                    #                           len(out[i][1]) - (start - bui)
                    #
                    # NOTE we make an assumption that the difference in the
                    # rule length, as when we have hard rules, is the number
                    # of characters that will appear in the next block. so
                    # if diff is 1, then orig_word will be split where 1 of
                    # its characters will go to the next block; so if we have
                    # a rule ab > fgh; then "h" will appear in the next block

                    # if diff is -1, we dont want to use any chars from
                    # the next block in the current block; so temporarily
                    # set diff to 0 to avoid errors
                    diff_holder = 0
                    if diff < 0:
                        # see case 88 in the tests
                        diff_holder = diff
                        diff = 0

                    # if the word we are going to substitute is identical to the
                    # lemma for that row, then we use the lemma-word
                    # note the "の" when parsed by ginza
                    piece_b = orig_word[0:len(out[i][1]) -
                                        (start - begin_unit_index) - diff]
                    new_str = piece_a + piece_b
                    vprint(f"a : {piece_a} b: {piece_b}")
                    vprint(f"orig word before update: {orig_word}")
                    vprint(f"new str : {new_str}")
                    # need to update original word for the next loop since we used
                    # parts of it to form new_str and need to cut that piece off
                    orig_word = orig_word[
                        len(out[i][1]) - (start - begin_unit_index) - diff:]

                    # now note that the original word has changed because
                    # in the above piece_b only took a part of orig_word;
                    # so also need to update start
                    vprint(f"(out[i][1]) : {out[i][1]}")
                    vprint(f"len(out[i][1]) : {len(out[i][1])}")
                    start += len(piece_b)

                    # restore diff if diff_holder was used
                    if diff_holder != 0:
                        diff = diff_holder

                    ## corner case to handle when a node becomes blank
                    # if new string comes out empty we don't want to leave that block
                    # empty because otherwise lemma doesn't correspond to anything
                    # so we bet that the first character of orig_word should go there,
                    # but it's only a bet
                    if len(new_str) == 0 and orig_word == "〜〜":
                        # for squiggly just replace and be done; there are no
                        # leftovers to worry about
                        new_str = orig_word
                    elif len(new_str) == 0:
                        # this condition is why we have a blank node
                        global fname
                        assert len(out[i][1]) - \
                            (start - begin_unit_index) - diff == 0, f"{fname}"
                        lemma_len = len(out[i][2])
                        assert lemma_len == 1, f"{fname}"
                        vprint(f"taking {lemma_len} chars from org w {orig_word}")
                        new_str = orig_word[0:lemma_len]
                        out[i][1] = new_str  # do the substitution
                        # start isn't being updated to anything but we just did
                        # an update using lemma len; so update start using that
                        start += lemma_len
                        orig_word = orig_word[lemma_len:]

                    vprint(f"orig word after update : {orig_word}")
                    vprint(f"new start is {start}")

                    out[i][1] = new_str  # do the substitution

def call_ginza(sent_list):
    """
    returns None if sentence does not conform to udpipe annotation standards
    """
    pos_to_sent = {k:v for k,v in zip(range(len(sent_list)),sent_list)}
    sent_s = "\n".join([s for s in sent_list if s is not None])
    outs_s = os.popen(f'echo "{sent_s}" | ginza -d').read()
    vprint(outs_s)

    outs = []
    for out in outs_s.split("\n\n"):
        out = out.split("\n")
        for i in range(len(out)):
            if "\t" in out[i]:
                out[i] = out[i].split("\t")
        # first some preprocessing: reject any sentences that have lemmas or
        # tokens that are too long; also replace "." with "_"
        def utf8len(s):
            return len(s.encode('utf-8'))

        none_flag = False
        for i in range(1, len(out)):
            if len(out[i]) == 0:
                continue
            if out[i][1] == ".":
                out[i][1] == "_"
            if utf8len(out[i][1]) > 200 or utf8len(out[i][2]) > 200:
                outs.append(None)
                none_flag = True
                break
        if none_flag:
            continue
        out = out + ['\n'] * 1
        outs.append(out)

    outs_aligned = [] # need to align with possible none's
    for i in range(len(pos_to_sent)):
        if pos_to_sent[i] is not None:
            # treat outs as a stack and pop from the front
            outs_aligned.append(outs.pop(0))
        elif pos_to_sent[i] is None:
            outs_aligned.append(None)
        else:
            raise ValueError("should not be reached")

    assert len(outs) == 1
    assert len(outs_aligned) == len(sent_list)
    # one remaining newline char at end
    outs_aligned.append(outs.pop(0))

    return outs_aligned

def call_esupar(sent_list):

    outs = []
    for sent in sent_list:
        if sent is None:
            outs.append(None)
            continue
        out = str(esupar_model(sent))
        vprint(out)
        out = out.split("\n")
        out.append("")
        for i in range(len(out)):
            if "\t" in out[i]:
                out[i] = out[i].split("\t")
        out.insert(0, f"# text = {sent}")
        if out[-1] == '':
            del out[-1]
        # first some preprocessing: reject any sentences that have lemmas or
        # tokens that are too long; also replace "." with "_"
        def utf8len(s):
            return len(s.encode('utf-8'))

        none_flag = False
        for i in range(1, len(out)):
            if len(out[i]) == 0:
                continue
            if out[i][1] == ".":
                out[i][1] == "_"
            if utf8len(out[i][1]) > 200 or utf8len(out[i][2]) > 200:
                outs.append(None)
                none_flag = True
                break
        if none_flag:
            continue
        # out = out + ['\n'] * 1
        out[-1] = '\n'
        outs.append(out)
    return outs

def call_model(sent_list, model):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    if len(sent_list) < 1000:
        return call_ginza(sent_list) if model == "ginza" \
            else call_esupar(sent_list)
    else:
        out = []
        for block in list(chunks(sent_list, 1000)):
            out.extend(call_ginza(block) if model == "ginza" \
                else call_esupar(block))
        out = list(filter(lambda x: x != ['', '\n'], out))
        out.append(['', '\n'])
        return out



def reverse_subst_step(out, new_s, old_s, log, model):
    # replace the first element
    replace_text_details(out, new_s, old_s)
    # now replace the rest
    console.print(f"\nlog : {log}")
    sub_ginza_out(out, old_s, log, model)


def modify_conllu(sent_list, apply_rules=True, return_log=False,
                  constrain_rules=None,normalize_only=False, model="ginza"):
    vprint(f"rule application ? {apply_rules} module ? {model}")

    new_s_list = []
    old_s_list = []
    log_list = []
    for sent in track(sent_list):
        if len(sent) == 0 or sent[0] == '◎':
            # ignore sentences with title symbol
            new_s_list.append(None)
            old_s_list.append(sent)
            log_list.append(None)
            continue

        if apply_rules:
            global global_array
            global_array = [0] * len(sent)

            if model == "esupar":
                # these rules aren't necessary for esupar
                for r in ["於て", "働か", "况や",
                        "定て", "焉ぞ", "况んや", "焉"]:
                    if r in all_rules:
                        del all_rules[r]

            if constrain_rules is not None:
                # delete from all_rules
                to_delete = []
                for rule in all_rules:
                    if rule not in constrain_rules:
                        to_delete.append(rule)

                for d in to_delete:
                    del all_rules[d]
            res = ginza_replace(sent, model)
            if normalize_only:
                # only the normalization pass requested by user
                return res
            new_s, old_s, log = res
            vprint(f"old s {old_s}")
            vprint(f"new s {new_s}")
            new_s_list.append(new_s)
            old_s_list.append(old_s)
            log_list.append(log)
        else:
            new_s = sent
            old_s = log = None
            new_s_list.append(new_s)
            old_s_list.append(old_s)
            log_list.append(log)

    # second step: do the reverse substitution to replace the new characters
    # with the old characters; for this we need to know positions of chars
    outs = call_model(new_s_list, model)
    #assert len(outs) - 1 == len(new_s_list) == len(old_s_list) == len(log_list) \
    #    == len(sent_list)
    substituted_outs = []
    for out, new_s, old_s, log in track(
        zip(outs,new_s_list,old_s_list,log_list),total=len(outs)):
        if out is None:
            substituted_outs.append(None)
            continue

        if apply_rules:
            reverse_subst_step(out, new_s, old_s, log, model)

        if not return_log:
            substituted_outs.append(out)
        else:
            rules_view = [{rule : all_rules[rule]} for rule in log]
            substituted_outs.append((out, log, rules_view))
    return substituted_outs

def print_usage():
    u = "python modify_conllu.py "
    u += "[INFILE] [APPLY_RULES] [MODEL] [OUT_DIR] [VERBOSE]"
    print(u)



if __name__ == "__main__":
    import conllu_check
    # take care of loading info
    fname = "sents.txt"
    app_rules = True
    model = "ginza"
    dir_name = 'tmp'
    print(sys.argv)

    # get filename from CLI if there exists one
    if sys.stdin.isatty() and len(sys.argv) > 1:
        fname = sys.argv[1]
    elif not sys.stdin.isatty() and len(sys.argv) > 1:
        fname = sys.stdin.readlines()
        # if that is still empty then an actual filename
        # is trying to be piped in
        if len(fname) == 0:
            fname = sys.argv[1]
            assert os.path.exists(fname)
        # user has piped in a list of sentences
        else:
            sys.argv.insert(1, fname)
    else:
        print_usage()
        exit()

    if len(sys.argv) >= 3:
        app_rules = False if sys.argv[2] == "False" else True
    if len(sys.argv) >= 4:
        model = sys.argv[3]
        assert model == "ginza" or model == "esupar"
    if len(sys.argv) >= 5:
        dir_name = sys.argv[4]
    if len(sys.argv) == 6:
        verbose = eval(sys.argv[5])
    if type(fname) is not list:
        ftype = fname.split(".")[-1]
    else:
        ftype = "list"
    # strange behavior when loading esupar model
    if type(sys.argv[1]) is list:
        del sys.argv[1]

    if model == "esupar":
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        esupar_model = esupar.load("ja")

    console.print((fname, ftype, app_rules, model, dir_name, verbose))
    if ftype == "txt":
        with open(fname, 'r') as f:
            body_lis = f.readlines()
    elif ftype == "xml":
        with open(fname, 'rb') as f:
            soup = bs.BeautifulSoup(f, 'xml')
        body_lis = []
        for link in soup.find_all('s'):
            body_lis.append(link.get_text().strip())
    elif ftype == "list":
        body_lis = fname
    else:
        raise ValueError("unknown file type. accepts xml, txt, or stdin")

    # some preprocessing
    body_lis = [''.join(b.split()) for b in body_lis]
    body_lis = [b for b in body_lis if len(b) > 0]
    assert len(body_lis) > 0

    outs = []
    for sent, sent_out in zip(body_lis,
        modify_conllu(body_lis, apply_rules=app_rules, model=model)):
        #out = modify_conllu(sent, apply_rules=app_rules, model=model)
        if sent_out is None:
            console.print(
                f"\n{sent}-----invalid sentence :stop_sign:", style="bold red")
            continue
        if conllu_check.test_sent_not_corrupted(sent, sent_out):
            console.print(f"\n{sent}-----no corruptions found in FORM :white_heavy_check_mark: ",
                        style="bold green")
        else:
            console.print(f"\n{sent}-----corruptions found in FORM :x: ",
                        style="bold red")

        # time to go back
        for i in range(len(sent_out)):
            if type(sent_out[i]) == list:
                sent_out[i] = "\t".join(sent_out[i])
        sent_out = "\n".join(sent_out)
        console.print(sent_out)
        outs.append(sent_out)

    # write to file
    if type(fname) is not list:
        out_fname = fname.split("/")[-1].split(".")[0] + ".conllu"
    else:
        out_fname = "out.conllu"
    now = datetime.now()
    os.makedirs(dir_name, exist_ok=True)

    with open(f"{dir_name}/{out_fname}", "w") as f:
        f.write("".join(outs))
    console.print(f"wrote output conllu to {dir_name}/{out_fname}",
                  style="bold green")
