import os
import sys
import argparse
import numpy as np
import time

# Some notes to myself: (To help reduce the run time of my program)
# 1) Try to use dictionaries where ever possible because they have constant access time
# 2) Try to use numpy where ever possible because numpy allows for faster calculation, which will save you time 
#    when working with huge probability tables

# My global variables

# Basic initialization
my_sentences = []
my_sentences_tag = []
my_sentences_test_file = []
my_sentences_tag_index_test_file = []

my_tag_index = dict() 
my_words_index = dict()

# I = []     # Converted to a numpy array in initialize_data()
# T = []     # Converted to a numpy array in initialize_data()
# M = []     # Converted to a numpy array in initialize_data()

# These tags have been taken from Piazza post @943

my_tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD'] 

######################################################################################################################################

def read_in_the_files(training_list):
    """Read in the training files.
    Also, generate the desired sentences at this step. By doing so we would not have
    to refer back to the files again and again.
    """
    temp_sentence = []
    temp_sentence_tag = []
    word_index = 0

    for file in training_list:
        
        f = open(file)
        
        lines = f.readlines()
        
        for line in lines:
            
            # Note that each line has the format of -> word : tag
            temp = line.split(":")
            
            # Now adding word to the sentence
            if temp[0].strip() != "":
                if temp[1].strip() != "": # To account for the noise/invalid entries in the training dataset
                
                    if temp[0].strip() not in my_words_index:
                        my_words_index[temp[0].strip()] = word_index
                        word_index = word_index + 1
                    
                    if temp[0].strip() in ['.', '?', '!']:
                        temp_sentence.append(temp[0].strip())
                        temp_sentence_tag.append(temp[1].strip())
                        my_sentences.append(temp_sentence)
                        my_sentences_tag.append(temp_sentence_tag)
                        temp_sentence = []
                        temp_sentence_tag = []
                    else:
                        temp_sentence.append(temp[0].strip())
                        temp_sentence_tag.append(temp[1].strip())
                    
        f.close()
        

def initialize_data():
    """Now that we have read in the file and generated our sentences,
    we need to initialize some stuff like dictionaries which would help us maintain
    indexes to our tags and words.
    Note: We should use dictionaries here which would help reduce the lookup time
    since dictionaries have constant time for most operations
    """
    
    i = 0
    
    for tag in my_tags:
        my_tag_index[tag] = i
        i = i + 1
        
    # Optimized the code and put it in with the read_in_the_files() function
         
    # i = 0
    
    # for sentence in my_sentences:
    #     for word in sentence:
    #         if word not in my_words_index:
    #             my_words_index[word] = i
    #             i = i + 1
            
    
    # I = np.zeros((1, len(my_tags)))
    # T = np.zeros((len(my_tags), len(my_tags)))
    # M = np.zeros((len(my_tags), len(my_words_index)))
            
            
def create_initial_probability_matrix_I():
    """Need to create the matrix for the initial probabilities.
    From the assignment handout we know that :-
    This probability is how likely each tag appears at the beginning of a sentence.
    Note that from Piazza post @956 we have - 
    "The initial probability table is one distribution"
    """
    global I 
    I = np.zeros((1, 91))
    # Now iterating over the sentences
    
    for sentence in my_sentences_tag:
        
        index = my_tag_index[sentence[0]]
        
        I[0][index] = I[0][index] + 1
        
    I[0] = I[0] / len(my_sentences_tag)
    
    
def create_transition_probability_matrix_T():
    """Need to create the transition matrix probabilities.
    Note that from Piazza post @956 we have -
    "For the transition probability table, for each starting tag, there is a probability distribution over the next tag."
    This means that we can have -
    rows - would represent the tags, while the columns - would represent the next tag
    """
    global T 
    T = np.zeros((len(my_tags), len(my_tags)))
    # Now iterating over the sentences
    
    for sentence in my_sentences_tag:
        for i in range(0, len(sentence) - 1):
            row_index = my_tag_index[sentence[i]]
            column_index = my_tag_index[sentence[i + 1]]
            T[row_index][column_index] = T[row_index][column_index] + 1
            
    # Converting to valid probability distribution
    for i in range(0, 91):
        if sum(T[i]) != 0:
            T[i] = T[i] / sum(T[i])




def create_observation_probability_matrix_M():
    """Need to create the observation probability matrix.
    Note that from Piazza post @956 we have -
    "For the observation probability table, for each tag, there is a probability distribution over the observed words."
    This means that we can have -
    rows - would represent the tags, while the columns - would represent the words
    """
    global M
    M = np.zeros((len(my_tags), len(my_words_index)))
    # Now iterating over the sentences
    
    for i in range(0, len(my_sentences)):
        for j in range(0, len(my_sentences[i])):
            row_index = my_tag_index[my_sentences_tag[i][j]]
            column_index = my_words_index[my_sentences[i][j]]
            M[row_index][column_index] = M[row_index][column_index] + 1
            
    # Converting to valid probability distribution
    for i in range(0, 91):
        if sum(M[i]) != 0:
            M[i] = M[i] / sum(M[i])



def viterbi_modified(sentence):
    """ 
    From Piazza post @956 we have -
    "After that, you can perform Viterbi with each sentence in the test file."
    
    Thus, viterbu_modified accepts a sentence as input.
    We can treat this sentence (from the test file) as 'E' in the pseudo-code from lectures.
    We can treat our 'my_tags' as 'S' from the pseudo-code as 'my_tags' essentially,
    represents all the possible hidden states.
    
    The small modification which we make to our pseudo-code from lecture is according to
    Piazza post @1075 -
    If a word in the test file is not present in the training files, then we do not
    multiply the respective observation probability
    """
    
    # Copying the pseudo-code from lecture with some minor changes
    
    prob = np.zeros((len(sentence), len(my_tags)))
    prev = np.zeros((len(sentence), len(my_tags)))
    
    # Determine values for time step 0
    for i in range(0, len(my_tags)):
        # At this point, according to Piazza post @1075
        # we need to see if a word from the test file was present
        # in the training file or not. If not present in the training file,
        # then we do not multiply the respective observation probability
        if sentence[0] not in my_words_index:
            prob[0][i] = I[0][i]
            prev[0][i] = None
        else:
            prob[0][i] = I[0][i] * M[i][my_words_index[sentence[0]]]
            prev[0][i] = None
            
    # For time steps 1 to len(sentence) - 1,
    # find each current state's most likely prior state 'x'
    for t in range(1, len(sentence)):
        for i in range(0, len(my_tags)):
            # At this point, according to Piazza post @1075
            # we need to see if a word from the test file was present
            # in the training file or not. If not present in the training file,
            # then we do not multiply the respective observation probability
            if sentence[t] not in my_words_index:
                
                # The usage of np.argmax() has been copied from Piazza post @1065
                
                x = np.argmax(prob[t - 1, : ] * T[ :, i])
                prob[t][i] = prob[t - 1][x] * T[x][i]
                prev[t][i] = x
            else:
                
                # The usage of np.argmax() has been copied from Piazza post @1065
                
                x = np.argmax(prob[t - 1, : ] * T[ :, i] * M[i, my_words_index[sentence[t]]])
                prob[t][i] = prob[t - 1][x] * T[x][i] * M[i][my_words_index[sentence[t]]]
                prev[t][i] = x
    
    return prob, prev
        


def read_in_the_test_file(filename):
    """Reading in the test file by breaking it up into sentences.
    """
    
    f = open(filename)
    
    lines = f.readlines()
    
    temp = []
    
    for line in lines:
        
        x = line.strip()
        
        if x != "":    # checking for noise from the test file
            if x in ['.', '?', '!']:
                temp.append(x)
                my_sentences_test_file.append(temp)
                temp = []
            else:
                temp.append(x)
            
    f.close()
        
    


def trace_back_to_find_solution():
    """Trace back the viterbi algorithm using 'x' and 'prev' described
    in the pseudo-code from lectures to find the required sequence of tags
    for each sentence in the test file.
    
    To keep in mind according to Piazza post @1065_f1 that the trace back would
    give the tags in reverse order and thus we would need to reverse them, so as 
    to get the correct order.
    "Then when you have all the max probabilities you take the best path by going 
    backwards through your max probabilities and save the tags, which you will need 
    to reverse to match to the words in your sentence.
    Forgetting to reverse, or not choosing the best path will both give you extremely 
    low accuracy." - Piazza post @1065_f1
    """
    
    # Getting each sentence from the test file and using Viterbi on it to get the most
    # likely sequence and then tracing back to get the desired solution
    
    for sentence in my_sentences_test_file:
        
        temp = []
        
        prob, prev = viterbi_modified(sentence)
        
        temp = [int(np.argmax(prob[len(sentence)-1, :]))] + temp
        
        for i in range(len(sentence) - 1, 0, -1): 
            if prev[i][temp[0]] != None:   
                temp = [int(prev[i][temp[0]])] + temp
                
        my_sentences_tag_index_test_file.append(temp)
        




def print_solution(outputfile):
    """Print the solution to the output file contained in
    my_sentences_test_file and my_sentences_tag_index_test_file.
    """
    
    f = open(outputfile, 'w')
    
    for i in range(0, len(my_sentences_test_file)):
        for j in range(0, len(my_sentences_test_file[i])):
            f.write(str(my_sentences_test_file[i][j]) + ' : ' + str(my_tags[my_sentences_tag_index_test_file[i][j]]) + '\n')
            
    f.close()


if __name__ == '__main__':
    
    # start_time = time.time()  # My added code

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    # print("training files are {}".format(training_list))

    # print("test file is {}".format(args.testfile))

    # print("output file is {}".format(args.outputfile))


    # print("Starting the tagging process.")
    
    # Starting my functions from here
    
    read_in_the_files(training_list)
    initialize_data()
    # print("--- %s seconds ---" % (time.time() - start_time))
    create_initial_probability_matrix_I()
    create_transition_probability_matrix_T()
    create_observation_probability_matrix_M()
    # print("--- %s seconds ---" % (time.time() - start_time))
    read_in_the_test_file(args.testfile)
    trace_back_to_find_solution()
    print_solution(args.outputfile)
    
    # print("--- %s seconds ---" % (time.time() - start_time))
                      
    


