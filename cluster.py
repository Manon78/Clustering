import numpy
#import pandas
import nltk
import re
import itertools


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, average, weighted, dendrogram
import matplotlib.pyplot as plt
import matplotlib as mpl
#from sklearn.cluster import AgglomerativeClustering


################################ PART I: From file input create email objects & list thereof #####################

#definition of Email class:
class Email:
    'Common base class for all emails'
    emCount = 0 # number of emails

    def __init__(self, number, content, recipient, date):
        self.number = number # is an integer
        self.content = content # is a string
        self.recipient = recipient # is a list of 1 or more strings (where string is recipient email address)
        self.date = date # is a string
        Email.emCount += 1

    def displayCount(self):
        print("Total Emails: %d" % Email.emCount)

    def details(self):
        print("Email : ", self.number, ", Recipient: ", self.recipient, ", Date: ", self.date)

# From a list of strings, create a list of email objects with relevant attributes (details).
# Takes list of strings & list of objects. Returns updated output list of objects
def getDetails(input_list, output_list):

    code = 0 # assign code numbers to emails, iteratively increasing from 0

    for item in input_list:

        found_date = False
        recipient = ["Unknown"] # reference variables before finding permanent values
        date = "Unknown"

        # find recipient
        for line in item.split("\n"):
            if "To: " in line:
                recipient = re.findall(r'[\w.-]+@[\w.-]+', line) #attention! How do I end Reg Ex at the end of the line?!
                break # avoids full iteration, since "To: " is 2nd line in text (unless recipient not found)

        # find date
        for line in item.split("\n"):
            if "Date: " in line:
                date_list = re.findall(r'.*', line)
                date = date_list[0][6:] # capture only the first element returned by RegEx and slice out "Date: "
                found_date = True # informs code that data is available for this email (& must be removed from contents)
                break # avoids full iteration, since "Date: " is 5th line in text (unless date not found)

        # find content. content is everything contained between the date (if provided) and the end of the string.
        if found_date:
            content = item.split("\n", 5)[5] # remove first 5 lines of email details if Date is found, ie if email data exists
        else:
            content = item

        email_obj = Email(code, content, recipient, date) # instantiate email object with permanent attributes
        output_list.append(email_obj) # append email object to a list thereof, called output_list

        #update email object code. Note: different coding system needed if emails not contained in single string.
        code += 1

    return output_list # return (updated) list of email objects.


# Debug 1

################################# PART II: Define cluster class  #######################

class Cluster():
    'Common base class for all clusters'
    cluCount = 0 #current number of clusters

    def __init__(self, branches, content, names, sim, number = cluCount):
        self.branches = branches # last two branches of the cluster
        self.content = content # list of strings
        self.names = names # flat list of the email objects contained in the cluster
        self.sim = sim # cosine similarity between the last two branches
        self.number = number
        Cluster.cluCount += 1

    def displayCount(self):
        print("Total Clusters: %d" % Cluster.cluCount)

    def details(self):
        print("Cluster: ", self.number, ", Branches: ", self.branches)


########################### PART III: define do_cluster which clusters two items per run ######################

# from text file, obtain list of email objects. //Note: text file should contain emails separated by END_OF_EMAIL.
strings_list = open('semifull.txt').read().split('END_OF_EMAIL\n')
current_list = []
getDetails(strings_list, current_list) # from a list of strings, append initial email objects to current_list

# 2.) use find_most_similar() to determine which objects to cluster
# 3.) cluster the two objects & return new objects list

# take list of objects as input, return new object list after clustering the two most similar items (tf-idf cosine)
def do_clustering(current_list):

    # sub_f1
    # takes ordered list of strings. Returns: index of maximum similarity (tuple) & corresponding similarity value.
    def find_most_similar(list_of_strings, threshold=None):
        thresh_met = False

        # Define tokenize function, used for TF-IDF vectorization and (optionally) for lexicon creation.
        def tokenize(text):
            # following @brandonrose's advice: tokenize by sentence and by word, to ensure that punctuation is caught
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            # filter out any tokens not containing letters (numeric tokens, raw punctuation)
            accepted_tokens = []
            for token in tokens:
                if re.search('[a-zA-Z]', token):
                    accepted_tokens.append(token)
            return accepted_tokens  # returns a list of tokens for the given text

        vect = TfidfVectorizer(  # define TfidfVectorizer parameters. TfidfVectorizer imported from sklearn
            max_df=0.8, max_features=200000,
            min_df=0.06, stop_words=None,
            use_idf=True, tokenizer=tokenize, ngram_range=(1, 3))

        # create and adjust similarity matrix:
        term_mat = vect.fit_transform(list_of_strings)  # create term-document matrix (numpy array)
        similarity_matrix = cosine_similarity(term_mat)  # given term-doc, get document cos-sim matrix (numpy array)
        numpy.fill_diagonal(similarity_matrix, 0)  # replace all diagonal values (similarity with self) from 1 to 0.

        # takes document similarity matrix, threshold limit, matrix dimension indicators as input.
        # Returns indices of most similar documents  &  their similarity value.
        def find_max_similarity(matrix, thresh, dim_indicator):

            # Using numpy method, obtain flat index of the highest cosine similarity in similarity matrix.
            flat_index = numpy.argmax(matrix)  # Note: flat index: index counted from left to right and row by row

            # Given dimensions of the matrix, convert flat index to (x,y) tuple form using numpy unravel method
            dim = (len(dim_indicator), len(dim_indicator))
            index = numpy.unravel_index(flat_index, dim)

            # Given the index and matrix, obtain highest similarity value of current similarity matrix.
            val = (similarity_matrix[index[0]][index[1]])

            # Allow existence of a minimum similarity threshold below which objects are not clustered.
            if thresh: # threshold optional
                if val >= thresh:
                    return index, val
                else:
                    global thresh_met
                    thresh_met = True
                    print("Similarity below threshold.")
                    return thresh_met
            else:  # if no threshold is set, return index and similarity value always.
                return index, val

        # run function and store output items in temporary variable while checking for threshold
        results = find_max_similarity(similarity_matrix, threshold, list_of_strings)
        if thresh_met:
            end = "end"
            return end
        else:
            index, value = results[0], results[1]
            return index, value, similarity_matrix

            # debug 2
            # debug 3

    # sub_f2
    # takes list objects (cl* or em*). Returns a list of corresponding content strings. {Only content, no email details}
    def to_text(current_list):
        text_list = []
        for obj in current_list:
            if isinstance(obj, Email):
                text_list.append(obj.content)
            elif isinstance(obj, Cluster):
                if isinstance(obj.content, list):
                    text_list.append("\n".join(obj.content))  # text of a cluster is the concatenation of its constituents
        #print(text_list)
        if len(text_list) == len(current_list):
            return text_list
        else:
            print("error in creating text list")
            return []

    # sub_f3
    # takes two objects and their similarity value. Clusters them. Returns one cluster object with updated contents list
    def make_cluster(first, second, similarity):
        branches = [first, second]  # list of email object(s) and/or cluster object(s).
        text = [] # list of strings contained in new cluster (attribute content)
        objects = [] # list of email objects contained in new cluster (attribute names)
        sim = similarity

        # create list of text string, which constitutes the content of this cluster
        for item in branches:

            if isinstance(item, Email):
                text.extend(item.content)
                objects.append(item) # append current email object to cluster's list of email objects

            elif isinstance(item, Cluster):
                for x in item.names: # for every object in the previous cluster
                    objects.append(x) # flat list of all email objects contained in the current cluster
                    text.extend(x.content)  # flat list of texts (strings) contained in the current cluster

            # debug 4?

        clus = Cluster(branches, text, objects, sim, Cluster.cluCount)
        return clus

    # sub_f4
    # takes list of objects & the tuple with indices of objects to remove & cluster object to append. Returns new updated list
    def update(lst, must_remove, must_append):

        # a.) remove unwanted items (those which are being clustered into one object)
        # if the first item is removed to the left of the second, the index of second item will be off-put by 1
        if must_remove[0] < must_remove[1] and must_remove[1] != 0:
            del lst[must_remove[0]] # debug 5 before this line
            del lst[must_remove[1] - 1] # debug 6 before this line
        else:  # else both indices remain the same, since the list is shortening behind the second item
            del lst[must_remove[0]]
            del lst[must_remove[1]]

        # b.) append the new cluster object
        lst.append(must_append) # debug 7 after this line

        return lst


    ########## The four inner functions defined above are used below ##############################################

    # call the function which determines a.) whether to end algorithm and b.) which two objects to cluster next
    # run function and store output items in temporary variable while checking for threshold
    output = find_most_similar(to_text(current_list))
    if output == "end":
        global end
        end = True  # inform level+1 of need to end algorithm
    else:
        must_cluster, similarity_value, similarity_matrix = output # otherwise store output into variables.
        # Output: indices of the two objects to cluster (in a tuple), similarity value, similarity_matrix of current objects

    new_cluster = make_cluster(current_list[must_cluster[0]], current_list[must_cluster[1]], similarity_value) # cluster the relevant objects contained in cluster_list. creates a new cluster object.
    new_obj_list = update(current_list, must_cluster, new_cluster)
    return new_obj_list

#################################### PART IV: call do_cluster iteratively until end #######################################

#set k to the final number of clusters desired, else iterate until len(current_list) == 2
k = 4

end = False # end occurs if either k is reached or the smallest similarity threshold between object contents is reached
s = 60 # s is the number of iterations acceptable for the algorithm. Adjust according to needs.


for iteration in range(0, s):
    while not end:
        if (len(current_list) > 2) and (len(current_list) > k):
            state = do_clustering(current_list)

        else:
            if k > 1:
                print("\nFinal layout for " + str(k) + " clusters:")
            else:
                print("\nFinal layout for 2 clusters:")
            for everything in current_list:
                if isinstance(everything, Cluster):
                    print("\nCluster number " + str(everything.number) + " contains the following emails:" )
                    for item in everything.names:
                        print(item.number)
                else:
                    print("\nThe following email is not clustered:")
                    print(everything.number)
            print("\ndone")
            end = True

