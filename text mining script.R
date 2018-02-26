# DATA 650 Assignment 1, Part II Script
# Written by Daanish Ahmed
# Semester Spring 2018
# February 10, 2018
# Professor Elena Gortcheva

# This R script involves performing a text mining analysis on the State of the 
# Union addresses between 1989 and 2017.  It involves creating several word clouds
# to analyze interesting words based on their frequencies.  Next, it involves 
# finding word associations and building correlation plots to understand the 
# relationships between words.  Finally, it will incorporate k-means clustering 
# and use the elbow method to categorize the terms into clusters.



# This section of code involves opening the dataset and initializing the packages 
# that are used in this script.

# Sets the working directory for this assignment.  Please change this directory 
# to whichever directory you are using, and make sure that all files are placed 
# in that location.
setwd("~/Class Documents/2017-18 Spring/DATA 650/Assignment 1/Part II")

# Installs the required packages (please remove the # symbol before any package 
# that you have not installed yet).
# install.packages("tm")
# install.packages("SnowballC")
# install.packages("wordcloud")
# install.packages("cluster")

# Loads the packages we need for this assignment.
library(tm)
library(SnowballC)
library(wordcloud)
library(cluster)

# Lists all of the text files in the current directory.
dir(".")

# Builds the corpus "speeches" containing all of our SOTU text files.
speeches <- Corpus(DirSource("."))

# Shows all of the files in the corpus.
summary(speeches)

# End of loading the data.



# This section of code covers data preprocessing.  We will clean up the words in the 
# texts before loading the corpus into the Document Term Matrix.

# Examines the first speech in the corpus.
inspect(speeches[1])

# These commands remove any URLs that may exist in these documents.
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
speeches <- tm_map(speeches, content_transformer(removeURL))

# Removes all numbers and punctuation, and changes all letters to lowercase.
speeches = tm_map(speeches, removeNumbers)
speeches = tm_map(speeches, removePunctuation)
speeches = tm_map(speeches, content_transformer(tolower))

# This is a list of additional stop words and unnecessary words that were not included 
# in the default stopwords lists.
stop = c("just", "good", "watch", "time", "join", "get", "big", "going", "much", "said", 
         "like", "will", "now", "new", "can", "amp", "doesnt", "gave", "means", "one", 
         "mr", "less", "from", "looking", "ago", "come", "sat", "cut", "must", "full", 
         "im", "ive", "make", "next", "give", "let", "put", "thing", "weve", "american", 
         "back", "dont", "let", "meet", "begin", "bring", "make", "set", "stay", "send", 
         "step", "stop", "open", "ask", "hold", "come", "wont", "run", "seek", "hear", 
         "lot", "theyre", "their")

# Removes stop words and unimportant words.  It identifies stop words from the English 
# and Smart stopwords lists, as well as words included in the 'stop' variable.
speeches = tm_map(speeches, removeWords, c("the", "and", stop, stopwords("english"), 
                                           stopwords("SMART")))

#Removes special characters such as @, â, and the Euro symbol.
toSpace <- content_transformer(function (x, pattern) gsub(pattern, " ", x))
speeches <- tm_map(speeches, toSpace, "@")
speeches <- tm_map(speeches, toSpace, "â")
speeches <- tm_map(speeches, toSpace, "\n")             # Removes new line character.
speeches <- tm_map(speeches, toSpace, "\u20ac")         # Removes the Euro symbol.
speeches <- tm_map(speeches, toSpace, "\u201d")         # Removes the " symbol.

# Performs stemming on each word, reducing it to its root word.
speeches <- tm_map(speeches, stemDocument)

# Removes stop words again, since some stop words may appear after stemming.
speeches = tm_map(speeches, removeWords, c("the", "and", stop, stopwords("english"), 
                                           stopwords("SMART")))

# Removes all extra whitespace between words.
speeches =  tm_map(speeches, stripWhitespace)

# Verifies that the first speech has been preprocessed.
inspect(speeches[1])

# End of data preprocessing.



# This section of code involves initializing the Document Term Matrix and word 
# frequency list.  These variables are needed for the text mining algorithms and 
# visualizations used in my analysis.

# Builds the Document Term Matrix (DTM) using the speech documents.
speech_dtm <- DocumentTermMatrix(speeches)

# Shows the initial properties of the DTM.
speech_dtm

# Creates a variable that shows the initial number of documents and unique words.
td_count <- as.matrix(speech_dtm)
dim(td_count)

# Removes terms from the DTM that appear in less than 50% of the documents.
speech_dtm <- removeSparseTerms(speech_dtm, 0.5)

# Shows the properties of the DTM after removing sparse terms.
speech_dtm

# Creates a list of all unique words and their frequency counts.
freq <- colSums(as.matrix(speech_dtm))

# Shows the number of unique words remaining.
length(freq)

# Displays all terms that appear at least 100 times.
findFreqTerms(speech_dtm, lowfreq=100)

# End of variable initialization.



# This section covers the creation of word clouds to visualize the most frequent 
# terms appearing in the SOTU addresses.

# Color scheme using up to 6 different colors for words depending on their frequency.
dark2 <- brewer.pal(6, "Dark2")

# Builds a word cloud that colors terms according to their frequency.  It shows words 
# that appear at least 150 times, with a maximum of 40 words.
set.seed(12345)           # Random seed to reproduce the results.
wordcloud(names(freq), freq, min.freq=150, max.words=40, rot.per=0.2, colors=dark2)

# The following segments of code create a word cloud using Term Frequency-Inverse 
# Document Frequency (TF-IDF).  Rather than showing words based on their frequency, 
# this method attempts to evaluate terms based on their importance within the document.

# Creates a TF-IDF matrix to find more relevant terms.
speech_dtm_tfidf <- DocumentTermMatrix(speeches, control = list(weighting = weightTfIdf))

# Removes sparse terms from the TF-IDF matrix.
speech_dtm_tfidf = removeSparseTerms(speech_dtm_tfidf, 0.5)

# Obtains the most frequent words, sorted in descending order by count.
freqTerms = data.frame(sort(colSums(as.matrix(speech_dtm_tfidf)), decreasing=TRUE))

# Creates a new word cloud using TF-IDF, with a maximum of 30 words.
set.seed(12345)           # Random seed to reproduce the results.
wordcloud(rownames(freqTerms), freqTerms[,1], max.words=30, random.order=TRUE, 
          colors=brewer.pal(3, "Dark2"))

# End of generating word clouds.



# This section of code covers association mining, which involves finding words that 
# commonly appear together.  It is useful for understanding the attitudes that 
# presidents have towards issues such as jobs, terrorism, debt, immigration, etc.

# Finds common words associated with jobs.
findAssocs(speech_dtm, term = "job", 0.6)

# Finds common words associated with schools.
findAssocs(speech_dtm, term = "school", 0.65)

# Finds common words associated with health.
findAssocs(speech_dtm, term = "health", 0.6)

# Finds common words associated with the economy, taxes, and debt.
findAssocs(speech_dtm, c("economi", "tax", "debt"), 0.5)

# Finds common words associated with war and terrorism.
findAssocs(speech_dtm, c("war", "terror"), c(0.4, 0.6))

# Finds common words associated with immigration.
findAssocs(speech_dtm, term = "immigr", 0.5)

# End of association mining section.



# This section involves building two correlation plots that show the correlations 
# between words (indicated by lines connecting the words).  One of these plots is 
# weighted, for which the line's width indicates the strength of the correlation.

# The following commands set up the "Rgraphviz" package.  These commands only need
# to be run once per session.
source("http://bioconductor.org/biocLite.R")
biocLite("Rgraphviz")

# Builds a correlation plot containing 10 words that appear at least 150 times, with 
# a correlation threshold of 0.4.
plot(speech_dtm, terms=findFreqTerms(speech_dtm, lowfreq = 150)[1:10], corThreshold = 0.4)

# Builds a weighted correlation plot containing 6 words that appear at least 150 times, 
# with a correlation threshold of 0.2.  Line thickness indicates the strength of the 
# correlation.
plot(speech_dtm, terms=findFreqTerms(speech_dtm, lowfreq = 150)[1:6], corThreshold = 0.2, 
     weighting = T)

# End of generating correlation plots.



# This section of code covers the implementation of the k-means clustering algorithm. 
# We will first use k=4 clusters.  This value is obtained by using k = sqrt(n/2), 
# where n equals the number of documents (29).

# We will only use words that appear in all documents, removing all other terms.
speech_dtm2 <- removeSparseTerms(speech_dtm, 0.01)

# Builds the dissimilarity matrix, which is used to create the clustering model.
dsm <- dist(t(speech_dtm2), method="euclidian")

# Creates the clustering model using k=4 clusters.
kfit <- kmeans(dsm, 4)

# Shows the properties of the clustering model.
kfit

# Displays the cluster plot.
clusplot(as.matrix(dsm), kfit$cluster, color=T, shade=T, labels=2, lines=0)

# End of k-means clustering model with k=4.



# This section involves using the elbow method to obtain a more ideal k-value. 
# After using this method, we will perform k-means clustering again using the 
# new k-value.

# Sets the margins for the elbow method plot.
par(mar=c(4, 4, 4, 4))

# Plots the between clusters sum-of-squares by k value.
bss <- integer(length(2:15))
for (i in 2:15) bss[i] <- kmeans(dsm, centers=i)$betweenss
plot(1:15, bss, type="b", xlab="Number of Clusters",
     ylab="Sum of squares", col="blue")

# Plots the within clusters sum-of-squares by k value.
wss <- integer(length(2:15))
for (i in 2:15) wss[i] <- kmeans(dsm, centers=i)$tot.withinss
lines(1:15, wss, type="b")

# The plot suggests that the "elbow" is at approximately k=3.  Thus, we will 
# build the clustering model again using 3 clusters.
kfit <- kmeans(dsm, 3)

# Shows the properties of the new clustering model.
kfit

# Displays the new cluster plot.
clusplot(as.matrix(dsm), kfit$cluster, color=T, shade=T, labels=2, lines=0)

# End of k-means clustering model using the elbow method.



# End of script.

