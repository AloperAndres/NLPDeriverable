
# Topic Modelling applied to SOTU corpus
# By Andres Alonso Perez

#1 Imports of the libraries needed ________________________________________

library(tm)
library(ggplot2)
library(wordcloud)
library(RWeka)
library(reshape2)
library(SnowballC)
library(topicmodels)
library(tidytext)
library(dplyr)
library(ldatuning)
library(stm)

#2 Setting the work directory and charging the corpus __________________________

# Change this directory as needed
setwd("C:/Users/Usuario/Documents/1 2 UPM ETSI Inf - MSc in Data Science/Intelligent Systems/NLP/Union")
source.pos = DirSource("./sotu", encoding = "UTF-8")
corpus = Corpus(source.pos)


#3 First address of the corpus: ________________________________________________

inspect(corpus[[1]])


#4 Document Term Matrices ______________________________________________________

dtm_non_filtered = DocumentTermMatrix(corpus)
dtm_non_filtered

# More cleaned dtm
dtm = DocumentTermMatrix(corpus,
                         control=list(stopwords = T,
                                      removePunctuation = T, 
                                      removeNumbers = T,
                                      stemming = T))
dtm

# Freq plots of the words
freq=colSums(as.matrix(dtm))
head(sort(freq, decreasing = T),40)
plot(sort(freq, decreasing = T),col="blue",main="Word frequencies", xlab="Frequency-based rank", ylab = "Frequency")

# Elimination of empty documents, most and less common words
dtm2 = dtm[rowSums(as.matrix(dtm))!=0,(freq<quantile(freq, 0.999)) & (freq>4)]

# New freq plot
freq<-colSums(as.matrix(dtm2))
head(sort(freq, decreasing = T),40)
plot(sort(freq, decreasing = T),col="blue",main="Word frequencies", xlab="Frequency-based rank", ylab = "Frequency")
# head(freq,10)
# tail(freq,10)

#5 Wordcloud ___________________________________________________________________

pal=brewer.pal(8,"Reds")
pal=pal[-(1:3)]
word.cloud=wordcloud(words=names(freq), freq=freq,
                     min.freq=1000, random.order=F, colors=pal)


#6 Topic modelling algorithms___________________________________________________

# Finding the optimal k for LDA

result <- FindTopicsNumber(
  dtm2,
  topics = 4:18,
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  mc.cores = 6L
)

# Plot of the metrics
FindTopicsNumber_plot(result)

# ___6.1 Creating LDA model_____________
lda.model = LDA(dtm2, 16)

# Most common terms of each topic
as.data.frame(terms(lda.model, 10))

# Beta matrix ordered by topic and beta
text_topics <- tidy(lda.model, matrix = "beta")
text_top_terms <- text_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Plot of the beta values of the most common words for each topic
text_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  #geom_col(show.legend = FALSE) +
  geom_bar(stat="identity", show.legend = FALSE)+
  labs(y="beta", x="term")+
  facet_wrap(~ topic, scales = "free")+
  coord_flip()+
  scale_x_reordered()

# Gamma matrix (presence of each topic in each document)
ap_documents <- tidy(lda.model, matrix = "gamma")
ap_documents

# Topics share for each document
data.frame(Topic = topics(lda.model))

# ___6.2 Creating STM model_____________
inmodel<-readCorpus(dtm2)
stm.model = stm(inmodel$documents, inmodel$vocab, K=16)
as.data.frame(t(labelTopics(stm.model, n = 10)$prob))

# Beta matrix ordered by topic and beta
text_topics2<- tidy(stm.model, matrix = "beta")
text_topics2
text_top_terms2 <- text_topics2 %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Plot of the beta values of the most common words for each topic
text_top_terms2 %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  #geom_col(show.legend = FALSE) +
  geom_bar(stat="identity", show.legend = FALSE)+
  labs(y="beta", x="term")+
  facet_wrap(~ topic, scales = "free")+
  coord_flip()+
  scale_x_reordered()

# Gamma matrix (LDA)
lda_gamma <- as.data.frame(ap_documents)
lda_gamma$row <- dtm2$dimnames$Docs

# Plot of topic share for each document (LDA)
ggplot(lda_gamma, aes(y = gamma, x = row, fill = topic)) + 
  geom_bar(stat = "identity") +
  xlab("Documents") +
  ylab("Topic") +
  guides(fill = FALSE) +
  theme_bw()+
  coord_flip()

# Gamma matrix (STM)
stm_gamma <- as.data.frame(tidy(stm.model, matrix = "gamma"))
stm_gamma$row <- dtm2$dimnames$Docs

# Plot of topic share for each document (LDA)
ggplot(stm_gamma, aes(y = gamma, x = row, fill = topic)) + 
  geom_bar(stat = "identity") +
  xlab("Documents") +
  ylab("Topic") +
  guides(fill = FALSE) +
  theme_bw()+
  coord_flip()

# Theta matrix for STM (probabilities of the topics for each document)
stm_theta <- as.data.frame(stm.model$theta)
stm_theta$row <- dtm2$dimnames$Docs
dat2 <- melt(stm_theta, id.vars = "row")

#Plot of the theta matrix
ggplot(dat2, aes(y = value, x = row, fill = variable)) + 
  geom_bar(stat = "identity") +
  xlab("Documents") +
  ylab("Topic") +
  guides(fill = FALSE) +
  theme_bw()+
  coord_flip()