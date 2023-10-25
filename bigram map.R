library(ggraph)
library(dplyr)
library(tidyr)
library(igraph)
library(ggplot2)


bigram_graph <- negation_phrases_Neg %>%
  filter(frequency > 1) %>%
  graph_from_data_frame()

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)


ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = frequency, edge_width = frequency), edge_colour = "red") +
  geom_node_point(size = 2, col = "white") +
  geom_node_point(
    size = 3, col = "red"
  )+
  geom_node_text(aes(label = name), repel = TRUE, point.padding = unit(0.2, "lines")) 

