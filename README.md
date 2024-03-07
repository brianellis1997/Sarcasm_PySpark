# Sarcasm_PySpark
Sarcasm Detection using distributed computing with PySpark. 
Computing done using Penn State's ROAR supercomputer.
## Introduction
Sentiment analysis with large language models has become an important area of study in
recent years for different stakeholders. Specifically, the sentiment analysis of consumer data can
allow companies to better understand customer opinions and preferences which can lead to better
marketing strategies and business improvement. However, sarcasm is causing misconceptions
about the sentiment in customer data, impacting the accuracy of the insights taken from these
analyses. Due to this issue, researchers have attempted to create accurate sarcasm detection deep
learning models in many different domains. These attempts include architectures based on feed
forward networks, RNNs, CNN’s, Transformers, as well as hybrid models that use different
natural language processing techniques to aid performance (Verma et al. 2021). Despite these
attempts, there is still a need for further research and implementation in this area.

It would be nice to quantify the model's confidence in it's classification. For example, instead of outputting binary sarcastic labels, it can give probabilistic estimates such as 70% sarcasm.