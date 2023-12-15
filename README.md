# FRIENDS-Chatbot

## Project Description:
The project aims to develop a conversational AI model capable of generating coherent and contextually relevant responses in natural language. The goal is to create an intelligent chatbot that can produce conversations involving Monica, Chandler, Joey, Ross and Phoebe from the popular TV show FRIENDS. 

The motivation behind this project is to fine-tune a pretrained model for a downstream task. Traditional chatbots often struggle with context retention and generating human-like responses. This project seeks to address these challenges by leveraging advanced natural language processing (NLP) techniques and state-of-the-art deep learning models.

## Model Architecture:
The model architecture is based on the Transformer architecture, a powerful sequence-to-sequence model introduced in the paper "Attention is All You Need" by Vaswani et al. Transformers have proven to be highly effective in various NLP tasks due to their parallelization capabilities and attention mechanisms.

Specifically, the model employed in this project is a variant of the GPT (Generative Pre-trained Transformer) architecture, named DialoGPT. GPT is a generative language model that is pre-trained on vast amounts of diverse text data. DialoGPT is trained on a huge dataset involving conversations from reddit and other social media platform. The fine tuning allows the model to capture intricate language patterns and contextual information, enabling it to generate coherent and contextually appropriate responses during the fine-tuning stage.

The GPT architecture is characterized by its autoregressive nature, where the model predicts the next word in a sequence based on the preceding context. This autoregressive decoding mechanism makes it well-suited for conversational AI applications.

### Approach to Problem Solving:
The first step is to create a dataset that has contextual awareness. Follow the code for an indepth experience. Each datapoint label is supported by n number of previous dialogues to create context and generate awareness for the model.

The model undergoes fine-tuning on a dataset specifically tailored to the desired conversational domain. The fine-tuning process refines the model's responses and adapts it to the specific nuances of the target application, ensuring it generates contextually relevant and coherent replies.

To evaluate the model's performance, metrics such as perplexity and BLEU scores may be employed. Additionally, user testing and feedback can be valuable for assessing the chatbot's real-world effectiveness and identifying areas for improvement.

In summary, the project integrates advanced transformer-based architecture with pre-training and fine-tuning strategies to develop a state-of-the-art conversational AI model. The ultimate aim is to create a chatbot that not only understands the intricacies of natural language but also engages users in dynamic and contextually rich conversations.