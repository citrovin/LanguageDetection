# Language Detection on the Edge

## Introduction

AI on the edge, which refers to running artificial intelligence algorithms directly on devices such as smartphones, cameras, and sensors, has become increasingly significant in recent years. This approach to AI processing can improve efficiency, reduce latency, and enhance privacy by processing data locally rather than sending it to a remote server. One important application of AI is the ability to identify spoken languages. This technology can benefit various industries such as customer service, law enforcement, and healthcare by providing real-time language translation services, improving communication, and reducing errors. Overall, AI on the edge and language identification have the potential to improve productivity, enhance user experience, and facilitate global communication. By combining these two fields of research, we found that new question arise, such as:
- How can we design a neural network, that classifies languages for use on the edge?
- How does the neural network perform on the edge compared to on a computer with
full computing power?
- How much storage does the neural network require on the edge?

In this project, we develop a neural network (NN), that is able to classify spoken language. To this end, we pre-train our model on the FLEURS data set and then fine-tune it on our own generated data set. This is done, to increase the sensitivity of our network to noise and device specific patterns of the used microphone. Subsequently, we convert the trained model to a format, which enables the NN to run on the edge, on an STM32, equipped with a microphone. To this end, we use the MicroAI framework, which uses quantization to convert our model from a Tensorflow model to a model in C. This paper will cover our approach to developing such a NN and implementing it on the edge. We commence, by explaining our methodology. Subsequently, we talk about the results of our NN running on a computer with full computing capacities and then talk about the results, that we get from running the program on the edge. The conclude, this paper with a short conclusion of our project and a short outlook on further research.

## Data

### Pre-Training

### Fine-Tuning

## On the Edge