import openai
import os

# Set up OpenAI API credentials
os.environ["openai-api-key"] = "openai-api-key"
openai.api_key = os.environ["openai-api-key"]

# Define the prompt and parameters for the API call
# Prompt: The prompt is the starting text or input that is provided to the language model to generate the output. It can be a sentence or a phrase that defines the context and tone of the output.
prompt = "write poem on 21 faburary, international Language day."
# Model: The model refers to the specific pre-trained language model that is used to generate the text. Different models have different levels of complexity and can be fine-tuned for different tasks or applications.
'''
Language models are typically trained on large amounts of text data, and different models use different techniques and architectures to learn the patterns and structures in the data. Some of the popular language models used today include OpenAI's GPT-3, Google's BERT, and Facebook's RoBERTa.
These models can vary in terms of their size and complexity, with larger models generally having more parameters and thus a higher capacity for learning complex patterns in the data. For example, GPT-3 is one of the largest and most complex language models, with 175 billion parameters, while BERT has around 340 million parameters and RoBERTa has 355 million parameters.
These models can be fine-tuned for specific tasks or applications by training them on a smaller dataset of labeled examples that are relevant to the task. For example, a language model can be fine-tuned on a dataset of news articles to generate summaries of new articles. Fine-tuning involves updating the weights of the pre-trained model based on the new labeled data, which allows the model to learn more specific patterns that are relevant to the task.
Fine-tuning can be done with various techniques such as transfer learning, where a pre-trained model is used as a starting point for a new task and then trained on additional labeled data specific to that task. It can also involve changing the hyperparameters of the model, such as the learning rate, to achieve better performance on the specific task.
Overall, the ability to fine-tune language models for specific tasks and applications is one of the reasons why they are so useful in natural language processing and machine learning.
'''
model = "text-davinci-002"
# Temperature: Temperature is a parameter that controls the "creativity" of the language model. A higher temperature results in more diverse and unpredictable output, while a lower temperature results in more conservative and predictable output.
temperature = 0.5
# Max_tokens: The max_tokens parameter defines the maximum number of tokens (words or subwords) that the language model can generate as output.It is used to prevent the model from generating excessively long or irrelevant responses.
max_tokens = 100

# Call the API to generate text based on the prompt
response = openai.Completion.create(engine=model,prompt=prompt,temperature=temperature,max_tokens=max_tokens)
# Print the generated text
print(response.choices[0].text)
