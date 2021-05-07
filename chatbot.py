from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import spacy


nlp = spacy.load("en_core_web_sm")

# Creating ChatBot Instance
chatbot = ChatBot(
    'HealthBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=
    [

        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {

            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',#if there is any message chatbot doesnot understand
            'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
)
training_data_quesans = open('training_data/ques_ans.txt',encoding='utf-8').read().splitlines()
training_data_personal = open('training_data/personal_ques.txt',encoding='utf-8').read().splitlines()

training_data = training_data_quesans + training_data_personal

trainer = ListTrainer(chatbot)
trainer.train(training_data)

# Training with English Corpus Data
trainer_corpus = ChatterBotCorpusTrainer(chatbot)
trainer_corpus.train(
    'chatterbot.corpus.english'
)