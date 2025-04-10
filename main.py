# Chatbot Educacional com Hugging Face

# 1. Instalar as dependências
!pip install -q transformers
!pip install -q gradio

# 2. Importar bibliotecas
from transformers import pipeline
import gradio as gr

# 3. Criar o pipeline de conversa usando um modelo leve da Hugging Face
chatbot = pipeline("conversational", model="microsoft/DialoGPT-small")

# 4. Função para responder perguntas
def responder(pergunta):
    from transformers import ConversationalPipeline, Conversation
    conversa = Conversation(pergunta)
    resposta = chatbot(conversa)
    return resposta.generated_responses[-1]

# 5. Interface com Gradio
gr.Interface(fn=responder,
             inputs=gr.Textbox(label="Digite sua pergunta"),
             outputs=gr.Textbox(label="Resposta do Chatbot"),
             title="Chatbot Educacional com Hugging Face").launch()
