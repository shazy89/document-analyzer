from document_analyzer.services.together_client import TogetherChatService

class PromptBuilder:
    def __init__(self, system_prompt: str, service: TogetherChatService):
        self.system_prompt = system_prompt
        self.service = service

    def input_prompt(self, user_prompt: str) -> str:
        return f"{self.system_prompt}\n\nUser: {user_input}\nAssistant:"