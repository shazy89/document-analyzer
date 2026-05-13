from .config import DocumentAnalyzerConfig


class DocumentAnalyzerAgent:
    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder
    
