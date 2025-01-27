from langchain_ollama import ChatOllama
from langchain_core.messages.ai import AIMessage
from ray import serve
from starlette.requests import Request
@serve.deployment(num_replicas=1, ray_actor_options={'num_cpus': 0.5})
class JokeGenerator:
    def __init__(self):
        self.llm = ChatOllama(model='gemma:2b')

    def generate_response(self, input_text):
        response = self.llm.invoke(input_text)
        content = response.content
        return content
    
    async def __call__(self, http_request: Request):
        joke = await http_request.json()
        return self.generate_response(joke)

joke = JokeGenerator.bind()
serve.run(joke, route_prefix='/joke', blocking =True)