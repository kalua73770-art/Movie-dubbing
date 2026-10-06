from hindi_dubbing.openrouter import OpenRouterService
from hindi_dubbing.settings import settings

if __name__ == "__main__":
    print("OpenRouter smoke:", OpenRouterService(settings).smoke_models())
