"""
LLM client wrapper supporting multiple backends (Google GenAI, OpenAI, Anthropic).
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Unified LLM client supporting multiple backends.
    
    Parameters
    ----------
    api_key : str, optional
        API key for the LLM service. If None, reads from environment
    model_name : str
        Model identifier (e.g., 'gemini-2.0-flash-exp', 'gpt-4o', 'claude-sonnet-4')
    provider : str, optional
        Force specific provider ('google', 'openai', 'anthropic'). 
        If None, infers from model_name
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'gemini-2.0-flash-exp',
        provider: Optional[str] = None
    ):
        self.model_name = model_name
        self.provider = provider or self._infer_provider(model_name)
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()
    
    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        if 'gemini' in model_name.lower():
            return 'google'
        elif 'gpt' in model_name.lower():
            return 'openai'
        elif 'claude' in model_name.lower():
            return 'anthropic'
        else:
            raise ValueError(
                f"Cannot infer provider from model_name '{model_name}'. "
                "Please specify provider explicitly."
            )
    
    def _get_api_key(self) -> str:
        """Get API key from environment based on provider."""
        env_vars = {
            'google': 'GENAI_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }
        
        env_var = env_vars.get(self.provider)
        if not env_var:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key not found. Set {env_var} environment variable."
            )
        
        return api_key
    
    def _initialize_client(self):
        """Initialize the appropriate client."""
        if self.provider == 'google':
            from google import genai
            return genai.Client(api_key=self.api_key)
        elif self.provider == 'openai':
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        elif self.provider == 'anthropic':
            from anthropic import Anthropic
            return Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500
    ) -> str:
        """
        Generate text using the LLM.
        
        Parameters
        ----------
        prompt : str
            User prompt
        system_message : str, optional
            System instruction
        temperature : float
            Sampling temperature
        max_tokens : int
            Maximum tokens to generate
            
        Returns
        -------
        str
            Generated text
        """
        if system_message is None:
            system_message = (
                "You are a precise and detail-oriented assistant specializing "
                "in analyzing text for specific concepts and constructs."
            )
        
        if self.provider == 'google':
            return self._generate_google(prompt, system_message, temperature)
        elif self.provider == 'openai':
            return self._generate_openai(prompt, system_message, temperature, max_tokens)
        elif self.provider == 'anthropic':
            return self._generate_anthropic(prompt, system_message, temperature, max_tokens)
    
    def _generate_google(
        self,
        prompt: str,
        system_message: str,
        temperature: float
    ) -> str:
        """Generate using Google GenAI."""
        from google.genai import types
        
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=temperature
            ),
            contents=prompt
        )
        return response.text
    
    def _generate_openai(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Anthropic."""
        response = self.client.messages.create(
            model=self.model_name,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text


def query_llm(
    prompt: str,
    model: str,
    system_message: Optional[str] = None,
    client: Optional[LLMClient] = None,
    **kwargs
) -> str:
    """
    Convenience function to query an LLM.
    
    Parameters
    ----------
    prompt : str
        User prompt
    model : str
        Model name
    system_message : str, optional
        System instruction
    client : LLMClient, optional
        Existing client to reuse
    **kwargs
        Additional arguments for generation
        
    Returns
    -------
    str
        Generated text
    """
    if client is None:
        client = LLMClient(model_name=model)
    
    return client.generate(prompt, system_message, **kwargs)