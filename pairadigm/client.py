import os
from typing import Any, Dict, Optional

class LLMClient:
    """
    Unified LLM client supporting multiple backends.
    
    Parameters
    ----------
    api_key : str, optional
        API key for the LLM service. If None, reads from environment
    model_name : str
        Model identifier (e.g., 'gemini-3.8-flash', 'gpt-6-luna',
        'claude-sonnet-5-5', 'llama3.2', 'meta-llama/Llama-3.3-70B-Instruct')
    base_url : str, optional
        Base URL for the LLM service API (default: 'http://localhost:11434' for Ollama)
    provider : str, optional
        Force specific provider ('google', 'openai', 'anthropic', 'ollama', 'huggingface'). 
        If None, infers from model_name
    reasoning : str, bool, int, or dict, optional
        Per-client reasoning control. Named levels are translated to the
        provider's parameter (for example, ``reasoning_effort`` for OpenAI,
        ``thinking_level`` for Google, and ``effort`` for Anthropic). ``None``
        leaves the provider/model default unchanged. Dictionaries may contain
        provider-native reasoning parameters.
    """
    
    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = 'gemini-3.8-flash',
            base_url: Optional[str] = None,
            provider: Optional[str] = None,
            reasoning: Optional[Any] = None):

        self.model_name = model_name
        self.provider = provider or self._infer_provider(model_name)
        self.reasoning = reasoning
        self.base_url = base_url or self._get_default_base_url()
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()
    
    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        model_lower = model_name.lower()
        
        # Check for HuggingFace patterns (model names with / or common HF orgs)
        hf_patterns = ['/', 'meta-llama', 'mistralai', 'tiiuae', 'bigscience', 'eleutherai']
        if any(pattern in model_name for pattern in hf_patterns):
            return 'huggingface'
        
        # Check for Ollama-specific patterns (colon indicates local model tag)
        if ':' in model_name:
            return 'ollama'
        
        # Check for known Ollama model names
        ollama_models = ['llama', 'mistral', 'phi', 'qwen', 'gemma', 'deepseek', 'vicuna', 'orca']
        if any(x in model_lower for x in ollama_models):
            return 'ollama'
        
        # Then check for cloud providers
        if 'gemini' in model_lower:
            return 'google'
        elif (
            model_lower.startswith('gpt-') and 'gpt-oss' not in model_lower
        ) or (
            model_lower.startswith('o')
            and len(model_lower) > 1
            and model_lower[1].isdigit()
        ):
            return 'openai'
        elif 'claude' in model_lower:
            return 'anthropic'
        else:
            # Default to ollama for unknown models (likely local)
            return 'ollama'
    
    def _get_default_base_url(self) -> Optional[str]:
        """Get default base URL based on provider."""
        if self.provider == 'ollama':
            return 'http://localhost:11434'
        return None
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment based on provider."""
        # Ollama doesn't require an API key
        if self.provider == 'ollama':
            return None
            
        env_vars = {
            'google': ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GENAI_API_KEY'),
            'openai': ('OPENAI_API_KEY',),
            'anthropic': ('ANTHROPIC_API_KEY',),
            'huggingface': ('HUGGINGFACE_API_KEY', 'HF_TOKEN')
        }
        
        provider_env_vars = env_vars.get(self.provider)
        if not provider_env_vars:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        for env_var in provider_env_vars:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key

        env_var_names = " or ".join(provider_env_vars)
        raise ValueError(
            f"API key not found. Set {env_var_names} environment variable."
        )

    def _reasoning_params(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Translate the client's reasoning setting to provider request fields."""
        reasoning = self.reasoning
        if reasoning is None:
            return {}

        if isinstance(reasoning, dict):
            if self.provider == 'google':
                if 'thinking_config' in reasoning:
                    return dict(reasoning)
                return {'thinking_config': dict(reasoning)}
            if self.provider == 'anthropic':
                if 'thinking' in reasoning or 'output_config' in reasoning:
                    return dict(reasoning)
                return {'thinking': dict(reasoning)}
            if self.provider == 'ollama' and 'think' not in reasoning:
                return {'think': dict(reasoning)}
            if self.provider == 'huggingface':
                if 'extra_body' in reasoning:
                    return dict(reasoning)
                return {'extra_body': dict(reasoning)}
            return dict(reasoning)

        if self.provider == 'openai':
            if isinstance(reasoning, bool):
                reasoning = 'medium' if reasoning else 'none'
            if not isinstance(reasoning, str):
                raise TypeError(
                    "OpenAI reasoning must be a named level, bool, dict, or None."
                )
            return {'reasoning_effort': reasoning}

        if self.provider == 'huggingface':
            if isinstance(reasoning, bool):
                reasoning = 'medium' if reasoning else 'none'
            if not isinstance(reasoning, str):
                raise TypeError(
                    "Hugging Face reasoning must be a named level, bool, dict, or None."
                )
            return {'extra_body': {'reasoning_effort': reasoning}}

        if self.provider == 'google':
            if isinstance(reasoning, bool):
                return {'thinking_config': {'thinking_budget': -1 if reasoning else 0}}
            if isinstance(reasoning, int):
                return {'thinking_config': {'thinking_budget': reasoning}}
            if isinstance(reasoning, str):
                return {'thinking_config': {'thinking_level': reasoning.upper()}}
            raise TypeError("Google reasoning must be a named level, bool, int, dict, or None.")

        if self.provider == 'anthropic':
            if isinstance(reasoning, bool):
                return {'thinking': {'type': 'adaptive' if reasoning else 'disabled'}}
            if isinstance(reasoning, int):
                if reasoning < 1024:
                    raise ValueError("Anthropic reasoning budgets must be at least 1024 tokens.")
                if max_tokens is not None and reasoning >= max_tokens:
                    raise ValueError(
                        "Anthropic reasoning budget must be less than max_tokens."
                    )
                return {
                    'thinking': {'type': 'enabled', 'budget_tokens': reasoning}
                }
            if isinstance(reasoning, str):
                return {
                    'thinking': {'type': 'adaptive'},
                    'output_config': {'effort': reasoning},
                }
            raise TypeError("Anthropic reasoning must be a named level, bool, int, dict, or None.")

        if self.provider == 'ollama':
            if not isinstance(reasoning, (str, bool)):
                raise TypeError("Ollama reasoning must be a named level, bool, dict, or None.")
            return {'think': reasoning}

        return {}
    
    def _initialize_client(self):
        """Initialize the appropriate client."""
        if self.provider == 'google':
            from google import genai
            return genai.Client(api_key=self.api_key)
        
        elif self.provider == 'openai':
            from openai import OpenAI
            if self.base_url:
                return OpenAI(api_key=self.api_key, base_url=self.base_url)
            return OpenAI(api_key=self.api_key)
        
        elif self.provider == 'anthropic':
            from anthropic import Anthropic
            return Anthropic(api_key=self.api_key)
        
        elif self.provider == 'ollama':
            import ollama
            # Use native Ollama client
            return ollama.Client(host=self.base_url)
        
        elif self.provider == 'huggingface':
            from huggingface_hub import InferenceClient
            return InferenceClient(token=self.api_key)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
            self,
            prompt: str,
            system_message: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: int = 1000,
            **kwargs) -> str:
        """
        Generate text using the LLM.
        
        Parameters
        ----------
        prompt : str
            User prompt
        system_message : str, optional
            System instruction
        temperature : float
            Sampling temperature.  If the model does not accept a temperature
            parameter (e.g., some reasoning models), the call is retried without
            it and the model is added to an internal no-temperature cache so
            future calls skip the retry.
        max_tokens : int
            Maximum tokens to generate
            
        Returns
        -------
        str
            Generated text
        """
        if system_message is None:
            # Plan 3g: Hardcoded system messages should be module-level constants
            system_message = _DEFAULT_BREAKDOWN_SYSTEM_MSG
        
        # 9a-temp: track models that don't support temperature to skip retries
        if not hasattr(self, "_no_temperature"):
            self._no_temperature = False

        effective_temperature = None if self._no_temperature else temperature

        try:
            return self._dispatch(prompt, system_message, effective_temperature, max_tokens, **kwargs)
        except Exception as exc:
            # Retry without temperature if the error looks temperature-related
            exc_str = str(exc).lower()
            if (
                not self._no_temperature
                and any(kw in exc_str for kw in ("temperature", "unsupported param", "invalid param"))
            ):
                import warnings as _warnings
                _warnings.warn(
                    f"Model '{self.model_name}' rejected the temperature parameter ({exc}). "
                    "Retrying without temperature — future calls will skip it automatically.",
                    UserWarning,
                    stacklevel=2,
                )
                self._no_temperature = True
                return self._dispatch(prompt, system_message, None, max_tokens, **kwargs)
            raise

    def _dispatch(
            self,
            prompt: str,
            system_message: str,
            temperature: Optional[float],
            max_tokens: int,
            **kwargs) -> str:
        """Route to the correct provider implementation."""
        if self.provider == 'google':
            return self._generate_google(
                prompt, system_message, temperature, max_tokens, **kwargs
            )
        
        elif self.provider == 'openai':
            return self._generate_openai(prompt, system_message, temperature, max_tokens, **kwargs)
        
        elif self.provider == 'anthropic':
            return self._generate_anthropic(prompt, system_message, temperature, max_tokens, **kwargs)
        
        elif self.provider == 'ollama':
            return self._generate_ollama(prompt, system_message, temperature, max_tokens, **kwargs)
        
        elif self.provider == 'huggingface':
            return self._generate_huggingface(prompt, system_message, temperature, max_tokens, **kwargs)
    
    def _generate_google(
            self,
            prompt: str,
            system_message: str,
            temperature: Optional[float],
            max_tokens: int,
            **kwargs) -> str:
        """Generate using Google GenAI."""
        from google.genai import types

        config_params = {
            "system_instruction": system_message,
            "max_output_tokens": max_tokens,
        }
        if temperature is not None:
            config_params["temperature"] = temperature
        config_params.update(self._reasoning_params(max_tokens))
        config_params.update(kwargs)

        thinking_config = config_params.get("thinking_config")
        thinking_config_type = getattr(types, "ThinkingConfig", None)
        thinking_config_fields = getattr(thinking_config_type, "model_fields", {})
        if (
            isinstance(thinking_config, dict)
            and "thinking_level" in thinking_config
            and "thinking_level" not in thinking_config_fields
        ):
            raise RuntimeError(
                "Named Google reasoning levels require google-genai>=1.51.0. "
                "Upgrade google-genai or pass reasoning as an integer token budget."
            )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(**config_params),
            contents=prompt
        )
        return response.text
    
    def _generate_openai(
            self,
            prompt: str,
            system_message: str,
            temperature: Optional[float],
            max_tokens: int,
            **kwargs) -> str:
        """Generate using OpenAI."""
        
        # Newer models use max_completion_tokens instead of max_tokens
        model_lower = self.model_name.lower()
        completion_token_prefixes = ('gpt-4o', 'gpt-5', 'gpt-6', 'o1', 'o3', 'o4')
        uses_completion_tokens = model_lower.startswith(completion_token_prefixes)
        
        params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
        }
        
        if temperature is not None:
            params["temperature"] = temperature
        
        if uses_completion_tokens:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        params.update(self._reasoning_params(max_tokens))
        params.update(kwargs)
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_message: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Anthropic."""
        
        params = {
            "model": self.model_name,
            "system": system_message,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens
        }
        
        if temperature is not None:
            params["temperature"] = temperature

        params.update(self._reasoning_params(max_tokens))
        params.update(kwargs)
        
        response = self.client.messages.create(**params)
        for block in response.content:
            if getattr(block, 'type', None) == 'text':
                return block.text
        raise RuntimeError("Anthropic response did not contain a text block.")
    
    def _generate_ollama(
        self,
        prompt: str,
        system_message: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Ollama."""
        options = {
            "max_tokens": max_tokens,
        }
        
        if temperature is not None:
            options["temperature"] = temperature
            
        options.update(kwargs)

        params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "options": options,
            "stream": False,
        }
        params.update(self._reasoning_params(max_tokens))
        response = self.client.chat(**params)
        
        return response['message']['content']
    
    def _generate_huggingface(
        self,
        prompt: str,
        system_message: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using HuggingFace Hub Inference API."""
        # Format messages for chat completion
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens
            }
            
            if temperature is not None:
                params["temperature"] = temperature

            params.update(self._reasoning_params(max_tokens))
            params.update(kwargs)
            
            # Use OpenAI-compatible chat completions endpoint
            response = self.client.chat.completions.create(**params)
            
            # Extract text from response
            return response.choices[0].message.content
                
        except Exception as e:
            raise RuntimeError(f"HuggingFace inference failed: {e}")

_DEFAULT_BREAKDOWN_SYSTEM_MSG = (
    "You are a precise and detail-oriented assistant working to uncover nuance "
    "in data. Respond to the prompt concisely. Restate the core ask/idea of the prompt "
    "in your response (without repeating it). Do not include any additional commentary, questions, or information."
)
