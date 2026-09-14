import logging

from langchain.chat_models import init_chat_model

from app.settings import get_settings
from app.schemas import ModelDetail
from app.services.src.prompt_template import (
    CATEGORIZE_EMAIL_PROMPT,
    SUPERVISOR_EMAIL_VERIFICATION_PROMPT,
    few_shot_prompt,ChatPromptTemplate
)
from app.services.src.state import CategorizeEmailOutput, SupervisorOutput




logger = logging.getLogger(__name__)

import re


PROVIDER_MAP = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google_genai",
    "amazon bedrock": "bedrock_converse",
    "openrouter": "openrouter",
    "perplexity": "perplexity",
    "nvidia": "nvidia",
    "groq": "groq",
    "ollama": "ollama",
    "deepseek": "deepseek",
}


def normalize_provider(provider: str) -> str:
    provider = provider.strip().lower()

    if provider not in PROVIDER_MAP:
        raise ValueError(f"Unsupported provider: {provider}")

    return PROVIDER_MAP[provider]


def validate_model_config(model_detail: ModelDetail) -> tuple[bool, str, str]:
    """
    Pre-validate the user's model configuration before starting analysis.
    Returns (is_valid, error_code, detail_message).
    When default=True the system's own credentials are used, so no validation
    is needed on the user-supplied fields.
    """
    if model_detail.default:
        # Default mode uses the system API key — nothing to validate
        return True, "", ""

    # ── Provider check ────────────────────────────────────────────────────
    normalized = model_detail.provider.strip().lower()
    if normalized not in PROVIDER_MAP:
        supported = ", ".join(
            name.title() for name in PROVIDER_MAP.keys()
        )
        return (
            False,
            "INVALID_PROVIDER",
            f"Provider '{model_detail.provider}' is not supported. "
            f"Supported providers: {supported}",
        )

    # ── API key check ─────────────────────────────────────────────────────
    if not model_detail.api_key or not model_detail.api_key.strip():
        return (
            False,
            "MISSING_API_KEY",
            f"An API key is required when using a custom model "
            f"(provider: {model_detail.provider}, model: {model_detail.name}). "
            f"Please add your API key in Settings.",
        )

    # ── Model name check ──────────────────────────────────────────────────
    if not model_detail.name or not model_detail.name.strip():
        return (
            False,
            "INVALID_MODEL",
            "Model name cannot be empty. Please select or enter a valid model name.",
        )

    return True, "", ""



class Agent:
        
    def __init__(self, model_detail: ModelDetail, email_count: int):

        settings = get_settings()


        self.default_value = model_detail.default

        if self.default_value  and email_count <= 10:
            # Use system default model
            self.model_name = "gemini-3.5-flash-lite"
            self.model_provider = "google-genai"
            self.model_llm_api_key = settings.GOOGLE_API_KEY

        else:
            # Use requested model
            self.model_name = model_detail.name
            self.model_provider = normalize_provider(model_detail.provider)
            self.model_llm_api_key = model_detail.api_key 
            
        

        #classification agent

        
        try:
            
            if not self.model_llm_api_key:
                    raise ValueError("LLM_API_KEY is not set in the environment")

            
            llm = init_chat_model(model = self.model_name, model_provider = self.model_provider,api_key=self.model_llm_api_key)
            logger.info("Initialized classification LLM: %s (provider: %s)", self.model_name, self.model_provider)
            
            email_category_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        CATEGORIZE_EMAIL_PROMPT
                    ),

                    # inserts all examples
                    few_shot_prompt,

                    (
                        "human",
                        """
            

            # EMAIL SUBJECT
            {cleaned_subject}

            # EMAIL BODY
            {cleaned_body}

             # Available Categories with their category name, category description, category examples:
            
            {category_info}

            # Supervisor Feedback:

            {supervisor_feedback} 
            """
                    ),
                ]
            )

            self.categorize_email = email_category_prompt | llm.with_structured_output(CategorizeEmailOutput,method='json_mode',include_raw=True)
        except Exception:
             logger.exception("Failed to initialize classification agent")
             raise 


        #supervisor agent

        try:

             
            supervisor_key = settings.GOOGLE_API_KEY
            if not supervisor_key:
                  raise ValueError("SUPERVISOR_API_KEY is not set")

            
             
            suppervisor_llm = init_chat_model(model="gemma-4-26b-a4b-it",model_provider="google-genai",api_key=supervisor_key)
            logger.info("Initialized supervisor LLM: gemma-4-26b-a4b-it")
            supervisor_prompt = ChatPromptTemplate.from_messages([("system",SUPERVISOR_EMAIL_VERIFICATION_PROMPT),("human", """# Available Categories

                                                                                                                                {category_info}

                                                                                                                                # EMAIL SUBJECT

                                                                                                                                {cleaned_subject}

                                                                                                                                # EMAIL BODY

                                                                                                                                {cleaned_body}

                                                                                                                                # Classification Agent Output

                                                                                                                                Predicted Category: {predicted_category}

                                                                                                                                Confidence Score: {confidence_score}

                                                                                                                                Priority Score: {priority_score}""")])
            self.supervise_email = supervisor_prompt | suppervisor_llm.with_structured_output(SupervisorOutput,method='json_mode',include_raw=True)

        except Exception:
            logger.exception("Failed to initialize the supervisor agent")
            raise