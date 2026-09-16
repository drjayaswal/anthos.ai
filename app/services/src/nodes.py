import logging
import asyncio
import re
import html2text

from app.services.src.agents import Agent
from app.services.src.state import MainGraphState

logger = logging.getLogger(__name__)

class EmailCleaner:

    def __init__(self):
        pass  # parser is created fresh per call to avoid shared state issues

    def _make_parser(self) -> html2text.HTML2Text:
        """Create a fresh html2text parser for each call (avoids shared state corruption)."""
        parser = html2text.HTML2Text()
        parser.ignore_images = True
        parser.ignore_links = True
        parser.ignore_emphasis = True
        return parser

    def clean_subject_and_body(self,text:str) -> str:

        if not text:
            return ""

        decoded_text = text.encode("utf-16", "surrogatepass").decode("utf-16")

        # Try html2text first; fall back to regex strip if the parser
        # crashes on malformed HTML (AssertionError on Python 3.14).
        try:
            parser = self._make_parser()
            parsed_text = parser.handle(decoded_text)
        except (AssertionError, Exception):
            # Fallback: strip HTML tags with a simple regex
            parsed_text = re.sub(r"<[^>]+>", " ", decoded_text)

        cleaned_text = re.sub(r"[^a-zA-Z0-9]+", " ", parsed_text)
        result = " ".join(cleaned_text.split())

        return result




class Nodes:

    MAX_CONCURRENT_LLM_CALLS = 2

    def __init__(self):

        self.cleaner = EmailCleaner()
        self.llm_with_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_LLM_CALLS)

    

    
    def pre_processing_raw_emails(self,state: MainGraphState) -> MainGraphState:


        email = state["one_email"]
        logger.info("Cleaning email %s", email.id)
        try:

            email.cleaned_subject = self.cleaner.clean_subject_and_body(email.subject)
            email.cleaned_body = self.cleaner.clean_subject_and_body(email.body)
        except Exception:
            logger.exception("Failed to clean email %s", email.id)
            raise

        return {"one_email": email}


    def user_email_category_config(self,state:MainGraphState):
        
        user_dynamic_categories = state["user_defined_email_categories"] #list

        category_info = [
            f"""
        category_name : {category.name},
        category_description :{category.description},
        category_examples: {category.examples}
        """
        for category in user_dynamic_categories]
        
        category_info = " ".join(category_info)
        return category_info
    
    
    async def llm_categorization(self,state:MainGraphState) -> MainGraphState:

        model_detail = state["model_detail"]
        email_count = state["email_count"]

        agent = Agent(model_detail=model_detail,email_count=email_count)


        email = state["one_email"]

        category_info = self.user_email_category_config(state)

        supervisor_output = state.get("supervisor_output")
        logger.info("Running LLM categorization for email %s (retry_counter=%s)", email.id, state.get("retry_counter"))

        try:

            async with self.llm_with_semaphore:
                logger.info(
                "Acquired LLM semaphore")




                response = await agent.categorize_email.ainvoke({
                    "category_info" : category_info,
                    "supervisor_feedback": supervisor_output.feedback if supervisor_output else None,
                    "cleaned_subject" : email.cleaned_subject,
                    "cleaned_body" : email.cleaned_body})
                
                
                if response["parsing_error"]:
                # include_raw=True turns parsing failures into a normal
                # return value instead of an exception - raise it manually
                # so RetryPolicy(max_attempts=3) still retries on failure.
                  raise response["parsing_error"]
                result = response["parsed"]
                
        except Exception:
                logger.exception("LLM categorization failed for email %s", email.id)
                raise

        
        email.category = result.category
        email.priority_score = result.priority_score
        email.confidence_score = result.confidence_score
        email.summary = result.summary

        logger.info("Email %s categorized as '%s' (confidence=%.2f, priority=%.1f)",
                    email.id, result.category, result.confidence_score, result.priority_score)


        return {"one_email" : email}
    
    async def llm_supervisor(self,state:MainGraphState) -> MainGraphState:
        email = state["one_email"]
        model_detail = state["model_detail"]
        email_count = state["email_count"]
        
        agent = Agent(model_detail=model_detail,email_count=email_count)
        supervisor = state["supervisor_output"]
        logger.info("Running supervisor review for email %s", email.id)

        category_info = self.user_email_category_config(state)

        try:

            async with self.llm_with_semaphore:

                logger.info(
                "Acquired LLM semaphore")


                response = await agent.supervise_email.ainvoke({
                    "category_info" : category_info,
                    "predicted_category": email.category,
                    "confidence_score": email.confidence_score,
                    "priority_score": email.priority_score,
                    "cleaned_subject": email.cleaned_subject,
                    "cleaned_body": email.cleaned_body})
                if response["parsing_error"]:
                  raise response["parsing_error"]
                supervise = response["parsed"]
                
        except Exception:
                logger.exception("Supervisor review failed for email %s", email.id)
                raise
            
        supervisor.feedback = supervise.feedback
        supervisor.approved = supervise.approved

        logger.info("Supervisor %s email %s |","approved" if supervise.approved else "rejected", email.id)


        return {"supervisor_output": supervisor}
    
    def classification_router(self,state:MainGraphState) -> str:

        supervisor_output = state["supervisor_output"]
        

        if supervisor_output.approved:

            return "Approved"
        if state["retry_counter"] >= 2:
            logger.warning("Email %s reached max retries without approval", state["one_email"].id)
            return "max_retry_reached"
        
        return "reclassify"

        
    def retry_and_versioning(self,state:MainGraphState)-> MainGraphState:

        retry_count = state["retry_counter"] + 1
        version_list = state["version_list"].copy()
        version_list.append(len(version_list) + 1)
        logger.info("Reclassifying email %s: attempt %d, new version %d",
                    state["one_email"].id, retry_count, version_list[-1])


        return {"retry_counter":retry_count , "version_list":version_list}
            
            

            




    

    




       