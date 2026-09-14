import logging
import asyncio

from time import perf_counter

from app.services.src.state import SupervisorOutput,ProcessedEmail
from app.services.src.graph import build_graph

builder = build_graph()

logger = logging.getLogger(__name__)



class EmailWorkflow:

    MAX_CONCURRENT_EMAILS = 10

    async def gather_emails(self,incoming_emails,incoming_user_defined_categories, model_type):
        processed_input_emails = [
            ProcessedEmail(**email.model_dump()) for email in incoming_emails
        ]
        email_count = len(incoming_emails)

        logger.info("Starting analysis for %d email(s)", len(processed_input_emails))

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_EMAILS)

        async def run_one(email: ProcessedEmail):
            async with semaphore:
                return await builder.ainvoke(
                    {
                        "one_email": email,
                        "email_count": email_count,
                        "user_defined_email_categories": incoming_user_defined_categories,
                        "model_detail": model_type,
                        "supervisor_output": SupervisorOutput(approved=False, feedback=None),
                        "retry_counter": 0,
                        "version_list": [1],
                    }
                )
 
        start_time = perf_counter()
        results = await asyncio.gather(
            *[run_one(email) for email in processed_input_emails], return_exceptions=True)
        end_time = perf_counter()

        processed_results = []

        succeeded = 0
        failed = 0


        for source_email, result in zip(processed_input_emails, results):
                if isinstance(result, Exception):
                    logger.error( "Email %s failed during analysis: %s", source_email.id, result, exc_info=result)

                    failed+=1

                    continue

                email = result["one_email"]
                            

                # Log per-email analysis
                logger.info(
                    "Email %s completed | "
                    "category=%s | confidence=%.2f | priority=%.1f | "
                    "retries=%d",
                    email.id,
                    email.category,
                    email.confidence_score or 0,
                    email.priority_score or 0,
                    result["retry_counter"],
                )

                succeeded+=1

                processed_results.append({
                    "id": email.id,
                    "threadId": email.threadId,
                    "summary": email.summary,
                    "category": email.category,
                    "priority_score": email.priority_score,
                    "confidence_score": email.confidence_score,
                    "versions": result["version_list"],
                    "retry_count": result["retry_counter"],
                    })
                
           
                
            

        logger.info("Finished analysis: %d succeeded, %d failed", succeeded, failed)

        logger.info(
            "Total analysis time: %.2f seconds",
            end_time - start_time,
        )

        # ── Surface errors to the caller ──────────────────────────────────
        # Collect all exceptions from the gather results for inspection
        first_exc = next(
            (r for r in results if isinstance(r, Exception)), None
        )

        if succeeded == 0 and first_exc is not None:
            # Every email failed — almost certainly a model/API key problem.
            # Re-raise so the WebSocket handler can classify it and send a
            # proper error frame to the frontend.
            logger.error(
                "All %d email(s) failed; propagating exception to caller: %s",
                failed,
                first_exc,
            )
            raise first_exc

        return processed_results