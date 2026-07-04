import logging
from google import genai
from google.genai import types
from app.config import settings
from app.ocr.models import DocumentElement, ElementType

logger = logging.getLogger("ocr_pipeline")


class AnswerGenerator:
    def __init__(self) -> None:
        api_key = settings.resolved_gemini_api_key
        if not api_key:
            raise RuntimeError(
                "Gemini API key is not configured for answer generation. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = settings.gemini_model

    def generate_answers(self, elements: list[DocumentElement]) -> str:
        # Prepare the questions text
        question_lines = []
        for element in elements:
            if element.type in {ElementType.question, ElementType.sub_question, ElementType.mcq_option, ElementType.instruction}:
                prefix = ""
                if element.type == ElementType.sub_question:
                    prefix = "  "
                elif element.type == ElementType.mcq_option:
                    prefix = "    "
                
                line = f"{prefix}{element.raw_text}"
                if element.mark_allocation:
                    line += f" {element.mark_allocation}"
                question_lines.append(line)

        if not question_lines:
            return "Note: No questions were detected in the document to solve."

        questions_content = "\n".join(question_lines)
        logger.info(f"Generating answers for {len(question_lines)} question lines using model {self._model}...")

        system_prompt = (
            "You are an expert exam solver. You are given a list of questions extracted from an exam paper.\n"
            "Provide factual, concise answers for each question.\n\n"
            "Follow these strict guidelines:\n"
            "1. Solve every question and sub-question in the order they appear.\n"
            "2. Maintain the same question labels and structure (e.g. Q1, (a), (b), 1.).\n"
            "3. For MCQ questions, state the correct option letter/label clearly (e.g. 'Answer: (b)') and optionally a brief, concise explanation.\n"
            "4. Format the output in Markdown.\n"
            "5. Never output answers for section headings or instruction markers themselves. Just solve the questions.\n"
            "6. Make sure the output is concise, factual, and accurate."
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_text(text=f"Please solve the following exam questions:\n\n{questions_content}"),
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
        )

        text = response.text
        if not text:
            raise RuntimeError("Gemini returned empty output for answer generation.")

        disclaimer = "Note: Answers below are AI-generated. Please verify before use."
        cleaned_text = text.strip()
        
        # Ensure the disclaimer is prepended
        if "Answers below are AI-generated" not in cleaned_text:
            cleaned_text = f"{disclaimer}\n\n{cleaned_text}"

        return cleaned_text
