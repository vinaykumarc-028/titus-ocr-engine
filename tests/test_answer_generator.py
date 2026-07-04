import unittest
from app.ocr.models import DocumentElement, ElementType
from app.ocr.answer_generator import AnswerGenerator


class AnswerGeneratorTests(unittest.TestCase):
    def test_question_filtering_empty(self) -> None:
        # Bypass __init__ to prevent requiring API key configuration during offline unit tests
        generator = AnswerGenerator.__new__(AnswerGenerator)
        result = generator.generate_answers([])
        self.assertEqual(result, "Note: No questions were detected in the document to solve.")


if __name__ == "__main__":
    unittest.main()
