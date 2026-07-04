import unittest

from app.core.examination_builder import persist_structured_document, structured_document_from_job
from app.core.examination_model import BlockType
from app.core.html_renderer import HTMLRenderer
from app.ocr.parser import parse_markdown


class ExaminationModelTests(unittest.TestCase):
    def test_structured_document_preserves_all_pages_in_order(self) -> None:
        job = {
            "id": "job-test",
            "name": "English Exam",
            "metadata": {
                "title": "English Exam",
                "subject": "English",
                "classGrade": "10",
                "institutionName": "TITUS School",
            },
            "pages": [],
        }

        for page_number in range(1, 15):
            parsed = parse_markdown(
                page_number=page_number,
                markdown=f"Section {page_number}\nQuestion {page_number} Explain topic {page_number}. [2 Marks]",
            )
            job["pages"].append(
                {
                    "page_number": page_number,
                    "status": "completed",
                    "image_url": f"/static/page_{page_number}.png",
                    "markdown": parsed.markdown,
                    "plain_text": parsed.plain_text,
                    "elements": [element.model_dump() for element in parsed.elements],
                }
            )

        document = persist_structured_document(job)

        self.assertEqual([page.page_number for page in document.pages], list(range(1, 15)))
        self.assertEqual(len(job["structured_document"]["pages"]), 14)

        html = HTMLRenderer().render(structured_document_from_job(job))

        self.assertIn('data-page="1"', html)
        self.assertIn('data-page="14"', html)
        self.assertEqual(html.count('<article class="page"'), 14)
        self.assertLess(html.index('data-page="1"'), html.index('data-page="14"'))

    def test_renderer_consumes_structured_semantics(self) -> None:
        parsed = parse_markdown(
            page_number=1,
            markdown="\n".join(
                [
                    "SECTION A",
                    "Answer all questions.",
                    "1. Choose the correct option.",
                    "A) RAM",
                    "B) ROM",
                    "2. Fill in the blank: CPU stands for ________.",
                    "3. Match the following",
                    "CPU      Processor",
                    "RAM      Memory",
                ]
            ),
        )
        job = {
            "id": "semantic-test",
            "name": "Computer Exam",
            "metadata": {"subject": "Computer Science"},
            "pages": [
                {
                    "page_number": 1,
                    "status": "completed",
                    "elements": [element.model_dump() for element in parsed.elements],
                }
            ],
        }

        document = persist_structured_document(job)
        block_types = [block.type for page in document.pages for block in page.blocks]

        self.assertIn(BlockType.section, block_types)
        self.assertIn(BlockType.instruction, block_types)
        self.assertIn(BlockType.option, block_types)
        self.assertIn(BlockType.fill_blank, block_types)
        self.assertIn(BlockType.match_following, block_types)

        html = HTMLRenderer().render(document)
        self.assertIn('class="match-table"', html)
        self.assertIn('class="blank"', html)
        self.assertIn('class="option"', html)


if __name__ == "__main__":
    unittest.main()
