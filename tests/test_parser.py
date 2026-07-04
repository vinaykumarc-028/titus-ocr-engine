from pathlib import Path
import unittest

from app.ocr.models import ElementType
from app.ocr.parser import classify_line, extract_mark_allocation, parse_markdown


SAMPLES_DIR = Path(__file__).parent / "samples"


class ParserTests(unittest.TestCase):
    def test_parse_sample_document_structures(self) -> None:
        markdown = (SAMPLES_DIR / "sample_ocr.md").read_text(encoding="utf-8")

        page = parse_markdown(page_number=1, markdown=markdown)

        self.assertEqual(page.page, 1)
        self.assertIn("Section A", page.plain_text)
        self.assertEqual(page.elements[0].type, ElementType.section_heading)
        self.assertEqual(page.elements[1].type, ElementType.instruction)

        questions = [item for item in page.elements if item.type == ElementType.question]
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].marker, "1.")
        self.assertEqual(questions[0].mark_allocation, "[2 Marks]")

        sub_questions = [
            item for item in page.elements if item.type == ElementType.sub_question
        ]
        self.assertEqual(len(sub_questions), 2)
        self.assertEqual(sub_questions[0].parent_marker, "1.")

        fill_blanks = [item for item in page.elements if item.has_fill_blank]
        self.assertEqual(len(fill_blanks), 1)
        self.assertEqual(fill_blanks[0].type, ElementType.sub_question)

        mcq_options = [item for item in page.elements if item.type == ElementType.mcq_option]
        self.assertEqual(len(mcq_options), 4)
        self.assertEqual(mcq_options[0].parent_marker, "2.")

        self.assertEqual(page.elements[-1].type, ElementType.paragraph)

    def test_line_classifier(self) -> None:
        self.assertEqual(classify_line("## Part II"), ElementType.section_heading)
        self.assertEqual(
            classify_line("Note: Show rough work clearly."),
            ElementType.instruction,
        )
        self.assertEqual(classify_line("Q.1 Name the gas."), ElementType.question)
        self.assertEqual(classify_line("(ii) Explain briefly."), ElementType.sub_question)
        self.assertEqual(classify_line("(c) 45 degrees"), ElementType.mcq_option)
        self.assertEqual(classify_line("Total marks: 2"), ElementType.mark_allocation)
        self.assertEqual(classify_line("Answer: ________"), ElementType.fill_blank)
        self.assertEqual(classify_line("Subject: Geography"), ElementType.header)
        self.assertEqual(classify_line("Class: X"), ElementType.header)
        self.assertEqual(classify_line("| Match A | Match B |"), ElementType.match_row)
        self.assertEqual(classify_line("A plain continuation line."), ElementType.paragraph)

    def test_mark_allocation_extraction(self) -> None:
        self.assertEqual(extract_mark_allocation("1. Define force. [1 Mark]"), "[1 Mark]")
        self.assertEqual(extract_mark_allocation("Explain inertia. 2M"), "2M")
        self.assertEqual(extract_mark_allocation("No marks shown."), None)

    def test_examination_intelligence_semantics(self) -> None:
        markdown = "\n".join(
            [
                "Final Examination Paper",
                "Institution: TITUS School",
                "SECTION B",
                "Answer any five questions.",
                "Question 1 Define fragmentation. (2 Marks)",
                "(i) True or False: Memory is contiguous.",
                "2. Choose the correct option.",
                "A) RAM",
                "B) ROM",
                "C) CPU",
                "D) HDD",
                "3. Fill in the blank: Process id is ________.",
                "4. Match the following",
                "Memory      Storage",
                "CPU         Processor",
                "5. Draw a labelled diagram of process states.",
            ]
        )

        page = parse_markdown(page_number=1, markdown=markdown)

        self.assertEqual(page.elements[0].type, ElementType.document_title)
        self.assertEqual(page.elements[1].type, ElementType.institution_name)
        self.assertTrue(any(el.type == ElementType.section_heading for el in page.elements))
        self.assertTrue(any(el.type == ElementType.instruction for el in page.elements))

        q1 = next(el for el in page.elements if el.marker == "Question 1")
        self.assertEqual(q1.type, ElementType.question)
        self.assertEqual(q1.mark_allocation, "(2 Marks)")
        self.assertEqual(q1.question_type, "short_answer")

        true_false = next(el for el in page.elements if "True or False" in el.text)
        self.assertEqual(true_false.question_type, "true_false")

        options = [el for el in page.elements if el.type == ElementType.mcq_option]
        self.assertEqual([el.marker for el in options], ["A)", "B)", "C)", "D)"])

        fill_blank = next(el for el in page.elements if "Process id" in el.text)
        self.assertTrue(fill_blank.has_fill_blank)

        match_rows = [el for el in page.elements if el.type == ElementType.match_row]
        self.assertEqual(len(match_rows), 2)
        self.assertEqual(match_rows[0].match_column_a, "Memory")
        self.assertEqual(match_rows[0].match_column_b, "Storage")

        diagram = next(el for el in page.elements if "diagram" in el.text.lower())
        self.assertEqual(diagram.question_type, "diagram")


if __name__ == "__main__":
    unittest.main()
