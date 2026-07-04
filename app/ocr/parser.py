import re
from dataclasses import dataclass
from typing import Protocol

from app.ocr.models import DocumentElement, ElementType, PageResult


@dataclass(frozen=True)
class LineContext:
    line_number: int
    raw_text: str
    text: str
    markdown_heading_level: int | None
    mark_allocation: str | None
    has_fill_blank: bool
    current_question_marker: str | None
    current_question_text: str | None


@dataclass(frozen=True)
class ParsedLine:
    element_type: ElementType
    marker: str | None = None
    parent_marker: str | None = None
    hierarchy_level: int = 0
    question_type: str | None = None


class ParseRule(Protocol):
    def match(self, context: LineContext) -> ParsedLine | None:
        raise NotImplementedError


MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
SECTION_HEADING_RE = re.compile(
    r"^(?:section|part|unit|chapter)\s+[A-Z0-9IVXLC]+(?:\b|[-:]).*"
    r"|^(?:short|very\s+short|long)\s+answers?.*"
    r"|^essay\s+questions?.*"
    r"|^compulsory\s+questions?.*",
    re.IGNORECASE,
)
SUBSECTION_HEADING_RE = re.compile(
    r"^(?:subsection|sub-section|group)\s+[A-Z0-9IVXLC]+(?:\b|[-:]).*",
    re.IGNORECASE,
)
INSTRUCTION_RE = re.compile(
    r"^(?:instructions?|general instructions?|note|notes|directions?)\b[:\-]?"
    r"|^(?:answer|attempt|choose|tick|write|fill|match|read|solve)\b.+"
    r"|^each\s+question\s+carries\b.+"
    r"|^all\s+questions\s+are\s+compulsory\b.*",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(
    r"^(?P<marker>(?:Q(?:uestion)?\.?\s*)?(?:\(?\d+\)?|\d+|[IVXLC]+)[.)]?)\s+(?P<body>.+)",
    re.IGNORECASE,
)
SUB_QUESTION_RE = re.compile(
    r"^(?P<marker>\([a-zivxlcdm]+\)|[a-zivxlcdm]+[.)])\s+(?P<body>.+)",
    re.IGNORECASE,
)
ALPHABETIC_MCQ_RE = re.compile(
    r"^(?P<marker>\([a-d]\)|[A-D][.)])\s+(?P<body>.+)",
    re.IGNORECASE,
)
NUMERIC_MCQ_RE = re.compile(
    r"^(?P<marker>\([1-4]\))\s+(?P<body>.+)",
)
HEADER_RE = re.compile(
    r"^\s*(?:subject|class|grade|time(?:\s+allowed)?|institution|school|exam|paper|date)\b\s*[:\-]",
    re.IGNORECASE,
)
DOCUMENT_TITLE_RE = re.compile(
    r"^(?:.+\s+)?(?:exam(?:ination)?|question\s+paper|test|assessment)\s*(?:paper)?$",
    re.IGNORECASE,
)
QUESTION_GROUP_RE = re.compile(
    r"^(?:questions?\s+\d+\s*(?:to|-)\s*\d+).*$",
    re.IGNORECASE,
)
TRUE_FALSE_RE = re.compile(r"\b(?:true\s*/\s*false|true\s+or\s+false)\b", re.IGNORECASE)
ASSERTION_REASON_RE = re.compile(r"\bassertion\b.*\breason\b|\breason\b.*\bassertion\b", re.IGNORECASE)
CASE_STUDY_RE = re.compile(r"\b(?:case\s+study|read\s+the\s+passage|source\s+based)\b", re.IGNORECASE)
DIAGRAM_RE = re.compile(r"\b(?:draw|diagram|label(?:led)?\s+diagram|figure)\b", re.IGNORECASE)
SIGNATURE_RE = re.compile(r"^(?:signature|teacher'?s?\s+signature|examiner'?s?\s+signature)\b", re.IGNORECASE)
PAGE_BREAK_RE = re.compile(r"^(?:---|\*\*\*|___|page\s+break)$", re.IGNORECASE)
TABLE_ROW_RE = re.compile(r"^(.+?)\s{2,}(.+)$")
MARK_ALLOCATION_RE = re.compile(
    r"(?P<mark>"
    r"\[\s*\d+(?:\.\d+)?\s*(?:m|mark|marks)\s*\]"
    r"|\(\s*\d+(?:\.\d+)?\s*(?:m|mark|marks)?\s*\)"
    r"|\b\d+(?:\.\d+)?\s*(?:m|mark|marks)\b"
    r")",
    re.IGNORECASE,
)
ONLY_MARK_ALLOCATION_RE = re.compile(
    r"^\s*(?:(?:total\s+)?marks?\s*[:\-]?\s*)?"
    r"(?P<mark>\d+(?:\.\d+)?\s*(?:m|mark|marks)?)\s*$",
    re.IGNORECASE,
)
FILL_BLANK_RE = re.compile(r"(?:_{3,}|-{3,}|\.{3,})")
TABLE_DIVIDER_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


class HeaderRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        if HEADER_RE.match(context.text) and context.line_number <= 15:
            lower = context.text.lower()
            if lower.startswith("institution") or lower.startswith("school"):
                return ParsedLine(ElementType.institution_name)
            if lower.startswith("exam"):
                return ParsedLine(ElementType.exam_name)
            return ParsedLine(ElementType.header)
        return None


class DocumentTitleRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        if context.line_number <= 8 and DOCUMENT_TITLE_RE.match(context.text):
            return ParsedLine(ElementType.document_title)
        return None


class SectionHeadingRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        if context.markdown_heading_level is not None:
            if context.markdown_heading_level >= 3:
                return ParsedLine(ElementType.subsection_heading, hierarchy_level=1)
            return ParsedLine(ElementType.section_heading)
        if SUBSECTION_HEADING_RE.match(context.text):
            return ParsedLine(ElementType.subsection_heading, hierarchy_level=1)
        if SECTION_HEADING_RE.match(context.text):
            return ParsedLine(ElementType.section_heading)
        return None


class MatchRowRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        parts = [p.strip() for p in context.raw_text.split("|")]
        if len(parts) >= 3:
            valid_parts = [p for p in parts if p]
            if len(valid_parts) == 2:
                return ParsedLine(
                    ElementType.match_row,
                    parent_marker=context.current_question_marker
                )
        if "match" in (context.current_question_text or "").lower():
            match = TABLE_ROW_RE.match(context.text)
            if match and len(match.group(1).split()) <= 8:
                return ParsedLine(ElementType.match_row, parent_marker=context.current_question_marker)
        return None


class InstructionRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        if INSTRUCTION_RE.match(context.text):
            return ParsedLine(ElementType.instruction)
        return None


class MCQOptionRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        match_alpha = ALPHABETIC_MCQ_RE.match(context.text)
        if match_alpha:
            if context.current_question_marker is None or _current_question_expects_options(context.current_question_text):
                return ParsedLine(
                    ElementType.mcq_option,
                    marker=match_alpha.group("marker"),
                    parent_marker=context.current_question_marker,
                    hierarchy_level=2,
                )

        match_num = NUMERIC_MCQ_RE.match(context.text)
        if match_num:
            if context.current_question_marker is not None and _current_question_expects_options(context.current_question_text):
                return ParsedLine(
                    ElementType.mcq_option,
                    marker=match_num.group("marker"),
                    parent_marker=context.current_question_marker,
                    hierarchy_level=2,
                )

        return None


class SubQuestionRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        match = SUB_QUESTION_RE.match(context.text)
        if match:
            return ParsedLine(
                ElementType.sub_question,
                marker=match.group("marker"),
                parent_marker=context.current_question_marker,
                hierarchy_level=1,
                question_type=_detect_question_type(context.text),
            )
        return None


class QuestionRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        if QUESTION_GROUP_RE.match(context.text):
            return ParsedLine(ElementType.question_group)
        match = QUESTION_RE.match(context.text)
        if match:
            return ParsedLine(
                ElementType.question,
                marker=match.group("marker"),
                question_type=_detect_question_type(context.text),
            )
        return None


class FillBlankRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        if context.has_fill_blank:
            return ParsedLine(
                ElementType.fill_blank,
                parent_marker=context.current_question_marker,
            )
        return None


class SpecializedQuestionRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        if PAGE_BREAK_RE.match(context.text):
            return ParsedLine(ElementType.page_break)
        if SIGNATURE_RE.match(context.text):
            return ParsedLine(ElementType.signature)
        if TRUE_FALSE_RE.search(context.text):
            return ParsedLine(ElementType.true_false, parent_marker=context.current_question_marker)
        if ASSERTION_REASON_RE.search(context.text):
            return ParsedLine(ElementType.assertion_reason, parent_marker=context.current_question_marker)
        if CASE_STUDY_RE.search(context.text):
            return ParsedLine(ElementType.case_study, parent_marker=context.current_question_marker)
        if DIAGRAM_RE.search(context.text):
            return ParsedLine(ElementType.diagram_placeholder, parent_marker=context.current_question_marker)
        return None


class MarkAllocationRule:
    def match(self, context: LineContext) -> ParsedLine | None:
        if ONLY_MARK_ALLOCATION_RE.match(context.text):
            return ParsedLine(
                ElementType.mark_allocation,
                parent_marker=context.current_question_marker,
            )
        return None


class ParagraphRule:
    def match(self, context: LineContext) -> ParsedLine:
        return ParsedLine(
            ElementType.paragraph,
            parent_marker=context.current_question_marker,
        )


PARSER_RULES: list[ParseRule] = [
    DocumentTitleRule(),
    HeaderRule(),
    SectionHeadingRule(),
    MatchRowRule(),
    MCQOptionRule(),
    SubQuestionRule(),
    QuestionRule(),
    FillBlankRule(),
    SpecializedQuestionRule(),
    InstructionRule(),
    MarkAllocationRule(),
    ParagraphRule(),
]


def parse_markdown(page_number: int, markdown: str) -> PageResult:
    reconstructed_elements: list[DocumentElement] = []
    current_question_marker: str | None = None
    current_question_text: str | None = None
    prev_line_num = -100

    for line_number, raw_line in enumerate(markdown.splitlines(), start=1):
        if _is_ignorable_line(raw_line):
            continue

        markdown_heading_level, stripped_line = _extract_heading(raw_line)
        text = _strip_markdown(stripped_line)
        if not text:
            continue

        mark_allocation = extract_mark_allocation(text)
        if mark_allocation:
            text = MARK_ALLOCATION_RE.sub("", text).strip()
            
        has_fill_blank = bool(FILL_BLANK_RE.search(text))
        context = LineContext(
            line_number=line_number,
            raw_text=raw_line.rstrip(),
            text=text,
            markdown_heading_level=markdown_heading_level,
            mark_allocation=mark_allocation,
            has_fill_blank=has_fill_blank,
            current_question_marker=current_question_marker,
            current_question_text=current_question_text,
        )

        parsed_line = _classify(context)
        if parsed_line.element_type == ElementType.question:
            current_question_marker = parsed_line.marker
            current_question_text = text
        elif parsed_line.element_type == ElementType.question_group:
            current_question_text = text

        match_column_a = None
        match_column_b = None
        if parsed_line.element_type == ElementType.match_row:
            if "|" in raw_line:
                parts = [p.strip() for p in raw_line.split("|")]
                valid_parts = [p for p in parts if p]
                if len(valid_parts) >= 2:
                    match_column_a = _strip_markdown(valid_parts[0])
                    match_column_b = _strip_markdown(valid_parts[1])
            else:
                match = TABLE_ROW_RE.match(text)
                if match:
                    match_column_a = _strip_markdown(match.group(1))
                    match_column_b = _strip_markdown(match.group(2))

        # Attempt to merge consecutive lines into logical paragraph blocks
        merged = False
        if reconstructed_elements:
            prev_el = reconstructed_elements[-1]
            
            is_consecutive = (line_number == prev_line_num + 1)
            
            can_prev_merge = prev_el.type in {
                ElementType.paragraph,
                ElementType.instruction,
                ElementType.question,
                ElementType.sub_question
            }
            
            can_curr_merge = parsed_line.element_type in {
                ElementType.paragraph
            }
            
            if is_consecutive and can_prev_merge and can_curr_merge:
                prev_el.text = f"{prev_el.text} {text}"
                prev_el.raw_text = f"{prev_el.raw_text}\n{raw_line.rstrip()}"
                if mark_allocation:
                    prev_el.mark_allocation = f"{prev_el.mark_allocation} {mark_allocation}" if prev_el.mark_allocation else mark_allocation
                prev_el.has_fill_blank = prev_el.has_fill_blank or has_fill_blank
                merged = True

        if not merged:
            element_index = len(reconstructed_elements) + 1
            reconstructed_elements.append(
                DocumentElement(
                    id=f"p{page_number}-l{line_number}-{element_index}",
                    type=parsed_line.element_type,
                    text=text,
                    line_number=line_number,
                    raw_text=raw_line.rstrip(),
                    marker=parsed_line.marker,
                    mark_allocation=mark_allocation,
                    parent_marker=parsed_line.parent_marker,
                    hierarchy_level=parsed_line.hierarchy_level,
                    question_type=parsed_line.question_type,
                    has_fill_blank=has_fill_blank,
                    match_column_a=match_column_a,
                    match_column_b=match_column_b,
                )
            )

        prev_line_num = line_number

    reconstructed_md = _elements_to_markdown(reconstructed_elements)
    return PageResult(
        page=page_number,
        markdown=reconstructed_md,
        plain_text=_markdown_to_plain_text(reconstructed_md),
        elements=reconstructed_elements,
    )


def _elements_to_markdown(elements: list[DocumentElement]) -> str:
    md_blocks = []
    for el in elements:
        block_text = el.text
        if el.mark_allocation:
            block_text = f"{block_text} {el.mark_allocation}"
            
        if el.type in {ElementType.section_heading, ElementType.document_title}:
            md_blocks.append(f"# {block_text}")
        elif el.type == ElementType.subsection_heading:
            md_blocks.append(f"## {block_text}")
        elif el.type in {ElementType.question, ElementType.sub_question, ElementType.mcq_option}:
            marker_str = el.marker if el.marker else ""
            md_blocks.append(f"{marker_str} {block_text}")
        elif el.type == ElementType.match_row:
            md_blocks.append(f"| {el.match_column_a} | {el.match_column_b} |")
        elif el.type == ElementType.header:
            md_blocks.append(block_text)
        elif el.type == ElementType.instruction:
            md_blocks.append(block_text)
        else:
            md_blocks.append(block_text)
            
    return "\n\n".join(md_blocks)


def classify_line(line: str) -> ElementType:
    heading_level, stripped_line = _extract_heading(line)
    text = _strip_markdown(stripped_line)
    context = LineContext(
        line_number=1,
        raw_text=line,
        text=text,
        markdown_heading_level=heading_level,
        mark_allocation=extract_mark_allocation(text),
        has_fill_blank=bool(FILL_BLANK_RE.search(text)),
        current_question_marker=None,
        current_question_text=None,
    )
    return _classify(context).element_type


def extract_mark_allocation(text: str) -> str | None:
    match = MARK_ALLOCATION_RE.search(text)
    if not match:
        return None
    return " ".join(match.group("mark").split())


def _classify(context: LineContext) -> ParsedLine:
    for rule in PARSER_RULES:
        parsed_line = rule.match(context)
        if parsed_line is not None:
            return parsed_line
    return ParsedLine(ElementType.paragraph)


def _extract_heading(line: str) -> tuple[int | None, str]:
    match = MARKDOWN_HEADING_RE.match(line.strip())
    if not match:
        return None, line.strip()
    return len(match.group(1)), match.group(2).strip()


def _markdown_to_plain_text(markdown: str) -> str:
    lines = []
    for raw_line in markdown.splitlines():
        if _is_ignorable_line(raw_line):
            continue
        _, stripped_line = _extract_heading(raw_line)
        text = _strip_markdown(stripped_line)
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def _strip_markdown(text: str) -> str:
    text = re.sub(
        r"<span\s+[^>]*class=[\"']low-confidence[\"'][^>]*>(.*?)</span>",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<span\s+[^>]*data-confidence=[\"']\d+[\"'][^>]*>(.*?)</span>",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text)
    text = re.sub(r"^\s*>\s?", "", text)
    text = text.replace("**", "")
    text = text.replace("`", "")
    return text.strip().strip("|").strip()


def _is_ignorable_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or bool(TABLE_DIVIDER_RE.match(stripped))


def _current_question_expects_options(current_question_text: str | None) -> bool:
    if not current_question_text:
        return False
    return bool(
        re.search(
            r"\b(?:choose|correct option|multiple choice|mcq|tick)\b",
            current_question_text,
            re.IGNORECASE,
        )
    )


def _detect_question_type(text: str) -> str | None:
    lower = text.lower()
    if FILL_BLANK_RE.search(text) or "fill in the blank" in lower or "fill in the blanks" in lower:
        return "fill_blank"
    if TRUE_FALSE_RE.search(text):
        return "true_false"
    if ASSERTION_REASON_RE.search(text):
        return "assertion_reason"
    if CASE_STUDY_RE.search(text):
        return "case_study"
    if "match the following" in lower:
        return "match_the_following"
    if _current_question_expects_options(text):
        return "mcq"
    if DIAGRAM_RE.search(text):
        return "diagram"
    if re.search(r"\b(?:program|algorithm|code|function)\b", lower):
        return "programming"
    if re.search(r"[=+\-*/^√∑π]", text):
        return "mathematical"
    if re.search(r"\b(?:explain|describe|discuss|elaborate)\b", lower):
        return "long_answer"
    if re.search(r"\b(?:define|state|name|list|write)\b", lower):
        return "short_answer"
    return None
