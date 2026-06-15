"""Converted from Jupyter Notebook."""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging
import re

# =====================================================
# MODELS
# =====================================================

BART_MODEL = "facebook/bart-large-cnn"
T5_MODEL = "google/flan-t5-base"
BULLET = "\u2022"

# Hide noisy Transformers warnings without changing model configs or weights.
hf_logging.set_verbosity_error()

print("Loading BART...")
bart_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL)

print("Loading FLAN-T5...")
t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL)

# =====================================================
# TEXT UTILITIES
# =====================================================


def word_count(text):
    return len(text.split())


def clean_word_count(text):
    return len(get_words(text))


def get_paragraphs(text):
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    if not paragraphs:
        return [text.strip()]

    return paragraphs


def split_into_sentences(text):
    normalized = re.sub(r"\s+", " ", text.strip())

    if not normalized:
        return []

    sentences = re.findall(r"[^.!?]+[.!?]+(?=\s|$)", normalized)
    trailing = re.sub(r".*[.!?]+\s*", "", normalized)

    if trailing and trailing != normalized:
        sentences.append(trailing)

    if not sentences:
        return [normalized]

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_into_chunks(text, chunk_size=400):
    words = text.split()

    return [
        " ".join(words[index:index + chunk_size])
        for index in range(0, len(words), chunk_size)
    ]


def ensure_sentence(text):
    text = " ".join(text.strip().split())

    if not text:
        return text

    if text[-1] not in ".!?":
        text += "."

    return text


def lowercase_first(text):
    if not text:
        return text

    return text[0].lower() + text[1:]


def strip_list_marker(text):
    marker_chars = "-*" + BULLET + "0123456789. )\t"
    return text.lstrip(marker_chars).strip()


def force_one_sentence(text):
    text = strip_list_marker(" ".join(text.split()))
    sentences = split_into_sentences(text)

    if len(sentences) <= 1:
        return ensure_sentence(text)

    fused = "; ".join(
        sentence.rstrip(".!?")
        for sentence in sentences
    )

    return ensure_sentence(fused)


def strip_summary_intro(text):
    text = force_one_sentence(text).rstrip(".!?")
    intro_patterns = [
        r"^this\s+(passage|paragraph|text|article)\s+"
        r"(talks about(?: how)?|gives you an idea about|gives an idea about|"
        r"discusses(?: how)?|explains|describes|covers|is about)\s+",
        r"^the\s+(passage|paragraph|text|article)\s+"
        r"(talks about(?: how)?|gives you an idea about|gives an idea about|"
        r"discusses(?: how)?|explains|describes|covers|is about)\s+",
    ]

    for pattern in intro_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return text


def apply_summary_prefix(text, prefix):
    body = strip_summary_intro(text)

    if not body:
        return ensure_sentence(prefix)

    if prefix.lower().endswith(" how") and body.lower().startswith("how "):
        body = body[4:].strip()

    return ensure_sentence(f"{prefix} {lowercase_first(body)}")


def is_meta_summary(text):
    lowered = text.lower()
    meta_phrases = [
        "a short summary",
        "brief summary",
        "main idea of this passage",
        "main idea of the passage",
        "main idea of this text",
        "what the passage is about",
        "what this passage is about",
        "what the text is about",
        "the passage talks about",
        "the text talks about",
        "this passage is about",
        "this text is about",
    ]

    return any(phrase in lowered for phrase in meta_phrases)


def format_with_newlines(text, word_limit=20):
    words = text.split()
    formatted_lines = []
    current_line_words = []
    for i, word in enumerate(words):
        current_line_words.append(word)
        if (i + 1) % word_limit == 0:
            formatted_lines.append(" ".join(current_line_words))
            current_line_words = []
    if current_line_words:
        formatted_lines.append(" ".join(current_line_words))
    return "\n".join(formatted_lines)

# =====================================================
# COPY DETECTION
# =====================================================


def get_words(text):
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def get_ngrams(text, n=4):
    words = get_words(text)

    return {
        tuple(words[index:index + n])
        for index in range(len(words) - n + 1)
    }


def copied_ngram_ratio(source, candidate, n=4):
    candidate_ngrams = get_ngrams(candidate, n)

    if not candidate_ngrams:
        return 0

    source_ngrams = get_ngrams(source, n)
    copied_ngrams = candidate_ngrams & source_ngrams

    return len(copied_ngrams) / len(candidate_ngrams)


def choose_least_copied(source, candidates):
    usable = [
        force_one_sentence(candidate)
        for candidate in candidates
        if candidate and candidate.strip()
    ]

    if not usable:
        return ""

    return min(
        usable,
        key=lambda candidate: copied_ngram_ratio(source, candidate)
    )

# =====================================================
# FLAN-T5 GENERATION
# =====================================================


def t5_generate(
    prompt,
    max_tokens=64,
    num_return_sequences=1,
    do_sample=True,
    num_beams=1
):
    inputs = t5_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = t5_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        top_p=0.92,
        temperature=0.95,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        num_return_sequences=num_return_sequences
    )

    decoded = [
        t5_tokenizer.decode(output, skip_special_tokens=True).strip()
        for output in outputs
    ]

    if num_return_sequences == 1:
        return decoded[0]

    return decoded


def t5_paraphrase(prompt, source, max_tokens=64, copy_threshold=0.28):
    candidates = t5_generate(
        prompt,
        max_tokens=max_tokens,
        num_return_sequences=4
    )
    result = choose_least_copied(source, candidates)

    if copied_ngram_ratio(source, result) <= copy_threshold:
        return result

    retry_prompt = f"""
The previous rewrite copied too much wording from the original.

Rewrite the idea again with stronger paraphrasing.

Strict rules:
- Preserve the meaning
- Use a new sentence structure
- Use synonyms where possible
- Do not copy any phrase of four or more words
- Keep unavoidable technical terms only
- Return one sentence only

Original:
{source}
"""

    retry_candidates = t5_generate(
        retry_prompt,
        max_tokens=max_tokens,
        num_return_sequences=4
    )
    retry = choose_least_copied(source, retry_candidates)

    if retry:
        return retry

    return result


def t5_deterministic(prompt, max_tokens=64):
    return t5_generate(
        prompt,
        max_tokens=max_tokens,
        do_sample=False,
        num_beams=4
    )


def choose_styled_summary(
    source,
    candidates,
    min_words,
    max_words,
    required_prefix
):
    usable = []
    fallback = []

    for candidate in candidates:
        if not candidate or not candidate.strip():
            continue

        candidate = force_one_sentence(candidate)

        if required_prefix:
            candidate = apply_summary_prefix(candidate, required_prefix)

        fallback.append(candidate)

        if not is_meta_summary(candidate):
            usable.append(candidate)

    if not usable:
        usable = fallback

    if not usable:
        return ""

    target = (min_words + max_words) / 2

    def score(candidate):
        words = clean_word_count(candidate)
        if min_words <= words <= max_words:
            word_penalty = 0
        else:
            word_penalty = min(abs(words - min_words), abs(words - max_words))

        return (
            int(is_meta_summary(candidate)),
            word_penalty,
            abs(words - target),
            copied_ngram_ratio(source, candidate)
        )

    return min(usable, key=score)


def extract_core_idea(text):
    if word_count(text) > 500:
        content_seed = hierarchical_bart(text)
    else:
        content_seed = bart_summarize(
            text,
            min_len=30,
            max_len=90
        )

    prompt = f"""
Create a factual content plan from the passage.

Rules:
- State what the passage says, not what a summary should do
- Include the main topic and the most important supporting points
- Mention the actual subject matter directly
- Do not use phrases like "main idea", "this passage", "this text", or "summary"
- Do not include minor examples
- Use one clear sentence
- Use 30 to 45 words

Passage:
{text}

Content seed:
{content_seed}
"""

    candidates = t5_generate(
        prompt,
        max_tokens=70,
        num_return_sequences=6
    )
    non_meta_candidates = [
        force_one_sentence(candidate)
        for candidate in candidates
        if candidate and not is_meta_summary(candidate)
    ]

    if non_meta_candidates:
        return choose_least_copied(text, non_meta_candidates)

    if content_seed and not is_meta_summary(content_seed):
        return force_one_sentence(content_seed)

    return force_one_sentence(text)


def style_core_idea(
    original_text,
    core_idea,
    style_name,
    style_rules,
    min_words,
    max_words,
    max_tokens,
    required_prefix
):
    prompt = f"""
Rewrite the core idea as a {style_name} summary.

The casual and formal summaries must describe the same core content.
The summary must begin exactly with: {required_prefix}

Style rules:
{style_rules}

Length rules:
- Use {min_words} to {max_words} words total, including the required opening
- Return one sentence only

Anti-copy rules:
- Use wording that is very different from the original passage
- Do not copy any phrase of four or more words
- Keep only necessary technical terms

Core idea:
{core_idea}

Original passage:
{original_text}
"""

    candidates = t5_generate(
        prompt,
        max_tokens=max_tokens,
        num_return_sequences=6
    )
    result = choose_styled_summary(
        original_text,
        candidates,
        min_words,
        max_words,
        required_prefix
    )

    if min_words <= clean_word_count(result) <= max_words:
        return result

    retry_prompt = f"""
Rewrite this same core idea again as a {style_name} summary.

Core idea:
{core_idea}

Strict requirements:
- Begin exactly with: {required_prefix}
- Use {min_words} to {max_words} words total, including the required opening
- One sentence only
{style_rules}
- Use wording very different from the original passage
- Do not copy any phrase of four or more words

Original passage:
{original_text}
"""

    retry_candidates = t5_generate(
        retry_prompt,
        max_tokens=max_tokens,
        num_return_sequences=6
    )
    retry = choose_styled_summary(
        original_text,
        retry_candidates,
        min_words,
        max_words,
        required_prefix
    )

    if retry and min_words <= clean_word_count(retry) <= max_words:
        return retry

    best_result = retry or result
    current_words = clean_word_count(best_result)
    repair_prompt = f"""
Fix the length of this {style_name} summary while keeping the same meaning.

Current summary:
{best_result}

Strict requirements:
- Begin exactly with: {required_prefix}
- Use {min_words} to {max_words} words total
- One sentence only
- Keep the same core idea and key contents
{style_rules}

Core idea:
{core_idea}
"""

    if current_words < min_words:
        repair_prompt += "\n- Add useful key content without adding minor details\n"
    elif current_words > max_words:
        repair_prompt += "\n- Shorten the sentence without losing the main idea\n"

    repair_candidates = t5_generate(
        repair_prompt,
        max_tokens=max_tokens,
        num_return_sequences=6
    )
    repair = choose_styled_summary(
        original_text,
        repair_candidates,
        min_words,
        max_words,
        required_prefix
    )

    if (
        repair
        and not is_meta_summary(repair)
        and min_words <= clean_word_count(repair) <= max_words
    ):
        return repair

    return apply_summary_prefix(best_result, required_prefix)

# =====================================================
# BART SUMMARIZATION
# =====================================================


def bart_summarize(text, min_len=20, max_len=100):
    inputs = bart_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    summary_ids = bart_model.generate(
        inputs["input_ids"],
        num_beams=6,
        min_length=min_len,
        max_length=max_len,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        early_stopping=True
    )

    return bart_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    ).strip()


def hierarchical_bart(text):
    chunks = split_into_chunks(text, chunk_size=400)
    partial_summaries = []

    for chunk in chunks:
        partial_summaries.append(
            bart_summarize(
                chunk,
                min_len=20,
                max_len=80
            )
        )

    merged = " ".join(partial_summaries)

    return bart_summarize(
        merged,
        min_len=30,
        max_len=120
    )

# =====================================================
# CASUAL MODE
# Compression operator: simple explanation
# =====================================================


def casual_summary(text):
    core_idea = extract_core_idea(text)
    style_rules = """
- Sound casual and easy to understand
- Write like you are explaining it to a 12-year-old student
- Give the main idea and key contents, not tiny details
- Prefer short, common words
- Use simple synonyms for words from the passage
- Use friendly wording such as "shows", "helps", "learns", or "uses" when suitable
- Avoid technical wording unless it is necessary
"""

    result = style_core_idea(
        original_text=text,
        core_idea=core_idea,
        style_name="casual",
        style_rules=style_rules,
        min_words=20,
        max_words=25,
        max_tokens=45,
        required_prefix="This passage talks about how"
    )

    return format_with_newlines(ensure_sentence(result))

# =====================================================
# FORMAL MODE
# Compression operator: academic elevation
# =====================================================


def formal_summary(text):
    core_idea = extract_core_idea(text)
    style_rules = """
- Use formal, academic, and scientific language
- Use more precise and elevated vocabulary than the casual version
- Include the main idea and key contents
- Expand the explanation with cause, function, significance, and broader implication
- Prefer words such as "examines", "demonstrates", "facilitates", "empirical", "computational", or "systematic" when suitable
- Avoid informal wording
- Keep the same central content as the casual summary
"""

    result = style_core_idea(
        original_text=text,
        core_idea=core_idea,
        style_name="formal academic",
        style_rules=style_rules,
        min_words=45,
        max_words=55,
        max_tokens=95,
        required_prefix="This passage discusses how"
    )

    return format_with_newlines(ensure_sentence(result))

# =====================================================
# BULLET MODE
# Compression operator: sentence decomposition -> paraphrase
# =====================================================


def compress_sentence_to_bullet(sentence):
    prompt = f"""
Rewrite this sentence as one concise bullet point using heavy paraphrasing.

Strict rules:
- Keep the original meaning
- Use fresh wording
- Change the sentence structure
- Do not copy any phrase of four or more words
- Keep unavoidable technical terms only
- Remove unnecessary wording
- Return one sentence only

Sentence:
{sentence}
"""

    result = t5_paraphrase(
        prompt,
        sentence,
        max_tokens=48,
        copy_threshold=0.22
    )

    if not result:
        result = sentence

    return f"{BULLET} {force_one_sentence(result)}"


def bullet_summary(text):
    bullets = []

    for paragraph in get_paragraphs(text):
        for sentence in split_into_sentences(paragraph):
            bullets.append(compress_sentence_to_bullet(sentence))

    return "\n".join(bullets)

# =====================================================
# PARAGRAPH MODE
# Compression operator: sentence fusion -> restructuring
# =====================================================


def fuse_sentences_to_one(sentence_group):
    prompt = f"""
Fuse these sentences into one heavily paraphrased summary sentence.

Strict rules:
- One sentence only
- Restructure the ideas instead of listing them
- Preserve the main meaning
- Use fresh wording
- Change the sentence structure
- Do not copy any phrase of four or more words
- Keep unavoidable technical terms only
- Do not use bullet points
- Maximum 45 words

Sentences:
{sentence_group}
"""

    result = t5_paraphrase(
        prompt,
        sentence_group,
        max_tokens=70,
        copy_threshold=0.22
    )

    if not result:
        return ensure_sentence(sentence_group)

    return force_one_sentence(result)


def paragraph_summary(text):
    fused_paragraphs = []

    for paragraph in get_paragraphs(text):
        if word_count(paragraph) > 500:
            extractive_summary = hierarchical_bart(paragraph)
            fused = fuse_sentences_to_one(extractive_summary)
        else:
            sentences = split_into_sentences(paragraph)
            fused = fuse_sentences_to_one(" ".join(sentences))

        fused_paragraphs.append(ensure_sentence(fused))

    if len(fused_paragraphs) == 1:
        return format_with_newlines(fused_paragraphs[0])

    return format_with_newlines(" ".join(fused_paragraphs))

# =====================================================
# MAIN FUNCTION
# =====================================================


def summarize_text(text, mode="formal"):
    mode = mode.lower()

    if mode == "casual":
        return casual_summary(text)

    if mode == "formal":
        return formal_summary(text)

    if mode == "bullet":
        return bullet_summary(text)

    if mode == "paragraph":
        return paragraph_summary(text)

    raise ValueError(
        "mode must be one of: casual, formal, bullet, paragraph"
    )

