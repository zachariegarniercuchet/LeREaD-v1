import sys
import re
from pathlib import Path
from collections import defaultdict
from bs4 import BeautifulSoup
import re
from collections import Counter, defaultdict
from typing import List, Tuple

import nltk
from nltk import CFG, PCFG
from nltk.parse import ChartParser, ViterbiParser
from sklearn.preprocessing import normalize


#sys.path.append(str(Path.cwd() / 'analysis/pattern_analysis'))
#from utils import extract_parent_level_annotations, get_sublabel_strings

import re
from typing import List

def normalize_fragment(text: str) -> List[str]:
    text = text.lower().strip()

    # --------------------------------------------------
    # 1. Standardize legal units
    # --------------------------------------------------
    text = re.sub(r'\bsections?\b|\bss\.', 'SECTION', text)
    text = re.sub(r's\.', 'SECTION', text)

    text = re.sub(r'\bsubsections?\b', 'SUBSECTION', text)
    text = re.sub(r'\bparagraphs?\b|\bpara\.', 'PARA', text)
    text = re.sub(r'\bsubparagraphs?\b', 'SUBPARA', text)

    text = re.sub(r'\barticles?\b|\barts?\.', 'ARTICLE', text)
    text = re.sub(r'\brules?\b', 'RULE', text)

    text = re.sub(r'\bparts?\b', 'PART', text)
    text = re.sub(r'\bschedules?\b|\bsched\.', 'SCHEDULE', text)
    text = re.sub(r'\bschs\.', 'SCHEDULE', text)

    text = re.sub(r'\bpp\.', 'PP', text)
    text = re.sub(r'\bp\.', 'P', text)
    text = re.sub(r'\bparas\.', 'PARAS', text)
    text = re.sub(r'\bpara\.', 'PARA', text)
    text = re.sub(r'\bparagraph', 'PARA', text)
    text = re.sub(r'\bfootnote', 'FOOTNOTE', text)
    text = re.sub(r'\bpages', 'PAGES', text)
    text = re.sub(r'\bpage', 'PAGE', text)

    # special legal phrases
    text = re.sub(r'\bet seq\.?', 'ETSEQ', text)
    text = re.sub(r'\bthrough\b|\bto\b', 'TO', text)
    text = re.sub(r'\band\b', 'AND', text)
    text = re.sub(r',', ' , ', text)

    # --------------------------------------------------
    # 2. Roman numerals (keep BEFORE NUM)
    # --------------------------------------------------
    text = re.sub(r'\b[ivx]+\b', 'ROMAN', text)

    # --------------------------------------------------
    # 3. Enumerations like (a), (ii)
    # --------------------------------------------------
    text = re.sub(r'\(([a-z]+)\)', '(ALPHA)', text)

    # --------------------------------------------------
    # 4. Hierarchical numeric patterns
    # --------------------------------------------------
    # 638(1)(d) → NUM(PAREN)(PAREN)
    text = re.sub(r'(\d+)\((\d+)\)\(([^)]+)\)', 'NUM(PAREN)(PAREN)', text)

    # 15(1) → NUM(PAREN)
    text = re.sub(r'(\d+)\((\d+)\)', 'NUM(PAREN)', text)

    # 27.09(e) → NUM.NUM(PAREN)
    text = re.sub(r'(\d+)\.(\d+)\(([^)]+)\)', 'NUM.NUM(PAREN)', text)

    # 6.1 → NUM.NUM
    text = re.sub(r'(\d+)\.(\d+)', 'NUM.NUM', text)

    # --------------------------------------------------
    # 5. Ranges (AFTER structure)
    # --------------------------------------------------
    text = re.sub(r'NUM\s*-\s*NUM', 'RANGE', text)
    text = re.sub(r'NUM\s*TO\s*NUM', 'RANGE', text)

    # --------------------------------------------------
    # 6. Standalone numbers
    # --------------------------------------------------
    text = re.sub(r'\d+', 'NUM', text)

    # --------------------------------------------------
    # 7. Cleanup spaces
    # --------------------------------------------------
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    return tokens


def normalize_decision_citation(text: str) -> List[str]:
    text = text.strip()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # --- ORDINAL series FIRST: (3d), (2d), (4th), (3rd), (2nd) ---
    # Must run before YEAR so "(2d)" isn't mistaken for a year context
    text = re.sub(r'\(\s*\d+(?:st|nd|rd|d|th)\s*\)', '(ORDINAL)', text)

    # --- DOCKET / file numbers (Quebec style): 500-53-000006-984 ---
    text = re.sub(r'\b\d{3}-\d{2}-\d{6,9}-\d{3,4}\b', 'DOCKET', text)

    # --- FULL DATES: "October 18, 1993" / "18 October 1993" ---
    text = re.sub(
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{4}\b',
        'DATE', text
    )
    text = re.sub(
        r'\b(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2},\s+\d{4}\b',
        'DATE', text
    )

    # --- BRACKETED YEAR: [1985], [2001] ---
    text = re.sub(r'\[\s*(19|20)\d{2}\s*\]', 'YEAR', text)

    # --- PARENTHETICAL YEAR with comma: (1981), (1944), ---
    # e.g. "(1981), 56 C.C.C." — year is citation date, keep as YEAR
    text = re.sub(r'\(\s*(19|20)\d{2}\s*\)\s*,', 'YEAR ,', text)

    # --- NEUTRAL CITATION TRIBUNALS (before bare year replacement) ---
    # Collapse ALL court codes in neutral citations to TRIBUNAL
    # Covers: SCC, FCA, FC, ONCA, ONSC, BCCA, BCSC, ABCA, QCCA, QCCS,
    #         NBCA, NBQB, CanLII, SKCA, MBCA, NSCA, PECA, etc.
    text = re.sub(
        r'\b(CanLII|[A-Z]{2,4}(?:CA|CS|SC|QB|KB|HCJ|PC)?)\s+(\d+)\b',
        r'TRIBUNAL NUM', text
    )

    # --- BARE YEAR (after bracketed + neutral citation passes) ---
    text = re.sub(r'\b(19|20)\d{2}\b', 'YEAR', text)

    # --- "No." followed by a number ---
    text = re.sub(r'\bno?\.\s*\d+', 'NO NUM', text, flags=re.IGNORECASE)

    # --- COURT / JURISDICTION in parens ---
    # e.g. (C.A.), (S.C.C.), (Ont. Gen. Div.), (Que. C.A.), (Canada PC)
    text = re.sub(
        r'\(\s*(?:'
        r'[A-Z][a-zA-Z.]*\.(?:\s*[A-Z][a-zA-Z.]*\.?)*'   # dotted abbrev
        r'|(?:Gen\.|H\.C\.J\.|Cir\.|Div\.)[\w\s.]*'       # Gen. Div. etc.
        r'|Canada\s+PC'                                     # Canada PC
        r'|[A-Z]{2,5}'                                      # bare caps: SCC
        r')\s*\)',
        '(COURT)', text
    )

    # --- REPORTER abbreviations ---
    # Must cover: O.R., S.C.R., C.C.C., C.R., N.R., D.L.R., W.W.R.,
    #   C.R.R., B.C.L.R., C.P.R., Q.A.C., A.R., C.L.L.C., C.H.R.R.,
    #   C.C.E.L., Alta. L.R., Que. K.B., Imm. L.R., App. Cas.,
    #   U.S., R.R.A., R.J.Q., J.Q., C.S.C.R., S. Ct., etc.
    reporter_pattern = (
        r'\b(?:'
        r'[A-Z](?:\.[A-Z]){1,5}\.?'                         # C.C.C. / S.C.R. / R.J.Q.
        r'|[A-Z][a-z]+\.(?:\s*[A-Z](?:\.[A-Z]+)*\.?)'       # Alta. L.R. / Que. K.B.
        r'|App\.\s*Cas\.'                                     # App. Cas.
        r'|[A-Z]+\s+[A-Z][a-z]+\.'                           # S. Ct.
        r')\b'
    )
    text = re.sub(reporter_pattern, 'REPORTER', text)

    # --- PLACE abbreviations leftover ---
    text = re.sub(r'\b(Hull|Ont|Que|Alta|Man|Sask|Montréal|Montreal)\b\.?', 'PLACE', text)

    # --- Standalone numbers ---
    text = re.sub(r'\b\d+\b', 'NUM', text)

    # --- Collapse whitespace ---
    text = re.sub(r'[ \t]+', ' ', text).strip()

    return text.split()



def normalize_legislation_citation(text: str) -> List[str]:
    text = text.strip()

    # --- REGNALE YEARS ---
    text = re.sub(
        r'\b\d{1,2}-\d{1,2}\s+(?:Elizabeth|George|Victoria|Edward|William|Anne|James|Mary|Henry)'
        r'\s+(?:I{1,3}|IV|V|VI|VII|VIII|IX|X|XI)',
        'REGNALE_YEAR',
        text,
    )
    text = re.sub(r'\b(19|20)\d{2}-\d{2}\b', 'YEAR_RANGE', text)

    # --- GAZETTE / DECISION IDENTIFIERS ---
    text = re.sub(r'\bD-\d{4}-\d{3,6}\b', 'DECISION_ID', text)
    text = re.sub(r'\bG\.O\.[A-Z]\.\b', 'GAZETTE', text)
    text = re.sub(r'\b\d{4}\.[IVXLCDM]+\.\d+[A-Z]?\b', 'GAZETTE_REF', text)

    # --- SOR / REGULATION IDENTIFIERS ---
    text = re.sub(r'\bSOR/\d{2,4}-\d{1,4}\b', 'SOR_REG', text)



    text = re.sub(
        r'\bc\.\s*(?:[A-Z][\-\u2011\u2012\u2013]?\d+(?:\.\d+)?|\d+)\b',
        'CHAPTER',
        text,
    )
    text = re.sub(
        r'\bchapter\s+(?:[A-Z][\-\u2011\u2012\u2013]?\d+(?:\.\d+)?|\d+)\b',
        'CHAPTER',
        text,
        flags=re.IGNORECASE,
    )

    # --- PARENTHESISED YEAR: (1985) → YEAR, absorbing the parens ---
    text = re.sub(r'\(\s*(19|20)\d{2}\s*\)', 'YEAR', text)

    # --- DATES ---
    text = re.sub(
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{4}\b',
        'DATE',
        text,
    )
    text = re.sub(r'\b(19|20)\d{2}\b', 'YEAR', text)

    # --- SESSION / LEGISLATURE ---
    text = re.sub(r'\b\d+(?:st|nd|rd|th)\s+Sess\.', 'SESSION', text)
    text = re.sub(r'\b\d+(?:st|nd|rd|th)\s+Leg\.', 'LEGISLATURE', text)

    # --- BILL references ---
    text = re.sub(r'\bBill\s+[A-Z0-9-]+\b', 'BILL', text)

    # --- NO. references ---
    text = re.sub(r'\bNo\.\s*\d+\b', 'NO_NUM', text)

    # --- REPORTER abbreviations ---
    text = re.sub(
        r'\b(?:Que\.|Ont\.|B\.C\.|Alta\.|Man\.|Sask\.|P\.E\.I\.)\s*'
        r'(?:Q\.B\.|K\.B\.|C\.A\.|H\.C\.|H\.C\.J\.)\b',
        'REPORTER',
        text,
    )

    # --- Generic dotted abbreviations (catch-all, runs last among abbrevs) ---
    text = re.sub(r'\b[A-Z](?:\.[A-Z]){1,4}\.?\b', 'ABBREV', text)

    # --- JUDGE names ---
    text = re.sub(
        r"\b[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)?\s+J\.?\b",
        'JUDGE',
        text,
    )

    # --- PLACE names ---
    text = re.sub(
        r'\b(Hull|Ottawa|Toronto|Montreal|Québec|Vancouver|Canada|Ontario|Quebec)\b',
        'PLACE',
        text,
    )

    # --- Collapse punctuation ---
    text = re.sub(r'\s*,\s*', ' , ', text)
    text = re.sub(r'\s*\(\s*', ' ( ', text)
    text = re.sub(r'\s*\)\s*', ' ) ', text)

    # --- Remaining bare numbers ---
    text = re.sub(r'\b\d+\b', 'NUM', text)

    # --- Clean whitespace ---
    text = re.sub(r'[ \t]+', ' ', text).strip()

    return text.split()



def normalize_secondary_source(text: str) -> List[str]:

    # ── 0. pre-clean ──────────────────────────────────────────────────────────
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()

    # ── 1. URLs ───────────────────────────────────────────────────────────────
    text = re.sub(r'on\s*line\s*:\s*\S+', 'URL', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', 'URL', text)

    # ── 2. ordinals in parens  (2d) (3rd) ────────────────────────────────────
    text = re.sub(r'\(\s*\d+(?:st|nd|rd|d|th)\s*\)', ' ORDINAL ', text)

    # ── 3. edition  ───────────────────────────────────────────────────────────
    text = re.sub(
        r'\(?\s*\d*\s*(?:st|nd|rd|th)?\s*ed(?:ition)?\.?\s*(?:(?:19|20)\d{2})?\s*\)?',
        ' EDITION ', text, flags=re.IGNORECASE
    )
    # bare word like "Second Edition", "Revised Fourth Edition"
    text = re.sub(
        r'\b(?:Revised\s+)?(?:Second|Third|Fourth|Fifth|Sixth|Seventh|'
        r'Eighth|Ninth|Tenth|First|College)\s+Edition\b',
        ' EDITION ', text, flags=re.IGNORECASE
    )

    # ── 4. PUB_INFO  "City: Publisher, YEAR" ─────────────────────────────────
    _CITIES = (
        r'(?:London|Toronto|Ottawa|Montreal|Montréal|Vancouver|'
        r'New\s+York|Chicago|Oxford|Cambridge|Totonto|'
        r'Portland|Kingston|St\.?\s*Paul|Markham|The\s+Hague)'
    )
    text = re.sub(
        r'\(?\s*' + _CITIES + r'(?:\s*,\s*[A-Z][a-z]+\.?)?\s*:\s*[^,]{2,60}?,\s*(?:19|20)\d{2}\s*\)?',
        ' PUB_INFO ', text, flags=re.IGNORECASE
    )

    # ── 5. VOL_ISSUE  "67:3" ─────────────────────────────────────────────────
    text = re.sub(r'\b\d+\s*:\s*\d+\b', ' VOL_ISSUE ', text)

    # ── 6. years ──────────────────────────────────────────────────────────────
    text = re.sub(r'\(?\s*(19|20)\d{2}\s*\)?', ' YEAR ', text)

    # ── 7. split digit prefix glued to letters  62McGill → 62 McGill ─────────
    text = re.sub(r'(?<!\w)(\d+)([A-Za-z])', r'\1 \2', text)

    # ── 8. split digits glued to the RIGHT of a dot  L.J.527 → L.J. 527 ─────
    text = re.sub(r'([A-Za-z]\.)(\d)', r'\1 \2', text)

    # ── 9. JOURNAL_ABBREV ─────────────────────────────────────────────────────
    #   Single big alternation — order longest/most-specific first.
    #   After steps 7-8 all digit gluing is resolved, so we can match cleanly.
    text = re.sub(
        r'\b(?:'
        # --- multi-word named journals ---
        r'Wash\.?\s*&\s*Lee\s+L\.?\s*Rev\.?'
        r'|Can\.?\s+Y\.?B\.?\s+Int\'?l\s+Law'
        r'|Indigenous\s+Law\s+Journal'
        r'|Osgoode\s+Hall\s+L\.?\s*J\.?'
        # --- "School L.J." / "School L. Rev." style ---
        r'|(?:McGill|Dalhousie|Columbia|Colum\.?)\s+L\.?\s*(?:J\.|Rev\.?)'
        r'|Queen\'?s\s+L\.?\s*J\.?'
        r'|Can\.?\s+Bar\s+Rev\.?'
        r'|Can\.?\s+Bus\.?\s+L\.?\s*J\.?'
        # --- pure dotted abbreviations ---
        r'|S\.C\.L\.R\.'
        r'|U\.N\.B\.L\.J\.'
        r'|U\.T\.L\.J\.'
        r'|U\.B\.C\.?\s*L\.?\s*Rev\.?'
        r'|C\.J\.A\.L\.P\.'
        r'|C\.L\.E\.L\.J\.'
        r'|R\.?\s*du\s+B\.'
        r'|Alta\.?\s+L\.?\s*R\.?'
        r'|Is\.?\s*L\.?\s*R\.?'
        r'|Mod\.?\s+L\.?\s*Rev\.?'
        r'|Rev\.?\s+Const\.?\s+Stud\.?'
        r'|Sask\.?\s+L\.?\s*Rev\.?'
        r'|Mich\.?\s+L\.?\s*Rev\.?'
        r'|Adv\.?\s+(?:Q\.?|J\.?)'
        r'|Labour'
        # --- generic fallback: "Xxx L.J." / "Xxx L. Rev." ---
        r'|[A-Z][a-z]+\.?\s+L\.?\s+(?:J\.|Rev\.?)'
        r')\b',
        ' JOURNAL_ABBREV ', text, flags=re.IGNORECASE
    )

    # ── 10. author list (only when followed by ", ed(s).") ───────────────────
    _N = r'(?:[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)?|(?:[A-Z]\.\s*)+[A-Z][a-z]+)'
    text = re.sub(
        r'(?:' + _N + r')(?:\s*,\s*' + _N + r'|\s+and\s+' + _N + r')*'
        r'(?=\s*,\s*eds?\b)',
        ' AUTHOR_LIST ', text, flags=re.IGNORECASE
    )

    # ── 11. editor marker ─────────────────────────────────────────────────────
    text = re.sub(r'\beds?\.(?=,|\s|$)', ' ED_MARKER ', text, flags=re.IGNORECASE)

    # ── 12. book title after ED_MARKER ───────────────────────────────────────
    text = re.sub(
        r'(ED_MARKER\s*,\s*)([A-Za-zÀ-ÿ][^,]*?)(?=\s*(?:YEAR|EDITION|PUB_INFO))',
        r'\1BOOK_TITLE ', text
    )

    # ── 13. residual city / publisher names ───────────────────────────────────
    text = re.sub(
        r'\b(?:London|Toronto|Ottawa|Montreal|Montréal|Vancouver|'
        r'New\s+York|Chicago|Oxford|Cambridge|Totonto)\b',
        ' PUB_PLACE ', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'\b(?:Macmillan|Emond|Thomson\s+Reuters|LexisNexis|Irwin\s+Law|'
        r'Carswell|Butterworths?|Oxford\s+University\s+Press|OUP|'
        r'Cambridge\s+University\s+Press|Hart|Kluwer)\b',
        ' PUBLISHER ', text, flags=re.IGNORECASE
    )

    # ── 14. institutions ──────────────────────────────────────────────────────
    text = re.sub(r'\bLaw\s+Reform\s+Commission\s+of\s+\w+\b', ' INSTITUTION ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bSupreme\s+Court(?:\s+(?:Law\s+)?of\s+Canada)?\b', ' INSTITUTION ', text, flags=re.IGNORECASE)

    # ── 15. volume / issue / month ────────────────────────────────────────────
    text = re.sub(r'\bvol(?:ume)?\.?\s*\d*', ' VOLUME ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bissue\b', ' ISSUE ', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\b',
        ' MONTH ', text
    )

    # ── 16. remaining dotted abbreviations → ABBREV ───────────────────────────
    text = re.sub(r'\b[A-Z](?:\.[A-Za-z]){1,6}\.?\b', ' ABBREV ', text)

    # ── 17. bare numbers → NUM ────────────────────────────────────────────────
    text = re.sub(r'\b\d+\b', ' NUM ', text)

    # ── 18. punctuation normalise ─────────────────────────────────────────────
    text = re.sub(r'\s*,\s*', ' , ', text)
    text = re.sub(r'\s*:\s*', ' : ', text)
    text = re.sub(r'\s*\(\s*', ' ( ', text)
    text = re.sub(r'\s*\)\s*', ' ) ', text)

    # ── 19. collapse whitespace ───────────────────────────────────────────────
    text = re.sub(r'[ \t]+', ' ', text).strip()

    return text.split()


def normalize_decision_title(text: str) -> List[str]:

    # ------------------------------------------------------------------ #
    # 0. Pre-clean: collapse newlines / extra whitespace                  #
    # ------------------------------------------------------------------ #
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()

    # ------------------------------------------------------------------ #
    # 1. Fix glued "v." / "c." (no spaces around them)                   #
    #    "Poirierv.Ville" → "Poirier v. Ville"                           #
    # ------------------------------------------------------------------ #
    text = re.sub(r'([A-Za-zÀ-ÿ])(v\.|c\.)([A-Za-zÀ-ÿ])', r'\1 \2 \3', text)

    # ------------------------------------------------------------------ #
    # 2. Year: (YYYY) → YEAR                                             #
    # ------------------------------------------------------------------ #
    text = re.sub(r'\(\s*(19|20)\d{2}\s*\)', 'YEAR', text)

    # ------------------------------------------------------------------ #
    # 3. Case separator "v." / "c." → V                                  #
    # ------------------------------------------------------------------ #
    text = re.sub(r'\bv\.', 'V', text)
    text = re.sub(r'\bc\.', 'V', text)

    # ------------------------------------------------------------------ #
    # 4. Semicolon between joined cases → CASE_SEP                       #
    # ------------------------------------------------------------------ #
    text = re.sub(r'\s*;\s*', ' CASE_SEP ', text)

    # ------------------------------------------------------------------ #
    # 5. Tokenize: split on structural tokens while preserving parens     #
    #    At this point text contains: words, V, YEAR, CASE_SEP, ( )      #
    #    Split into chunks separated by V / YEAR / CASE_SEP / ( / )      #
    #    Each chunk of plain words → PARTY                               #
    # ------------------------------------------------------------------ #
    # Insert spaces around parens so they split cleanly
    text = re.sub(r'\(', ' ( ', text)
    text = re.sub(r'\)', ' ) ', text)

    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text).strip()

    # Now walk tokens: any run of plain words becomes PARTY
    raw_tokens = text.split()
    structural = {'V', 'YEAR', 'CASE_SEP', '(', ')'}
    
    result = []
    party_buf = []

    def flush_party():
        if party_buf:
            result.append('PARTY')
            party_buf.clear()

    for tok in raw_tokens:
        if tok in structural:
            flush_party()
            result.append(tok)
        else:
            # Strip trailing punctuation like commas, periods
            tok_clean = re.sub(r'^[,\.]+|[,\.]+$', '', tok)
            if tok_clean:
                party_buf.append(tok_clean)

    flush_party()

    return result


def extract_parent_level_annotations_from_string(text_content):
    """
    Extract all parent-level manual_label annotations from a raw text string.
    Parent-level = manual_label tags that are NOT nested inside another manual_label.
    
    Returns a dict with keys: 'decision', 'legislation', 'secondary sources'
    Each value is a list of annotation dicts containing:
      - full_html: the complete annotation HTML
      - text_content: extracted text (no HTML tags)
      - direct_sublabels: list of direct sublabel labelnames
      - all_sublabels: list of all nested sublabel labelnames
    """
    soup = BeautifulSoup(text_content, 'html.parser')
    
    annotations = {
        'decision': [],
        'legislation': [],
        'secondary sources': []
    }
    
    # Find manual_label tags that have no manual_label ancestor = parent-level
    all_labels = soup.find_all('manual_label')
    parent_labels = [
        label for label in all_labels
        if not label.find_parent('manual_label')
    ]
    
    for label in parent_labels:
        labelname = label.get('labelname', '')
        
        if labelname not in annotations:
            continue
        
        # Direct children sublabels only
        direct_sublabels = [
            sl.get('labelname', '')
            for sl in label.find_all('manual_label', recursive=False)
        ]
        
        # All nested sublabels
        all_sublabels = [
            sl.get('labelname', '')
            for sl in label.find_all('manual_label')
        ]
        
        annotations[labelname].append({
            'full_html':       str(label),
            'text_content':    label.get_text(strip=True),
            'direct_sublabels': direct_sublabels,
            'all_sublabels':   all_sublabels,
        })
    
    return annotations

# ── Build all pattern dicts ───────────────────────────────────────────────────
def build_pattern_dict(tokenized, pattern_counts):
    """
    Returns {key: [(score, pattern_tuple), ...]}
    sorted by score descending, reconstructed fresh on every run.
    """
    total = len(tokenized)
    seen = set()
    result = []
    for token_seq in tokenized:
        pattern = tuple(token_seq)
        if pattern not in seen:
            seen.add(pattern)
            score = round(pattern_counts[pattern] / total * 100, 2)
            result.append((score, pattern))
    return sorted(result, key=lambda x: x[0], reverse=True)


if __name__ == "__main__":
    # Define the three HTML files to analyze
    from pathlib import Path
    data_dir = Path.cwd() / 'data' / 'Documents_Annotés'
    final_annotated_dir = Path.cwd() / 'data' / 'final' / 'Annotated'

    html_files = [
        final_annotated_dir / '1997CanLII16226_ONCA_annotated_EG_revRL_tech.html',
        final_annotated_dir / '2021QCCA1675_annotated_EG_revRL_tech.html',
        final_annotated_dir /  '1989CanLII1415CITT_annotated_GL_revRL_tech.html',
        final_annotated_dir / '2001CanLII21117QCTDP_annotated_GL_revRL_tech.html',
        final_annotated_dir / "2019SCC65_annotated_EG_revRL_tech.html",
        final_annotated_dir / "2024NBKB203_annotated_VP_rev2RL_tech.html",
        final_annotated_dir / "2005QCCA437_LLMv1.3_Verified_GL_revRL_tech.html",
    ]

    # Verify files exist
    for f in html_files:
        print(f"{'✓' if f.exists() else '✗'} {f.name}")

    # Process all files
    all_annotations = {
        'decision': [],
        'legislation': [],
        'secondary sources': []
    }

    for html_file in html_files:
        #print(f"\nProcessing: {html_file.name}")
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        annotations = extract_parent_level_annotations(html_content)
        
        # Aggregate results
        for label_type in ['decision', 'legislation', 'secondary sources']:
            count = len(annotations[label_type])
            #print(f"  - {label_type}: {count} annotations")
            all_annotations[label_type].extend(annotations[label_type])

    print("\n" + "="*60)
    print("TOTAL ANNOTATIONS ACROSS ALL FILES:")
    for label_type in ['decision', 'legislation', 'secondary sources']:
        print(f"  {label_type}: {len(all_annotations[label_type])}")
    print("="*60)
    



    decision_fragments = get_sublabel_strings(all_annotations, 'decision', 'fragment', max_items=None)
    sec_sources_fragments = get_sublabel_strings(all_annotations, 'secondary sources', 'fragment', max_items=None)
    legislation_fragments = get_sublabel_strings(all_annotations, 'legislation', 'fragment', max_items=None)

    decision_citation = get_sublabel_strings(all_annotations, 'decision', 'citation', max_items=None)
    legislation_citation = get_sublabel_strings(all_annotations, 'legislation', 'citation', max_items=None)
    sec_sources_source = get_sublabel_strings(all_annotations, 'secondary sources', 'source', max_items=None)
    decision_title = get_sublabel_strings(all_annotations, 'decision', 'title', max_items=None)


    tokenized_decision = [normalize_fragment(x) for x in decision_fragments]
    tokenized_sec_sources = [normalize_fragment(x) for x in sec_sources_fragments]
    tokenized_legislation = [normalize_fragment(x) for x in legislation_fragments]
    tokenized_decision_citation = [normalize_decision_citation(x) for x in decision_citation]
    tokenized_legislation_citation = [normalize_legislation_citation(x) for x in legislation_citation]
    tokenized_sec_sources_source = [normalize_secondary_source(x) for x in sec_sources_source]
    tokenized_decision_title = [normalize_decision_title(x) for x in decision_title]


    # ── Pattern counts ────────────────────────────────────────────────────────────
    pattern_counts_decision            = Counter(tuple(seq) for seq in tokenized_decision)
    pattern_counts_legislation         = Counter(tuple(seq) for seq in tokenized_legislation)
    pattern_counts_sec_sources         = Counter(tuple(seq) for seq in tokenized_sec_sources)
    pattern_counts_decision_citation   = Counter(tuple(seq) for seq in tokenized_decision_citation)
    pattern_counts_legislation_citation= Counter(tuple(seq) for seq in tokenized_legislation_citation)
    pattern_counts_sec_sources_source  = Counter(tuple(seq) for seq in tokenized_sec_sources_source)
    pattern_counts_decision_title      = Counter(tuple(seq) for seq in tokenized_decision_title)

    

    pattern_dicts = {
        'decision_fragment':    build_pattern_dict(decision_fragments,   tokenized_decision,             Counter(tuple(s) for s in tokenized_decision)),
        'legislation_fragment': build_pattern_dict(legislation_fragments, tokenized_legislation,         Counter(tuple(s) for s in tokenized_legislation)),
        'sec_sources_fragment': build_pattern_dict(sec_sources_fragments, tokenized_sec_sources,         Counter(tuple(s) for s in tokenized_sec_sources)),
        'decision_citation':    build_pattern_dict(decision_citation,    tokenized_decision_citation,    Counter(tuple(s) for s in tokenized_decision_citation)),
        'legislation_citation': build_pattern_dict(legislation_citation, tokenized_legislation_citation, Counter(tuple(s) for s in tokenized_legislation_citation)),
        'sec_sources_source':   build_pattern_dict(sec_sources_source,   tokenized_sec_sources_source,   Counter(tuple(s) for s in tokenized_sec_sources_source)),
        'decision_title':       build_pattern_dict(decision_title,       tokenized_decision_title,       Counter(tuple(s) for s in tokenized_decision_title)),
    }

    # Sanity check
    for key, entries in pattern_dicts.items():
        print(f"\n{key} ({len(entries)} patterns):")
        for score, pattern in entries[:3]:
            print(f"  [{score:.2f}%] {pattern}")