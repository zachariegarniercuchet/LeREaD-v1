SUBLABEL_DEFINITIONS_V1 = {
    "title": (
        "<title>: Official title or alias designating a legal authority, including "
        "a legislative text, a judicial decision, or a secondary source publication. "
        "For decisions: party names and the 'v.' formulation. "
        "For legislation: the name of the statute or regulation. "
        "For secondary sources: the title of the book, article, or specific contribution. "
        "Do NOT include publication venue or collection titles."
    ),

    "citation": (
        "<citation>: Bibliographic or publication information identifying a legislative "
        "text or a judicial decision, such as year, reporter, volume, or number "
        "(e.g., S.C.R., D.L.R., statute year and chapter). "
        "Applicable only to decisions and legislation."
    ),

    "source": (
        "<source>: Bibliographic or publication information identifying the source of a "
        "secondary publication, fully or partially. This may include journal name, "
        "collective work title, publisher, volume, year, or CanLIIdocs references. "
        "Applicable only to secondary sources. "
        "For collective works, include only the bibliographic information following "
        "'in' or 'dans', not the word itself."
    ),

    "authors": (
        "<authors>: Author or list of authors of a secondary source publication, "
        "including accompanying 'et al.' when present. "
        "All authors must be included within a single <authors> tag. "
        "Applicable only to secondary sources. "
        "Do NOT include scientific editors of collective works unless they are explicitly "
        "identified as authors of the cited contribution."
    ),

    "fragment": (
        "<fragment>: A precise part of a legal text, decision, or secondary source "
        "used to locate specific information, such as article, paragraph, page, "
        "section, or subsection number. Applicable to all authority types."
        "Sections 7 and 11(d) sould be in two separate <fragment> tags, not combined into one."
        "Same for ss. 15, 7, 11(d) and 11(f) -> ss.15, 7, 11(d), 11(f) should be in four separate <fragment> tags, not combined into one."
    ),
}

SUBLABEL_DEFINITIONS_V2 = {
    "title": (
        "<title>: Official name or commonly used designation of a legal authority. "
        "First determine the authority type (decision, legislation, or secondary source). "
        "Include only the name identifying the authority itself.\n\n"
        "Include:\n"
        "- Decisions: party names including the 'v.' formulation.\n"
        "- Legislation: statute or regulation name.\n"
        "- Secondary sources: title of the book, article, speech, or specific contribution cited.\n\n"
        "Exclude:\n"
        "- Journal names, publishers, collective work titles, or publication venues.\n"
        "- Citation numbers, reporters, years, or volume information."
    ),

    "citation": (
        "<citation>: Formal legal reference identifying a judicial decision or legislation. "
        "Applicable ONLY to decisions and legislation.\n\n"
        "Include:\n"
        "- Year, reporter, volume, chapter, or decision number.\n"
        "- Court acronym when it appears within the citation.\n\n"
        "Rules:\n"
        "- Each parallel citation must appear in a separate <citation> tag.\n\n"
        "Exclude:\n"
        "- Author names, journal titles, publishers, or doctrinal publication information."
    ),

    "source": (
        "<source>: Bibliographic information identifying where a secondary source "
        "publication appears. Applicable ONLY to secondary sources.\n\n"
        "Include:\n"
        "- Journal name, collective work title, publisher, volume, or year.\n"
        "- Partial bibliographic information if it helps identify the publication.\n"
        "- CanLIIDocs references.\n\n"
        "Rules:\n"
        "- For collective works, include only the bibliographic information following "
        "'in' or 'dans', not the word itself.\n"
        "- If both an original publication reference and a CanLIIDocs reference exist, "
        "use two separate <source> tags.\n\n"
        "Exclude:\n"
        "- Titles of articles, books, or contributions (these belong to <title>)."
        "If the source is under parentheses, the parentheses shoudn't be included in the <source> tag : (2nd ed.) -> (<source>2nd ed.</source>)."
    ),

    "authors": (
        "<authors>: Author or list of authors responsible for a cited secondary source. "
        "Applicable ONLY to secondary sources.\n\n"
        "Include:\n"
        "- All authors together inside a single <authors> tag.\n"
        "- 'et al.' when present.\n"
        "- Author of speeches or allocutions, including judges.\n\n"
        "Exclude:\n"
        "- Scientific editors of collective works unless explicitly identified as "
        "authors of the cited contribution.\n"
        "- Splitting authors across multiple <authors> tags."
    ),

    "fragment": (
        "<fragment>: Precise locator referring to a specific part of a legal authority. "
        "Applicable to decisions, legislation, and secondary sources.\n\n"
        "Include:\n"
        "- Articles, sections, subsections, paragraphs, pages, or similar references.\n"
        "- Each referenced element separately.\n\n"
        "Rules:\n"
        "- Multiple references must be split into distinct <fragment> tags.\n"
        "Example: 'ss. 7 and 11(d)' → "
        "<fragment>s. 7</fragment> and <fragment>s. 11(d)</fragment>.\n"
        "- 'ss. 15, 7, 11(d), 11(f)' must produce four separate <fragment> tags."
    ),
}