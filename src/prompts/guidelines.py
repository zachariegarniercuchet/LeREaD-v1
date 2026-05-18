ANNOTATION_GUIDELINES = """
        1. **Annotate every occurrence** of an authority, including abbreviated forms.
        2. Use the same `docid` for all mentions of the same authority within the same document.
        3. Apply a general tag first, then apply specific tags inside it.
        4. Consider fragments as part of the same mention only if they are ≤ 2 stop-words away from the title/citation/source. Otherwise, create a new mention with the same `docid`.

        ### Global Annotation Structure:

        Each mention should follow this structure:

        ```
        <auto_label labelname=AUTHORITY_TYPE docid="...">
            <auto_label labelname="title" titletype="official|alias">...</auto_label>
            <auto_label labelname="citation">...</auto_label>        (legislation/decisions only)
            <auto_label labelname="authors">...</auto_label>            (secondary sources only)
            <auto_label labelname="source">...</auto_label>              (secondary sources only)
            <auto_label labelname="fragment" fragmentid="..." non_standard="true|false">...</auto_label>
        </auto_label>
        ```

        Include only the elements that are present in the text.

        ---

        ### Specific Annotation Instructions:

        #### 1. Legislation:

        - **Wrap the entire mention** in `<auto_label labelname="legislation">`.
        - Assign a manual `docid` using a short, stable identifier.

        ##### Examples of `docid`:

        - IRPA → Immigration and Refugee Protection Act
        - Charter → Canadian Charter of Rights and Freedoms

        ##### Titles:

        - Tag the law name as `<auto_label labelname="title">`.
        - `titletype="official"` for full legal title
        - `titletype="alias"` for acronyms, short names, “the Act”

        ##### Citations:

        - Tag statute citations as `<auto_label labelname="citation">`.

        ##### Fragments:

        - Use standard fragment IDs (e.g., `sec 1`, `subsec 1(1)`) or set `non_standard=true` for non-standard formats.

        #### 2. Decisions:

        - **Wrap each mention** in `<auto_label labelname="decision">`.
        - Assign a short, distinctive `docid`.

        ##### Titles:

        - Use `<auto_label labelname="title">` with `titletype="official"` for full citation and `titletype="alias"` for short form.

        ##### Citations:

        - Each citation is tagged as its own `<auto_label labelname="citation">`.
        ##### Fragments:

        - Use `p` for pages and `para` for paragraphs, including both bounds for intervals.

        #### 3. Secondary Sources:

        - **Wrap each mention** in `<auto_label labelname="secondary_sources">`.
        - `docid` should be a short, distinctive title.

        ##### Titles:

        - Tag with `<auto_label labelname="title">` for books, articles, or contributions.

        ##### Authors:

        - Include all authors in one `<auto_label labelname="authors">` tag.
        ##### Source:

        - Tag journal/publisher/year/volume/starting page in `<auto_label labelname="source">`.

        ##### Fragments:

        - Use formats like `p 127` or `no 1958` for pages and paragraphs.

        #### 4. Unable to Classify:

        - Use `<auto_label labelname="unable_to_classify">` only if the source is unclear or borderline.
        - Assign a `docid` regardless.

        ---

        ### Edge Cases and Error Handling:

        - Annotate separately if laws incorporate other laws.
        - Each constitutional text must be distinct under `<auto_label labelname="legislation">`.
        - Different courts for the same parties should have different `docid`s.
        - When encountering non-standard fragments, clearly reproduce the format.

        """