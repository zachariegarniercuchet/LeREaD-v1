import json


class ReferenceProfile:
    def __init__(self, doctype, jurisdiction:str=None, main_title=None, docid=None):
        self.doc_type = doctype
        self.jurisdiction = jurisdiction
        self.main_title = main_title
        self.docid = docid
        self.alternative_titles = []
        self.citations = []
        self.fragments_mentioned = []
        self.authors = [] #Only for secondary sources

    def add_alternative_title(self, title):
        self.alternative_titles.append(title)
    
    def add_citation(self, citation):
        self.citations.append(citation)
    
    def add_fragment_mentioned(self, fragment):
        self.fragments_mentioned.append(fragment)
    
    def add_author(self, author):
        self.authors.append(author)

    def to_string(self, attributes=None):
        if attributes is None:
            attributes = ["doc_type", "jurisdiction", "main_title", "docid", "alternative_titles", "citations", "fragments_mentioned", "authors"]
        return {attr: getattr(self, attr) for attr in attributes}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
            return self.__str__()

    def to_dict(self, attributes=None) -> dict:
        """Return a plain-dict representation, safe for json.dumps()."""
        if attributes is None:
            attributes = ["doc_type", "jurisdiction", "main_title", "docid", "alternative_titles", "citations", "fragments_mentioned", "authors"]
        return {attr: getattr(self, attr) for attr in attributes}
 

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceProfile":
        """Reconstruct a ReferenceProfile from a dict produced by to_dict()."""
        profile = cls(
            doctype=data.get("doc_type"),
            jurisdiction=data.get("jurisdiction"),
            main_title=data.get("main_title"),
            docid=data.get("docid"),
        )
        profile.alternative_titles = list(data.get("alternative_titles", []))
        profile.citations = list(data.get("citations", []))
        profile.fragments_mentioned = list(data.get("fragments_mentioned", []))
        profile.authors = list(data.get("authors", []))
        return profile

    def to_json(self, **kwargs) -> str:
        """Serialize this profile directly to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)
    


class ReferenceProfileList:
    def __init__(self):
        self.profiles = []
        self._profiles_by_docid = {} 

    def add_profile(self, profile: ReferenceProfile):
        self.profiles.append(profile)
        if profile.docid is not None:
            self._profiles_by_docid[profile.docid] = profile

    def get_profile_by_docid(self, docid):
        return self._profiles_by_docid.get(docid)

    def update_from_annotations(self, annotations: list):
        from src.extractor import parse_full_html

        for annotation in annotations:
            full_html = annotation.get("full_html")
            if not full_html:
                continue

            parsed = parse_full_html(full_html)
            if parsed is None:
                continue

            docid = parsed["docid"] or annotation.get("docid")
            if docid is None:
                continue

            profile = self.get_profile_by_docid(docid)

            if profile is None:
                profile = ReferenceProfile(
                    doctype=parsed["doc_type"],
                    main_title=parsed["main_title"],
                    docid=docid,
                )
                self.add_profile(profile)
            else:
                if profile.main_title is None and parsed["main_title"] is not None:
                    profile.main_title = parsed["main_title"]
                if profile.doc_type is None and parsed["doc_type"] is not None:
                    profile.doc_type = parsed["doc_type"]

            for alt_title in parsed["alternative_titles"]:
                if alt_title not in profile.alternative_titles:
                    profile.add_alternative_title(alt_title)
            for citation in parsed["citations"]:
                if citation not in profile.citations:
                    profile.add_citation(citation)
            for fragment in parsed["fragments_mentioned"]:
                if fragment not in profile.fragments_mentioned:
                    profile.add_fragment_mentioned(fragment)
            for author in parsed["authors"]:
                if author not in profile.authors:
                    profile.add_author(author)


    def to_dict(self, attributes=None) -> dict:
        """Return a plain-dict representation, safe for json.dumps()."""
        return {"profiles": [profile.to_dict(attributes=attributes) for profile in self.profiles]}

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceProfileList":
        """Reconstruct a ReferenceProfileList from a dict produced by to_dict()."""
        rpl = cls()
        for profile_data in data.get("profiles", []):
            rpl.add_profile(ReferenceProfile.from_dict(profile_data))
        return rpl

    def to_json(self, **kwargs) -> str:
        """Serialize the whole list directly to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)
    
    def to_string(self, attributes=None):
        if attributes is None:
            attributes = ["doc_type", "jurisdiction", "main_title", "docid", "alternative_titles", "citations", "fragments_mentioned", "authors"]
        return str(self.to_dict(attributes=attributes))

    def __iter__(self):
        return iter(self.profiles)

    def __len__(self):
        return len(self.profiles)
    
    def __str__(self):
        return str(self.to_dict())


class ReferenceProfileJSONEncoder(json.JSONEncoder):
    """
    Lets you call json.dumps(obj, cls=ReferenceProfileJSONEncoder) directly
    on a ReferenceProfile/ReferenceProfileList (or anything nesting them)
    without manually calling to_dict() first.
    """
    def default(self, obj):
        if isinstance(obj, (ReferenceProfile, ReferenceProfileList)):
            return obj.to_dict()
        return super().default(obj)