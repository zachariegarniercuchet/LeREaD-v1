import html
import json
import copy


def normalize_docid(docid: str) -> str:
    if docid is None:
        return None
    return html.unescape(docid).strip()

class ReferenceProfile:
    def __init__(self, doctype, jurisdiction: str = None, docid=None, first_seen_id=None):
        self.doc_type = doctype
        self.jurisdiction = jurisdiction
        self.docid = normalize_docid(docid)
        self.main_title = None

        # Each of these is a dict mapping {value: first_seen_id}, i.e. the
        # id of the <manual_label>/<auto_label> tag whose mention first
        # introduced that specific value into this profile. This lets you
        # know, for any mention id X, exactly which alt titles / citations /
        # fragments / authors already existed *before* X — not just whether
        # the profile as a whole existed.
        self.alternative_titles = {}
        self.citations = {}
        self.fragments_mentioned = {}
        self.authors = {}  # Only for secondary sources

        # id (attribute) of the <manual_label>/<auto_label> tag that first
        # caused this profile to be created in a registry. Set once, never
        # overwritten afterwards.
        self.first_seen_id = first_seen_id

    def add_alternative_title(self, title, mention_id=None):
        if title not in self.alternative_titles:
            self.alternative_titles[title] = mention_id

        if len(self.alternative_titles.keys()) > 0 and self.main_title is None:
            self.main_title = list(self.alternative_titles.keys())[0]

    def add_citation(self, citation, mention_id=None):
        if citation not in self.citations:
            self.citations[citation] = mention_id

    def add_fragment_mentioned(self, fragment, mention_id=None):
        if fragment not in self.fragments_mentioned:
            self.fragments_mentioned[fragment] = mention_id

    def add_author(self, author, mention_id=None):
        if author not in self.authors:
            self.authors[author] = mention_id

    def to_string(self, attributes=None):
        if attributes is None:
            attributes = ["doc_type", "jurisdiction", "docid",
                          "alternative_titles", "citations", "fragments_mentioned",
                          "authors", "first_seen_id"]
        return {attr: getattr(self, attr) for attr in attributes}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def to_dict(self, attributes=None) -> dict:
        """Return a plain-dict representation, safe for json.dumps()."""
        if attributes is None:
            attributes = ["doc_type", "jurisdiction", "docid",
                          "alternative_titles", "citations", "fragments_mentioned",
                          "authors", "first_seen_id"]
        return {attr: getattr(self, attr) for attr in attributes}

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceProfile":
        """Reconstruct a ReferenceProfile from a dict produced by to_dict()."""
        profile = cls(
            doctype=data.get("doc_type"),
            jurisdiction=data.get("jurisdiction"),
            docid=data.get("docid"),
            first_seen_id=data.get("first_seen_id"),
        )
        profile.alternative_titles = cls._coerce_tracked_dict(data.get("alternative_titles"))
        profile.citations = cls._coerce_tracked_dict(data.get("citations"))
        profile.fragments_mentioned = cls._coerce_tracked_dict(data.get("fragments_mentioned"))
        profile.authors = cls._coerce_tracked_dict(data.get("authors"))
        return profile

    @staticmethod
    def _coerce_tracked_dict(value) -> dict:
        """
        Accepts either the new {value: first_seen_id} dict shape, or the old
        flat list-of-strings shape (for backward compatibility with caches
        produced before per-item tracking existed), and always returns a
        dict.
        """
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, list):
            # legacy format: list of bare strings, no first_seen_id known
            return {item: None for item in value}
        return {}

    def to_json(self, **kwargs) -> str:
        """Serialize this profile directly to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)


class ReferenceProfileRegistry:
    def __init__(self):
        self.profiles = []
        self._profiles_by_docid = {}

    def add_profile(self, profile: ReferenceProfile):
        self.profiles.append(profile)
        if profile.docid is not None:
            norm_id = normalize_docid(profile.docid)
            self._profiles_by_docid[norm_id] = profile



    def get_profile_by_docid(self, docid):
        return self._profiles_by_docid.get(normalize_docid(docid))

    
    def _filter_registry_before(self, mention_id: int) -> "ReferenceProfileRegistry":
        """
        Return a new ReferenceProfileRegistry containing only the information
        that existed strictly before `mention_id`.
        """
        tracked_fields = (
            "alternative_titles",
            "citations",
            "fragments_mentioned",
            "authors",
        )

        filtered_registry = ReferenceProfileRegistry()

        for profile in self.profiles:
            profile_first_seen = profile.first_seen_id

            # Profile didn't exist yet.
            if profile_first_seen is not None and int(profile_first_seen) >= mention_id:
                continue

            # Clone the profile
            new_profile = ReferenceProfile(
                doctype=profile.doc_type,
                jurisdiction=profile.jurisdiction,
                docid=profile.docid,
                first_seen_id=profile.first_seen_id,
            )

            # Filter each tracked field
            for field in tracked_fields:
                values = getattr(profile, field)

                setattr(
                    new_profile,
                    field,
                    {
                        value: seen_id
                        for value, seen_id in values.items()
                        if seen_id is None or int(seen_id) < mention_id
                    },
                )

            filtered_registry.add_profile(new_profile)

        return filtered_registry
    
    def _filter_by_doctype(self, doctype):
        filtered_registry = ReferenceProfileRegistry()
        for profile in self.profiles:
            if profile.doc_type == doctype:
                filtered_registry.add_profile(profile)
        return filtered_registry

    def update_from_mention(self, mention) -> ReferenceProfile | None:
        """
        Update the registry from a single ReferenceMention (one
        <manual_label>/<auto_label> tag).

        If this mention causes a brand-new ReferenceProfile to be created
        (i.e. its docid wasn't already in the registry), that profile's
        `first_seen_id` is stamped with this mention's tag id.

        Independently, every individual alternative title / citation /
        fragment / author introduced by this mention is stamped with this
        same mention id, *only if it's the first time that exact value is
        seen for this profile*. Values already present (from an earlier
        mention) are left with their original first_seen_id.

        Returns the ReferenceProfile that was created/updated, or None if
        the mention couldn't be parsed into a profile.
        """
        from src.extractor import parse_full_html

        full_html = mention.html_str
        if not full_html:
            return None

        parsed = parse_full_html(full_html)
        if parsed is None:
            return None

        docid = parsed["docid"] or getattr(mention, "docid", None)

        mention_id = mention.html_tag.attributes.get("id")

        profile = self.get_profile_by_docid(docid)

        if profile is None:
            profile = ReferenceProfile(
                doctype=parsed["doc_type"],
                docid=docid,
                first_seen_id=mention_id,
            )
            self.add_profile(profile)
        else:
            if profile.doc_type is None and parsed["doc_type"] is not None:
                profile.doc_type = parsed["doc_type"]

        for alt_title in parsed["alternative_titles"]:
            profile.add_alternative_title(alt_title, mention_id)
        for citation in parsed["citations"]:
            profile.add_citation(citation, mention_id)
        for fragment in parsed["fragments_mentioned"]:
            profile.add_fragment_mentioned(fragment, mention_id)
        for author in parsed["authors"]:
            profile.add_author(author, mention_id)

        return profile

    def replace_docid_with_main_title(self):
        """
        For every profile in the registry, replace its `docid` with its
        `main_title` (in place). If `main_title` is None, the original `docid`
        is kept as a fallback. The registry's internal docid index is rebuilt
        afterwards so lookups via get_profile_by_docid stay consistent.
        Returns a new ReferenceProfileRegistry with the updated profiles."""
        new_registry = copy.copy(self)
        for profile in new_registry.profiles:
            if profile.main_title is not None:
                profile.docid = normalize_docid(profile.main_title)
            # else: keep the existing docid as-is (fallback)

        # Rebuild the docid -> profile index since docids changed
        new_registry._profiles_by_docid = {}
        for profile in new_registry.profiles:
            if profile.docid is not None:
                new_registry._profiles_by_docid[normalize_docid(profile.docid)] = profile

        return new_registry



    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceProfileRegistry":
        """Reconstruct a ReferenceProfileRegistry from a dict produced by to_dict()."""
        registry = cls()
        for profile_data in data.get("profiles", []):
            registry.add_profile(ReferenceProfile.from_dict(profile_data))
        return registry
    
    def to_dict(self, attributes=None) -> dict:
        """Return a plain-dict representation, safe for json.dumps()."""
        if attributes is None:
            attributes = ["doc_type", "jurisdiction", "docid",
                          "alternative_titles", "citations", "fragments_mentioned",
                          "authors", "first_seen_id"]
        return {
            "profiles": [profile.to_dict(attributes=attributes) for profile in self.profiles]
        }

    def to_json(self, **kwargs) -> str:
        """Serialize the whole registry directly to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    def to_string(self, attributes=None):
        if attributes is None:
            attributes = ["doc_type", "jurisdiction", "docid",
                          "alternative_titles", "citations", "fragments_mentioned",
                          "authors", "first_seen_id"]
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
    on a ReferenceProfile/ReferenceProfileRegistry (or anything nesting them)
    without manually calling to_dict() first.
    """
    def default(self, obj):
        if isinstance(obj, (ReferenceProfile, ReferenceProfileRegistry)):
            return obj.to_dict()
        return super().default(obj)