"""
Pydantic models for SDGZero business data.
Maps fields from /wp-json/geodir/v2/businesses API response.
"""

import re
from html import unescape
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


def strip_html(text: str) -> str:
    """Remove HTML tags and clean up whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_mojibake(text: str) -> str:
    """
    Fix cp1252-decoded UTF-8 strings (mojibake).

    Happens when the API response is UTF-8 but was decoded as Windows-1252,
    e.g. '≤' → 'â‰¤'. Detect by re-encoding as cp1252 and decoding as UTF-8.
    Falls back to the original string if re-encoding fails.
    """
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class Business(BaseModel):
    """Core business data model."""

    id: int
    slug: str
    name: str = Field(alias="title")
    url: str = Field(alias="link")
    scraped_at: Optional[str] = Field(None, alias="modified")  # last modified date

    # Contact & location
    phone: Optional[str] = None
    website: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    zip: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Business info
    business_type: Optional[str] = None      # B2B / B2C / Both
    job_sector: Optional[str] = None         # Private / Public / Third Sector
    company_size: Optional[str] = None       # SME / Large etc.
    package_id: Optional[int] = None         # membership tier (1=basic, 9=premium)
    claimed: Optional[str] = None            # Yes/No — verified profile
    founder_name: Optional[str] = Field(None, alias="founder__signatoryname")

    # Categories (e.g. "Energy & Renewables", "Technology & Digital")
    categories: Optional[list] = Field(None, alias="post_category")

    # Social media (useful for Outreach Agent)
    linkedin: Optional[str] = None
    facebook: Optional[str] = None
    twitter: Optional[str] = None
    tiktok: Optional[str] = None
    instagram: Optional[str] = None
    video: Optional[str] = None
    logo: Optional[str] = None

    # Text content (used for embedding)
    content: Optional[str] = None           # full description ~2000 chars — 100% filled
    summary: Optional[str] = Field(None, alias="snippet_about_the_company2")
    achievements_summary: Optional[str] = None
    sdg_involvement: Optional[str] = Field(None, alias="sdg_involvement_summary2")

    # SDG data — source: post_tags (canonical WordPress taxonomy, has id/name/slug)
    # sdg_live is a redundant mirror of post_tags names; we use post_tags as primary.
    sdg_tags: Optional[list] = Field(None, alias="post_tags")    # SDG goal names, e.g. ["Climate Action", ...]
    sdg_slugs: Optional[list] = None                              # SDG goal slugs, e.g. ["climate-action", ...]
    membership_tier: Optional[list] = Field(None, alias="sdg_badges")  # Ambassador / Strategic Partner etc.

    # Ratings
    rating: Optional[float] = None
    rating_count: Optional[int] = None

    model_config = {"populate_by_name": True}

    @field_validator(
        "name", "business_type", "job_sector", "company_size", "claimed",
        "summary", "achievements_summary", "founder_name",
        "phone", "website", "street", "city", "region", "country", "zip",
        "linkedin", "facebook", "twitter", "tiktok", "instagram", "video",
        mode="before",
    )
    @classmethod
    def extract_rendered_string(cls, v):
        """Many API fields return {"rendered": "...", "raw": "..."} — extract rendered."""
        if isinstance(v, dict):
            val = v.get("rendered") or v.get("raw")
            if not val or val in ("No", "Select Sector", "Never"):
                return None
            return fix_mojibake(unescape(str(val)))
        if isinstance(v, str):
            return fix_mojibake(unescape(v)) if v else None
        return v

    @field_validator("content", "sdg_involvement", "achievements_summary", mode="before")
    @classmethod
    def extract_and_strip_html(cls, v):
        """Fields containing HTML — extract rendered then strip tags."""
        if isinstance(v, dict):
            val = v.get("rendered") or v.get("raw")
            if not val:
                return None
            return strip_html(str(val)) or None
        if isinstance(v, str):
            return strip_html(v) or None
        return v

    @field_validator("rating", mode="before")
    @classmethod
    def parse_rating(cls, v):
        """API returns rating as a string (e.g. '0', '4.5') — coerce to float."""
        try:
            return float(v) if v else 0.0
        except (TypeError, ValueError):
            return 0.0

    @field_validator("logo", mode="before")
    @classmethod
    def extract_logo_url(cls, v):
        """Logo field format: 'url|width|height|' — extract just the URL."""
        if isinstance(v, str) and v:
            return v.split("|")[0] or None
        return None

    @field_validator("sdg_tags", mode="before")
    @classmethod
    def extract_sdg_names_from_post_tags(cls, v):
        """post_tags is a list of {id, name, slug} dicts — extract and unescape names only."""
        if isinstance(v, list):
            names = [unescape(t["name"]) for t in v if isinstance(t, dict) and t.get("name")]
            return names or None
        return None

    @field_validator("membership_tier", mode="before")
    @classmethod
    def extract_membership_tier(cls, v):
        """sdg_badges returns {"raw": "...", "rendered": [...]} — extract rendered list."""
        if isinstance(v, dict):
            rendered = v.get("rendered")
            if isinstance(rendered, list):
                return [s for s in rendered if s] or None
            raw = v.get("raw")
            if raw:
                return [s.strip() for s in str(raw).split(",") if s.strip()] or None
            return None
        return None

    @model_validator(mode="before")
    @classmethod
    def extract_sdg_slugs(cls, data):
        """Derive sdg_slugs from post_tags before field validation."""
        post_tags = data.get("post_tags") or []
        if isinstance(post_tags, list):
            slugs = [t["slug"] for t in post_tags if isinstance(t, dict) and t.get("slug")]
            data["sdg_slugs"] = slugs or None
        return data

    @field_validator("categories", mode="before")
    @classmethod
    def extract_category_names(cls, v):
        """post_category is a list of {id, name, slug} objects — extract and unescape names."""
        if isinstance(v, list):
            return [unescape(c["name"]) for c in v if isinstance(c, dict) and c.get("name")] or None
        return None

    def to_embedding_text(self) -> str:
        """
        Combine all text fields into one string for embedding.
        Order: identity → location → categories → content → SDG info
        """
        parts = [
            f"Company: {self.name}",
            f"Categories: {', '.join(self.categories)}" if self.categories else "",
            f"Sector: {self.job_sector}" if self.job_sector else "",
            f"City: {self.city}" if self.city else "",
            f"Country: {self.country}" if self.country else "",
            f"Description: {self.content}" if self.content else "",
            f"Summary: {self.summary}" if self.summary else "",
            f"Achievements: {self.achievements_summary}" if self.achievements_summary else "",
            f"SDG involvement: {self.sdg_involvement}" if self.sdg_involvement else "",
            f"SDGs: {', '.join(self.sdg_tags)}" if self.sdg_tags else "",
        ]
        return "\n".join(p for p in parts if p)

    def to_metadata(self) -> dict:
        """
        Flat metadata dict for ChromaDB.
        Values must be str, int, float, or bool — no None, no nested objects.
        """
        return {k: v for k, v in {
            "id": self.id,
            "slug": self.slug,
            "name": self.name,
            "url": self.url,
            "scraped_at": self.scraped_at or "",
            # location
            "street": self.street or "",
            "zip": self.zip or "",
            "city": self.city or "",
            "region": self.region or "",
            "country": self.country or "",
            "latitude": self.latitude or 0.0,
            "longitude": self.longitude or 0.0,
            # contact
            "phone": self.phone or "",
            "website": self.website or "",
            "linkedin": self.linkedin or "",
            "facebook": self.facebook or "",
            "twitter": self.twitter or "",
            "tiktok": self.tiktok or "",
            "instagram": self.instagram or "",
            "logo": self.logo or "",
            # business info
            "business_type": self.business_type or "",
            "job_sector": self.job_sector or "",
            "company_size": self.company_size or "",
            "package_id": self.package_id or 1,
            "claimed": self.claimed or "",
            "founder_name": self.founder_name or "",
            "categories": ", ".join(self.categories) if self.categories else "",
            # sdg
            "sdg_tags": ", ".join(self.sdg_tags) if self.sdg_tags else "",
            "sdg_slugs": ", ".join(self.sdg_slugs) if self.sdg_slugs else "",
            "membership_tier": ", ".join(self.membership_tier) if self.membership_tier else "",
            # ratings
            "rating": self.rating or 0.0,
            "rating_count": self.rating_count or 0,
        }.items() if v is not None}
