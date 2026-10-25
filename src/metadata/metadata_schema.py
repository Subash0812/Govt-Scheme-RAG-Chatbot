from typing import List, Optional

from pydantic import BaseModel, Field


class SchemeMetadata(BaseModel):
    scheme_name: Optional[str] = None
    abbreviation: Optional[str] = None
    state: Optional[str] = None
    ministry: Optional[str] = None
    department: Optional[str] = None
    scheme_type: Optional[str] = None
    sponsorship: Optional[str] = None

    beneficiary_category: List[str] = Field(
        default_factory=list
    )

    application_mode: Optional[str] = None

    official_url: Optional[str] = None

    source_file: str