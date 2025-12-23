"""Shared Pydantic schemas for GreenBond-RAG outputs."""

from typing import List, Optional

from pydantic import BaseModel, Field


class AuditTrail(BaseModel):
    source_document: str = Field(..., description="Title or filename of the source document.")
    page_number: Optional[int] = Field(None, description="Page number where the content was retrieved.")
    snippet: str = Field(..., description="Exact retrieved text or table cell content.")


class IssueDetails(BaseModel):
    isin: Optional[str] = Field(None, description="Bond ISIN identifier.")
    tenor_years: Optional[float] = Field(None, description="Tenor in years.")
    coupon_rate: Optional[float] = Field(None, description="Coupon rate as percentage.")
    size_million: Optional[float] = Field(None, description="Issue size in million (currency-agnostic).")
    currency: Optional[str] = Field(None, description="Currency code, e.g., USD, EUR.")


class ESGAlignment(BaseModel):
    sdg_alignment: List[str] = Field(default_factory=list, description="Relevant SDG codes, e.g., SDG7, SDG11.")
    eu_taxonomy_status: Optional[str] = Field(
        None, description="Eligible, Aligned, or Not Aligned per EU Taxonomy."
    )
    verification_status: Optional[str] = Field(
        None, description="Second-party opinion or assurance status (e.g., SPO positive/qualified)."
    )


class KPI(BaseModel):
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    methodology: Optional[str] = None
    peer_percentile: Optional[float] = Field(
        None, description="Relative performance vs peers (0-100)."
    )
    audit: Optional[AuditTrail] = None


class StandardizedBondInformationCard(BaseModel):
    """Default LLM response schema for bond-level answers."""

    issuer: Optional[str] = Field(None, description="Issuing entity name.")
    objective: Optional[str] = Field(None, description="Use of proceeds / sustainability objective.")
    location: Optional[str] = Field(None, description="Project geography.")
    developer: Optional[str] = Field(None, description="Developer or operator of the project.")
    taxonomy_category: Optional[str] = Field(
        None, description="Mapped taxonomy category (renewable energy, green buildings, etc.)."
    )
    issue_details: IssueDetails = Field(default_factory=IssueDetails)
    esg_alignment: ESGAlignment = Field(default_factory=ESGAlignment)
    kpis: List[KPI] = Field(default_factory=list, description="Standardized KPIs with methodology linkage.")
    greenwashing_score: Optional[float] = Field(
        None, description="GreenImplement ratio from verification (0-1)."
    )
    alerts: List[str] = Field(
        default_factory=list, description="Missing reports, ambiguous claims, or data quality flags."
    )
    audit_trail: List[AuditTrail] = Field(
        default_factory=list, description="All snippets used to build the card."
    )


__all__ = ["AuditTrail", "IssueDetails", "ESGAlignment", "KPI", "StandardizedBondInformationCard"]



